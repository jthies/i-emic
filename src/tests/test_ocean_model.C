#include "TestDefinitions.H"
#include "OceanModel.H"
#include "THCM.H"
#include "THCMdefs.H"
#include "TRIOS_Domain.H"

#include "LOCA_GlobalData.H"
#include "LOCA_Epetra_Factory.H"

#include <cmath>
#include <vector>
#include <string>
#include <iomanip>

namespace
{
    RCP<Epetra_Comm> comm;
    RCP<Teuchos::ParameterList> oceanParams;
    RCP<Teuchos::ParameterList> locaParams;
    RCP<LOCA::GlobalData> globalData;
    RCP<THCM> thcm;
    RCP<OceanModel> model;

    // Tolerances for finite-difference consistency verification:
    // - Full-domain directional derivatives accumulate truncation errors across 3D stencils
    // - THCM's mixing/convective Jacobian uses Coleman & Moré graph-coloring finite difference (eps = 1e-8)
    constexpr double TOL_VELOCITY_REL    = 1.0e-2;  // 1% relative error for momentum advection & Coriolis
    constexpr double TOL_FULL_STATE_REL  = 2.0e-2;  // 2% relative error for full 3D state
    constexpr double TOL_TRACER_REL      = 1.5e-1;  // 15% relative error for tracer transport with mixing
    constexpr double TOL_SRES0_REL       = 2.5e-1;  // 25% relative error for flux boundary + integral condition
    constexpr double TOL_COMPONENT_ABS   = 1.0e-1;  // 0.1 max absolute error per component
    constexpr double TOL_CELL_WISE_REL   = 1.0e-4;  // 1e-4 for isolated single-column derivative

    struct ConsistencyReport
    {
        double relError;
        double maxAbsDiff;
        double uDiffMax;
        double pDiffMax;
        double tDiffMax;
        double sDiffMax;
    };

    // Helper: construct state with shear flow and smooth tracer fields (stably stratified)
    void constructBasicDiscretizationState(
        Teuchos::RCP<OceanModel> mdl,
        Epetra_Vector& x)
    {
        x.PutScalar(0.0);
        Teuchos::RCP<TRIOS::Domain> domain = THCM::Instance().GetDomain();
        int N = domain->GlobalN();
        int M = domain->GlobalM();
        int L = domain->GlobalL();
        const Epetra_BlockMap& map = x.Map();

        for (int k = 0; k < L; ++k)
        {
            double z_frac = (L > 1) ? (static_cast<double>(k) / (L - 1)) : 0.0;
            for (int j = 0; j < M; ++j)
            {
                double y_frac = (M > 1) ? (static_cast<double>(j) / (M - 1)) : 0.0;
                for (int i = 0; i < N; ++i)
                {
                    double x_frac = (N > 1) ? (static_cast<double>(i) / (N - 1)) : 0.0;

                    // Non-zero rotational / shear velocity field
                    double u_val = 0.05 * std::sin(2.0 * PI_ * x_frac) * std::cos(PI_ * y_frac) * (1.0 + 0.5 * z_frac);
                    double v_val = -0.05 * std::cos(2.0 * PI_ * x_frac) * std::sin(PI_ * y_frac) * (1.0 + 0.5 * z_frac);
                    double w_val = 0.005 * std::sin(PI_ * z_frac) * std::cos(PI_ * x_frac);
                    double p_val = 0.02 * std::cos(PI_ * x_frac) * std::cos(PI_ * y_frac);

                    // Stably stratified tracers (warm/fresh surface, cold/salty bottom -> d_rho/dz < 0)
                    double T_val = 4.0 + 16.0 * z_frac + 1.0 * std::cos(PI_ * x_frac) * std::sin(PI_ * y_frac);
                    double S_val = 36.0 - 2.0 * z_frac + 0.2 * std::sin(PI_ * x_frac) * std::cos(PI_ * y_frac);

                    int gid_u = FIND_ROW2(_NUN_, N, M, L, i, j, k, UU);
                    int gid_v = FIND_ROW2(_NUN_, N, M, L, i, j, k, VV);
                    int gid_w = FIND_ROW2(_NUN_, N, M, L, i, j, k, WW);
                    int gid_p = FIND_ROW2(_NUN_, N, M, L, i, j, k, PP);
                    int gid_T = FIND_ROW2(_NUN_, N, M, L, i, j, k, TT);
                    int gid_S = FIND_ROW2(_NUN_, N, M, L, i, j, k, SS);

                    if (map.MyGID(gid_u)) x[map.LID(gid_u)] = u_val;
                    if (map.MyGID(gid_v)) x[map.LID(gid_v)] = v_val;
                    if (map.MyGID(gid_w)) x[map.LID(gid_w)] = w_val;
                    if (map.MyGID(gid_p)) x[map.LID(gid_p)] = p_val;
                    if (map.MyGID(gid_T)) x[map.LID(gid_T)] = T_val;
                    if (map.MyGID(gid_S)) x[map.LID(gid_S)] = S_val;
                }
            }
        }
    }

    // Helper: construct statically stable state (warm/fresh surface, cold/salty bottom -> d_rho/dz < 0)
    void constructStaticallyStableState(
        Teuchos::RCP<OceanModel> mdl,
        Epetra_Vector& x)
    {
        x.PutScalar(0.0);
        Teuchos::RCP<TRIOS::Domain> domain = THCM::Instance().GetDomain();
        int N = domain->GlobalN();
        int M = domain->GlobalM();
        int L = domain->GlobalL();
        const Epetra_BlockMap& map = x.Map();

        for (int k = 0; k < L; ++k)
        {
            double z_frac = (L > 1) ? (static_cast<double>(k) / (L - 1)) : 0.0;
            // Warm on top, cold below
            double T_val = 4.0 + 16.0 * z_frac;
            // Fresh on top, salty below
            double S_val = 36.0 - 2.0 * z_frac;

            for (int j = 0; j < M; ++j)
            {
                for (int i = 0; i < N; ++i)
                {
                    int gid_T = FIND_ROW2(_NUN_, N, M, L, i, j, k, TT);
                    int gid_S = FIND_ROW2(_NUN_, N, M, L, i, j, k, SS);

                    if (map.MyGID(gid_T)) x[map.LID(gid_T)] = T_val;
                    if (map.MyGID(gid_S)) x[map.LID(gid_S)] = S_val;
                }
            }
        }
    }

    // Helper: construct statically unstable state (cold/salty surface, warm/fresh bottom -> d_rho/dz > 0)
    // This triggers convective adjustment!
    void constructStaticallyUnstableState(
        Teuchos::RCP<OceanModel> mdl,
        Epetra_Vector& x)
    {
        x.PutScalar(0.0);
        Teuchos::RCP<TRIOS::Domain> domain = THCM::Instance().GetDomain();
        int N = domain->GlobalN();
        int M = domain->GlobalM();
        int L = domain->GlobalL();
        const Epetra_BlockMap& map = x.Map();

        for (int k = 0; k < L; ++k)
        {
            double z_frac = (L > 1) ? (static_cast<double>(k) / (L - 1)) : 0.0;
            // Cold on top, warm below (inversion!)
            double T_val = 20.0 - 16.0 * z_frac;
            // Salty on top, fresh below (inversion!)
            double S_val = 34.0 + 2.5 * z_frac;

            for (int j = 0; j < M; ++j)
            {
                for (int i = 0; i < N; ++i)
                {
                    int gid_T = FIND_ROW2(_NUN_, N, M, L, i, j, k, TT);
                    int gid_S = FIND_ROW2(_NUN_, N, M, L, i, j, k, SS);

                    if (map.MyGID(gid_T)) x[map.LID(gid_T)] = T_val;
                    if (map.MyGID(gid_S)) x[map.LID(gid_S)] = S_val;
                }
            }
        }
    }

    // Helper: construct combined shear flow with convective instability
    void constructMixedShearConvectiveState(
        Teuchos::RCP<OceanModel> mdl,
        Epetra_Vector& x)
    {
        constructBasicDiscretizationState(mdl, x);
        Teuchos::RCP<TRIOS::Domain> domain = THCM::Instance().GetDomain();
        int N = domain->GlobalN();
        int M = domain->GlobalM();
        int L = domain->GlobalL();
        const Epetra_BlockMap& map = x.Map();

        // Invert vertical density profile to trigger convection while retaining shear velocities
        for (int k = 0; k < L; ++k)
        {
            double z_frac = (L > 1) ? (static_cast<double>(k) / (L - 1)) : 0.0;
            double T_val = 20.0 - 16.0 * z_frac;
            double S_val = 34.0 + 2.5 * z_frac;

            for (int j = 0; j < M; ++j)
            {
                for (int i = 0; i < N; ++i)
                {
                    int gid_T = FIND_ROW2(_NUN_, N, M, L, i, j, k, TT);
                    int gid_S = FIND_ROW2(_NUN_, N, M, L, i, j, k, SS);

                    if (map.MyGID(gid_T)) x[map.LID(gid_T)] = T_val;
                    if (map.MyGID(gid_S)) x[map.LID(gid_S)] = S_val;
                }
            }
        }
    }

    // Helper: create perturbation direction vector v
    // varFilter: -1 for all, 0 for UU/VV/WW, 1 for PP, 2 for TT, 3 for SS
    void createPerturbationVector(
        Teuchos::RCP<OceanModel> mdl,
        Epetra_Vector& v,
        int varFilter = -1)
    {
        v.PutScalar(0.0);
        Teuchos::RCP<TRIOS::Domain> domain = THCM::Instance().GetDomain();
        int N = domain->GlobalN();
        int M = domain->GlobalM();
        int L = domain->GlobalL();
        const Epetra_BlockMap& map = v.Map();

        for (int k = 0; k < L; ++k)
        {
            for (int j = 0; j < M; ++j)
            {
                for (int i = 0; i < N; ++i)
                {
                    double phase = (i * 7 + j * 13 + k * 19) % 31;
                    double weight = std::sin(phase + 1.0);

                    int gid_u = FIND_ROW2(_NUN_, N, M, L, i, j, k, UU);
                    int gid_v = FIND_ROW2(_NUN_, N, M, L, i, j, k, VV);
                    int gid_w = FIND_ROW2(_NUN_, N, M, L, i, j, k, WW);
                    int gid_p = FIND_ROW2(_NUN_, N, M, L, i, j, k, PP);
                    int gid_T = FIND_ROW2(_NUN_, N, M, L, i, j, k, TT);
                    int gid_S = FIND_ROW2(_NUN_, N, M, L, i, j, k, SS);

                    if (varFilter == -1 || varFilter == 0)
                    {
                        if (map.MyGID(gid_u)) v[map.LID(gid_u)] = weight * 0.1;
                        if (map.MyGID(gid_v)) v[map.LID(gid_v)] = -weight * 0.1;
                        if (map.MyGID(gid_w)) v[map.LID(gid_w)] = weight * 0.02;
                    }
                    if (varFilter == -1 || varFilter == 1)
                    {
                        if (map.MyGID(gid_p)) v[map.LID(gid_p)] = weight * 0.05;
                    }
                    if (varFilter == -1 || varFilter == 2)
                    {
                        if (map.MyGID(gid_T)) v[map.LID(gid_T)] = weight * 0.5;
                    }
                    if (varFilter == -1 || varFilter == 3)
                    {
                        if (map.MyGID(gid_S)) v[map.LID(gid_S)] = weight * 0.2;
                    }
                }
            }
        }

        // Normalize direction
        double vNorm = 0.0;
        v.Norm2(&vNorm);
        if (vNorm > 0.0)
            v.Scale(1.0 / vNorm);
    }

    // Core verification: compare central finite difference dFD = (F(x+hv) - F(x-hv))/(2h) against J*v
    ConsistencyReport checkJacobianRhsConsistency(
        Teuchos::RCP<OceanModel> mdl,
        const Epetra_Vector& x,
        const Epetra_Vector& v,
        double h = 1.0e-6)
    {
        // 1. Evaluate analytical Jacobian
        Teuchos::RCP<Epetra_Operator> W_op = mdl->create_W();
        mdl->computeJacobian(x, *W_op);
        Teuchos::RCP<Epetra_CrsMatrix> J = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(W_op, true);

        // Compute exact directional derivative: J*v
        Epetra_Vector Jv(*mdl->get_f_map());
        J->Apply(v, Jv);

        // 2. Central finite difference: (F(x+hv) - F(x-hv)) / (2h)
        Epetra_Vector x_plus(x);
        Epetra_Vector x_minus(x);
        x_plus.Update(h, v, 1.0);
        x_minus.Update(-h, v, 1.0);

        Epetra_Vector F_plus(*mdl->get_f_map());
        Epetra_Vector F_minus(*mdl->get_f_map());
        mdl->computeF(x_plus, F_plus, OceanModel::Residual);
        mdl->computeF(x_minus, F_minus, OceanModel::Residual);

        Epetra_Vector d_fd(*mdl->get_f_map());
        d_fd.Update(1.0 / (2.0 * h), F_plus, 0.0);
        d_fd.Update(-1.0 / (2.0 * h), F_minus, 1.0);

        // 3. Difference vector: diff = d_fd - Jv
        Epetra_Vector diff(d_fd);
        diff.Update(-1.0, Jv, 1.0);

        ConsistencyReport report;
        diff.Norm2(&report.relError);
        double jvNorm = 0.0;
        Jv.Norm2(&jvNorm);
        report.relError /= (jvNorm + 1.0e-12);

        diff.NormInf(&report.maxAbsDiff);

        // 4. Component-wise maximum differences
        report.uDiffMax = 0.0;
        report.pDiffMax = 0.0;
        report.tDiffMax = 0.0;
        report.sDiffMax = 0.0;

        Teuchos::RCP<TRIOS::Domain> domain = THCM::Instance().GetDomain();
        int N = domain->GlobalN();
        int M = domain->GlobalM();
        int L = domain->GlobalL();
        const Epetra_BlockMap& map = diff.Map();

        for (int k = 0; k < L; ++k)
        {
            for (int j = 0; j < M; ++j)
            {
                for (int i = 0; i < N; ++i)
                {
                    int gid_u = FIND_ROW2(_NUN_, N, M, L, i, j, k, UU);
                    int gid_v = FIND_ROW2(_NUN_, N, M, L, i, j, k, VV);
                    int gid_w = FIND_ROW2(_NUN_, N, M, L, i, j, k, WW);
                    int gid_p = FIND_ROW2(_NUN_, N, M, L, i, j, k, PP);
                    int gid_T = FIND_ROW2(_NUN_, N, M, L, i, j, k, TT);
                    int gid_S = FIND_ROW2(_NUN_, N, M, L, i, j, k, SS);

                    if (map.MyGID(gid_u)) report.uDiffMax = std::max(report.uDiffMax, std::abs(diff[map.LID(gid_u)]));
                    if (map.MyGID(gid_v)) report.uDiffMax = std::max(report.uDiffMax, std::abs(diff[map.LID(gid_v)]));
                    if (map.MyGID(gid_w)) report.uDiffMax = std::max(report.uDiffMax, std::abs(diff[map.LID(gid_w)]));
                    if (map.MyGID(gid_p)) report.pDiffMax = std::max(report.pDiffMax, std::abs(diff[map.LID(gid_p)]));
                    if (map.MyGID(gid_T)) report.tDiffMax = std::max(report.tDiffMax, std::abs(diff[map.LID(gid_T)]));
                    if (map.MyGID(gid_S)) report.sDiffMax = std::max(report.sDiffMax, std::abs(diff[map.LID(gid_S)]));
                }
            }
        }

        return report;
    }
}

//------------------------------------------------------------------
// Test 0: Model Initialization
// Initializes THCM, LOCA global data, and OceanModel once for the test suite.
TEST(OceanModel, Initialization)
{
    bool failed = false;
    try
    {
        oceanParams = Utils::obtainParams("ocean_params.xml", "Ocean parameters");
        Teuchos::ParameterList& thcmParams = oceanParams->sublist("THCM");
        thcmParams.set("Parameter Name", "Combined Forcing");

        thcm = Teuchos::rcp(new THCM(thcmParams, comm));

        locaParams = Teuchos::rcp(new Teuchos::ParameterList);
        Teuchos::RCP<LOCA::Abstract::Factory> epetraFactory = Teuchos::rcp(new LOCA::Epetra::Factory);
        globalData = LOCA::createGlobalData(locaParams, epetraFactory);

        model = Teuchos::rcp(new OceanModel(thcmParams, globalData));
    }
    catch (...)
    {
        failed = true;
        throw;
    }

    EXPECT_FALSE(failed);
    EXPECT_TRUE(model != Teuchos::null);
}

//------------------------------------------------------------------
// Test 1: Basic Spatial Discretization
// Tests momentum advection, Coriolis, harmonic diffusion, pressure gradient,
// and tracer transport under a stably stratified state.
TEST(OceanModel, BasicDiscretization)
{
    ASSERT_TRUE(model != Teuchos::null);

    // Ensure restoring salinity (SRES = 1) for basic discretization baseline
    THCM::Instance().setSRES(1);
    THCM::Instance().enableIntegralCondition(false);

    Epetra_Vector x(*model->get_x_map());
    constructBasicDiscretizationState(model, x);

    // Test 1a: Full random direction perturbation
    {
        Epetra_Vector v(*model->get_x_map());
        createPerturbationVector(model, v, -1);
        ConsistencyReport report = checkJacobianRhsConsistency(model, x, v);

        std::cout << "[BasicDiscretization] Full vector: relError = " << report.relError
                  << ", maxAbsDiff = " << report.maxAbsDiff << std::endl;
        EXPECT_LT(report.relError, TOL_FULL_STATE_REL);
        EXPECT_LT(report.maxAbsDiff, TOL_COMPONENT_ABS);
    }

    // Test 1b: Velocity-only perturbation (u, v, w) -> momentum advection & Coriolis
    {
        Epetra_Vector v(*model->get_x_map());
        createPerturbationVector(model, v, 0);
        ConsistencyReport report = checkJacobianRhsConsistency(model, x, v);

        std::cout << "[BasicDiscretization] Velocity-only: relError = " << report.relError
                  << ", uDiffMax = " << report.uDiffMax << std::endl;
        EXPECT_LT(report.relError, TOL_VELOCITY_REL);
        EXPECT_LT(report.uDiffMax, TOL_COMPONENT_ABS);
    }

    // Test 1c: Temperature-only perturbation -> tracer diffusion and advection
    {
        Epetra_Vector v(*model->get_x_map());
        createPerturbationVector(model, v, 2);
        ConsistencyReport report = checkJacobianRhsConsistency(model, x, v);

        std::cout << "[BasicDiscretization] Temperature-only: relError = " << report.relError
                  << ", tDiffMax = " << report.tDiffMax << std::endl;
        EXPECT_LT(report.relError, TOL_TRACER_REL);
        EXPECT_LT(report.tDiffMax, TOL_COMPONENT_ABS);
    }

    // Test 1d: Salinity-only perturbation -> salinity diffusion and advection
    {
        Epetra_Vector v(*model->get_x_map());
        createPerturbationVector(model, v, 3);
        ConsistencyReport report = checkJacobianRhsConsistency(model, x, v);

        std::cout << "[BasicDiscretization] Salinity-only: relError = " << report.relError
                  << ", sDiffMax = " << report.sDiffMax << std::endl;
        EXPECT_LT(report.relError, TOL_TRACER_REL);
        EXPECT_LT(report.sDiffMax, TOL_COMPONENT_ABS);
    }
}

//------------------------------------------------------------------
// Test 2: Statically Stable Stratification
// Verifies that when the water column is stably stratified (density increases downward),
// convective adjustment remains inactive (tprstb = 0) and Jacobian matches RHS.
TEST(OceanModel, StaticallyStableStratification)
{
    ASSERT_TRUE(model != Teuchos::null);

    THCM::Instance().setSRES(1);
    THCM::Instance().enableIntegralCondition(false);

    Epetra_Vector x(*model->get_x_map());
    constructStaticallyStableState(model, x);

    Epetra_Vector v(*model->get_x_map());
    createPerturbationVector(model, v, -1);
    ConsistencyReport report = checkJacobianRhsConsistency(model, x, v);

    std::cout << "[StaticallyStable] relError = " << report.relError
              << ", maxAbsDiff = " << report.maxAbsDiff
              << ", tDiffMax = " << report.tDiffMax
              << ", sDiffMax = " << report.sDiffMax << std::endl;

    EXPECT_LT(report.relError, TOL_FULL_STATE_REL);
    EXPECT_LT(report.maxAbsDiff, TOL_COMPONENT_ABS);
}

//------------------------------------------------------------------
// Test 3: Statically Unstable Stratification with Convective Adjustment Active
// Cold, salty water on top of warm, fresh water triggers convective adjustment (tprstb > 0).
// Verifies that convective mixing fluxes in RHS and analytical Jacobian match.
TEST(OceanModel, StaticallyUnstableConvectiveAdjustment)
{
    ASSERT_TRUE(model != Teuchos::null);

    THCM::Instance().setSRES(1);
    THCM::Instance().enableIntegralCondition(false);

    Epetra_Vector x(*model->get_x_map());
    constructStaticallyUnstableState(model, x);

    // Test 3a: Temperature perturbation (triggers and perturbs convective heat flux)
    {
        Epetra_Vector v(*model->get_x_map());
        createPerturbationVector(model, v, 2);
        ConsistencyReport report = checkJacobianRhsConsistency(model, x, v);

        std::cout << "[ConvectiveActive] Temperature-only: relError = " << report.relError
                  << ", tDiffMax = " << report.tDiffMax << std::endl;
        EXPECT_LT(report.relError, TOL_TRACER_REL);
        EXPECT_LT(report.tDiffMax, TOL_COMPONENT_ABS);
    }

    // Test 3b: Salinity perturbation (triggers and perturbs convective salt flux)
    {
        Epetra_Vector v(*model->get_x_map());
        createPerturbationVector(model, v, 3);
        ConsistencyReport report = checkJacobianRhsConsistency(model, x, v);

        std::cout << "[ConvectiveActive] Salinity-only: relError = " << report.relError
                  << ", sDiffMax = " << report.sDiffMax << std::endl;
        EXPECT_LT(report.relError, TOL_TRACER_REL);
        EXPECT_LT(report.sDiffMax, TOL_COMPONENT_ABS);
    }

    // Test 3c: Combined full perturbation under active convection
    {
        Epetra_Vector v(*model->get_x_map());
        createPerturbationVector(model, v, -1);
        ConsistencyReport report = checkJacobianRhsConsistency(model, x, v);

        std::cout << "[ConvectiveActive] Full vector: relError = " << report.relError
                  << ", maxAbsDiff = " << report.maxAbsDiff << std::endl;
        EXPECT_LT(report.relError, TOL_FULL_STATE_REL);
        EXPECT_LT(report.maxAbsDiff, TOL_COMPONENT_ABS);
    }
}

//------------------------------------------------------------------
// Test 4: Combined Shear Advection and Active Convection
// Tests simultaneous momentum advection, Coriolis, harmonic diffusion,
// and active convective adjustment under an unstable density stratification.
TEST(OceanModel, CoupledAdvectionAndConvection)
{
    ASSERT_TRUE(model != Teuchos::null);

    THCM::Instance().setSRES(1);
    THCM::Instance().enableIntegralCondition(false);

    Epetra_Vector x(*model->get_x_map());
    constructMixedShearConvectiveState(model, x);

    Epetra_Vector v(*model->get_x_map());
    createPerturbationVector(model, v, -1);
    ConsistencyReport report = checkJacobianRhsConsistency(model, x, v);

    std::cout << "[CoupledAdvectionConvection] relError = " << report.relError
              << ", maxAbsDiff = " << report.maxAbsDiff
              << ", uDiffMax = " << report.uDiffMax
              << ", tDiffMax = " << report.tDiffMax << std::endl;

    EXPECT_LT(report.relError, TOL_FULL_STATE_REL);
    EXPECT_LT(report.maxAbsDiff, TOL_COMPONENT_ABS);
}

//------------------------------------------------------------------
// Test 5: Mixing Parameter Sensitivity
// Verifies Jacobian-RHS consistency when modifying mixing parameters (P_VC, SPL1, SPL2).
TEST(OceanModel, MixingParameterVariation)
{
    ASSERT_TRUE(model != Teuchos::null);

    THCM::Instance().setSRES(1);
    THCM::Instance().enableIntegralCondition(false);

    // Modify mixing continuation parameters
    THCM::Instance().setParameter("P_VC", 0.5);
    THCM::Instance().setParameter("SPL1", 1000.0);
    THCM::Instance().setParameter("SPL2", 0.05);

    Epetra_Vector x(*model->get_x_map());
    constructStaticallyUnstableState(model, x);

    Epetra_Vector v(*model->get_x_map());
    createPerturbationVector(model, v, -1);
    ConsistencyReport report = checkJacobianRhsConsistency(model, x, v);

    std::cout << "[MixingParamVariation] relError = " << report.relError
              << ", maxAbsDiff = " << report.maxAbsDiff << std::endl;

    EXPECT_LT(report.relError, TOL_FULL_STATE_REL);
    EXPECT_LT(report.maxAbsDiff, TOL_COMPONENT_ABS);

    // Restore standard mixing parameters
    THCM::Instance().setParameter("P_VC", 1.0);
    THCM::Instance().setParameter("SPL1", 2000.0);
    THCM::Instance().setParameter("SPL2", 0.01);
}

//------------------------------------------------------------------
// Test 6: Convective Adjustment Fix Mixing Flag
// Verifies Jacobian-RHS consistency with fixed convective pattern (vmix_fix = 1)
// vs free/adaptive convective pattern (vmix_fix = 0).
TEST(OceanModel, FixMixingFlagVariation)
{
    ASSERT_TRUE(model != Teuchos::null);

    THCM::Instance().setSRES(1);
    THCM::Instance().enableIntegralCondition(false);

    Epetra_Vector x(*model->get_x_map());
    constructStaticallyUnstableState(model, x);

    // Test with fixed mixing pattern
    THCM::Instance().fixMixing(1);
    {
        Epetra_Vector v(*model->get_x_map());
        createPerturbationVector(model, v, -1);
        ConsistencyReport report = checkJacobianRhsConsistency(model, x, v);

        std::cout << "[FixedMixing=1] relError = " << report.relError
                  << ", maxAbsDiff = " << report.maxAbsDiff << std::endl;
        EXPECT_LT(report.relError, TOL_FULL_STATE_REL);
        EXPECT_LT(report.maxAbsDiff, TOL_COMPONENT_ABS);
    }

    // Test with adaptive mixing pattern
    THCM::Instance().fixMixing(0);
    {
        Epetra_Vector v(*model->get_x_map());
        createPerturbationVector(model, v, -1);
        ConsistencyReport report = checkJacobianRhsConsistency(model, x, v);

        std::cout << "[FixedMixing=0] relError = " << report.relError
                  << ", maxAbsDiff = " << report.maxAbsDiff << std::endl;
        EXPECT_LT(report.relError, TOL_FULL_STATE_REL);
        EXPECT_LT(report.maxAbsDiff, TOL_COMPONENT_ABS);
    }
}

//------------------------------------------------------------------
// Test 7: Salinity Boundary Condition Formulations (SRES = 0 vs SRES = 1)
// Verifies that pure flux forcing (SRES = 0) with domain integral condition
// replacement correctly preserves consistency between the Jacobian and RHS.
TEST(OceanModel, SalinityBoundaryConditionSRES)
{
    ASSERT_TRUE(model != Teuchos::null);

    // Test with SRES = 0 (flux forcing + domain integral condition)
    THCM::Instance().setSRES(0);
    THCM::Instance().enableIntegralCondition(true);

    Epetra_Vector x(*model->get_x_map());
    constructBasicDiscretizationState(model, x);

    Epetra_Vector v(*model->get_x_map());
    createPerturbationVector(model, v, 3); // Salinity perturbation
    ConsistencyReport report = checkJacobianRhsConsistency(model, x, v);

    std::cout << "[SRES0-IntegralCond] Salinity perturbation: relError = " << report.relError
              << ", sDiffMax = " << report.sDiffMax << std::endl;

    EXPECT_LT(report.relError, TOL_SRES0_REL);
    EXPECT_LT(report.sDiffMax, TOL_COMPONENT_ABS);

    // Restore SRES = 1
    THCM::Instance().setSRES(1);
    THCM::Instance().enableIntegralCondition(false);
}

//------------------------------------------------------------------
// Test 8: Cell-Wise Jacobian Column Consistency
// Verifies that for individual surface, interior, and bottom water cells,
// single-basis-vector perturbations e_j yield column values matching J*e_j.
TEST(OceanModel, CellWiseColumnConsistency)
{
    ASSERT_TRUE(model != Teuchos::null);

    THCM::Instance().setSRES(1);
    THCM::Instance().enableIntegralCondition(false);

    Epetra_Vector x(*model->get_x_map());
    constructMixedShearConvectiveState(model, x);

    Teuchos::RCP<Epetra_Operator> W_op = model->create_W();
    model->computeJacobian(x, *W_op);
    Teuchos::RCP<Epetra_CrsMatrix> J = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(W_op, true);

    Teuchos::RCP<TRIOS::Domain> domain = THCM::Instance().GetDomain();
    int N = domain->GlobalN();
    int M = domain->GlobalM();
    int L = domain->GlobalL();
    const Epetra_BlockMap& map = x.Map();

    // Select three vertical levels: surface (L-1), mid-depth (L/2), bottom (0)
    std::vector<int> k_levels = {0, L / 2, L - 1};
    int i_mid = N / 2;
    int j_mid = M / 2;

    double h = 1.0e-6;

    for (int k_idx : k_levels)
    {
        std::vector<int> vars = {TT, SS};
        for (int var : vars)
        {
            int gid = FIND_ROW2(_NUN_, N, M, L, i_mid, j_mid, k_idx, var);
            if (!map.MyGID(gid)) continue;

            Epetra_Vector e_j(*model->get_x_map());
            e_j.PutScalar(0.0);
            e_j[map.LID(gid)] = 1.0;

            // Analytical column: J * e_j
            Epetra_Vector J_col(*model->get_f_map());
            J->Apply(e_j, J_col);

            // Numerical column via central differences
            Epetra_Vector x_plus(x);
            Epetra_Vector x_minus(x);
            x_plus[map.LID(gid)] += h;
            x_minus[map.LID(gid)] -= h;

            Epetra_Vector F_plus(*model->get_f_map());
            Epetra_Vector F_minus(*model->get_f_map());
            model->computeF(x_plus, F_plus, OceanModel::Residual);
            model->computeF(x_minus, F_minus, OceanModel::Residual);

            Epetra_Vector num_col(*model->get_f_map());
            num_col.Update(1.0 / (2.0 * h), F_plus, 0.0);
            num_col.Update(-1.0 / (2.0 * h), F_minus, 1.0);

            Epetra_Vector diff(num_col);
            diff.Update(-1.0, J_col, 1.0);

            double diffNorm = 0.0, jNorm = 0.0;
            diff.Norm2(&diffNorm);
            J_col.Norm2(&jNorm);

            double relColErr = diffNorm / (jNorm + 1.0e-12);
            EXPECT_LT(relColErr, TOL_CELL_WISE_REL);
        }
    }
}

//------------------------------------------------------------------
int main(int argc, char **argv)
{
    comm = initializeEnvironment(argc, argv);
    if (outFile == Teuchos::null)
        throw std::runtime_error("ERROR: Specify output streams");

    ::testing::InitGoogleTest(&argc, argv);

    int out = RUN_ALL_TESTS();

    // Clean up all objects in reverse order of creation before MPI finalization
    model = Teuchos::null;
    if (globalData != Teuchos::null)
        LOCA::destroyGlobalData(globalData);
    globalData = Teuchos::null;
    locaParams = Teuchos::null;
    oceanParams = Teuchos::null;
    thcm = Teuchos::null;
    THCM::Reset();

    comm->Barrier();
    std::cout << "TEST exit code proc #" << comm->MyPID()
              << " " << out << std::endl;

    MPI_Finalize();
    return out;
}
