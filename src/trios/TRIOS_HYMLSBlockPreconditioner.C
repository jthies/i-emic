#include "Teuchos_Utils.hpp"
#include <sstream>
#include "Epetra_Map.h"

#include "TRIOS_Macros.H"

#include "TRIOS_HYMLSBlockPreconditioner.H"
#include "HYMLS_Solver.hpp"
#include "HYMLS_Preconditioner.hpp"
#include "HYMLS_MainUtils.hpp"

#include "Epetra_Vector.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Import.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_LinearProblem.h"
#include "Epetra_RowMatrixTransposer.h"
#include "EpetraExt_MatrixMatrix.h"
#include "Teuchos_ParameterList.hpp"
#include <Teuchos_StrUtils.hpp>
#include "AztecOO.h"
#include "Epetra_Time.h"
#include "AztecOO_string_maps.h"
#include <iomanip>
#include "Teuchos_oblackholestream.hpp"

#include "Utils.H"

#include "TRIOS_SolverFactory.H"
#include "TRIOS_Domain.H"

#include "TRIOS_Static.H"

#include "THCMdefs.H"
#include "GlobalDefinitions.H"

/// define this to set P=I, just remove checkerboard pressure modes
//#define DUMMY_PREC 1

using Teuchos::rcp_dynamic_cast;

// this reordering of the T/S system was intended
// for testing direct solvers like SuperLU_dist,
// but it has not really been tested in practice
//#define LINEAR_ARHOMU_MAPS

// here are some macros controling which parts
// of the preconditioner should be employed.
// This is only for testing purposes!

#ifdef TESTING
// error tolerance when checking i.e. result of linear solves etc.
#define _TESTTOL_ 1e-14
#endif


namespace TRIOS {

///////////////////////////////////////////////////////////////////////////////
// CLASS OCEANPREC
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
// Constructor
///////////////////////////////////////////////////////////////////////////////

HYMLSBlockPreconditioner::HYMLSBlockPreconditioner(Teuchos::RCP<Epetra_CrsMatrix> jac,
                                         Teuchos::RCP<Domain> domain,
                                         Teuchos::ParameterList &List)
    :
    label_("Ocean Preconditioner"),
    jacobian(jac),
    domain(domain),
    needs_setup(true),
    IsComputed_(false)
{
    INFO("Create new ocean preconditioner...");
    comm = domain->GetComm();
    this->SetParameters(List);
    this->Setup1();
}

///////////////////////////////////////////////////////////////////////////////
// destructor
///////////////////////////////////////////////////////////////////////////////

HYMLSBlockPreconditioner::~HYMLSBlockPreconditioner()
{
    if (verbose>5)
    {
        INFO("Destroy ocean preconditioner...");
    }
}

// to be called in constructors:
void HYMLSBlockPreconditioner::Setup1()
{
    if (verbose > 5) INFO("Enter Setup1()...");
    // set pointer to domain map (map of vector x in y=A*x)
    // this preconditioner acts on standard vectors, i.e.
    // with 'nun' (6) unknowns per node
    domainMap = domain->GetSolveMap();

    // the map for y is the same
    rangeMap = domainMap;

    Teuchos::RCP<Epetra_Map> RowMap = domain->GetSolveMap();

    // column maps are maps with every grid point
    // present on every processor. in Epetra, the
    // column map of a matrix indicates which
    // column indices can possibly occur on a
    // subdomain. The reason why we have to make
    // column maps at all is that some of the
    // blocks map into different variables, i.e.
    // BuvTS: TS -> etc.
    Teuchos::RCP<Epetra_Map> ColMap = domain->GetColMap();

    // split up row map:
    if (verbose>5) INFO("$   Split main map into uv|w|p|TS maps...");

    //  DEBVAR(*RowMap);

    const int labelUV[4] = {UU, VV, WW, PP};
    const int labelTS[2] = {TT, SS};

    mapUV = Utils::CreateSubMap(*RowMap, dof_, labelUV, 4);
    mapTS = Utils::CreateSubMap(*RowMap, dof_, labelTS);

    // split up column map:

    if (verbose>5)
    {
        INFO("$   Build corresponding column maps...");
    }

    colmapUV = Utils::CreateSubMap(*ColMap, dof_, labelUV, 4);
    colmapTS = Utils::CreateSubMap(*ColMap, dof_, labelTS);

    if (verbose>5)
    {
        INFO("$   Create Importers ...");
    }

    // construct import objects for row maps:

    // note: the terminology of import objects is very confusing:
    // Epetra_Import(target_map,source_map) means that we wish to
    // import values from the target map into the source map. The
    // corresponding function call is A.Export(B,Importer,...),
    // which means that the values of A are replaced by corresponding
    // values in B. In this preconditioner class we use importers not
    // for getting non-local data but mainly to extract certain rows/columns
    // from matrices (a permutation operation which doesn't involve any
    // communication)

    // Note that the same operation could be done with Export objects, and/or by
    // calling 'Import' instead of 'Export'. To be honest, there is no clear line
    // in this file, it is basically trial and error for every new import/export.

    importUV = Teuchos::rcp(new Epetra_Import(*RowMap, *mapUV) );
    importTS = Teuchos::rcp(new Epetra_Import(*RowMap, *mapTS) );

    // will be constructed when the preconditioner is computed
    AuvSolver  = Teuchos::null;
    ATSSolver  = Teuchos::null;
    AuvPrecond = Teuchos::null;
    ATSPrecond = Teuchos::null;

    QTS = Teuchos::null;

    if (verbose>5)
    {
        // this is called at the end of any constructor, so this message makes sense:
        INFO("HYMLSBlockPreconditioner constructor completed.");
    }

// note: Once the Jacobian is available we still have to
// do some more setup, that's why needs_setup == true at this point
}

// build preconditioner. This function is called by NOX/LOCA,
// I am not sure what the arguments mean but we do not use them
// anyway
bool HYMLSBlockPreconditioner::computePreconditioner(const Epetra_Vector& x,
                                                Epetra_Operator& Prec,
                                                Teuchos::ParameterList* p)
{
    int ierr = this->Compute();
    return (ierr == 0);
}


///////////////////////////////////////////////////////////////////////////////
// private setup function. Can only be done once the actual Jacobian
// is there (or at least it's pattern)
///////////////////////////////////////////////////////////////////////////////

void HYMLSBlockPreconditioner::Setup2()
{
    if (verbose>5)
    {
        INFO("Enter Setup2()...");
    }

    // Set arrays with pointers. This makes accessing the objects
    // more convenient later on.
    INFO(" Set Pointer Arrays ...");

    // diagonal blocks:

    // Auv: uv->uv
    SubMatrixRowMap[_Auv]    = mapUV;
    SubMatrixColMap[_Auv]    = colmapUV;
    SubMatrixRangeMap[_Auv]  = mapUV;
    SubMatrixDomainMap[_Auv] = mapUV;
    Importer[_Auv]           = importUV;
    SubMatrixLabel[_Auv]     = "Auv";

    // ATS: TS->TS
    SubMatrixRowMap[_ATS]    = mapTS;
    SubMatrixColMap[_ATS]    = colmapTS;
    SubMatrixRangeMap[_ATS]  = mapTS;
    SubMatrixDomainMap[_ATS] = mapTS;
    Importer[_ATS]           = importTS;
    SubMatrixLabel[_ATS]     = "ATS";

    // B matrices

    // B_{uv} in TS equation: UV->TS
    SubMatrixRowMap[_BTSuv]    = mapTS;
    SubMatrixColMap[_BTSuv]    = colmapUV;
    SubMatrixRangeMap[_BTSuv]  = mapTS;
    SubMatrixDomainMap[_BTSuv] = mapUV;
    Importer[_BTSuv]           = importTS;
    SubMatrixLabel[_BTSuv]     = "BTSuv";

    // B_{TS} in uv equation TS->UV
    SubMatrixRowMap[_BuvTS]    = mapUV;
    SubMatrixColMap[_BuvTS]    = colmapTS;
    SubMatrixRangeMap[_BuvTS]  = mapUV;
    SubMatrixDomainMap[_BuvTS] = mapTS;
    Importer[_BuvTS]           = importUV;
    SubMatrixLabel[_BuvTS]     = "BuvTS";


    // allocate memory for submatrices
    for (int i=0;i<_NUMSUBM;i++)
    {
        SubMatrix[i] =
            Teuchos::rcp(new Epetra_CrsMatrix(Copy,
                                              *(SubMatrixRowMap[i]),
                                              *(SubMatrixColMap[i]),0));

        SubMatrix[i]->SetLabel(SubMatrixLabel[i].c_str());
    }

    INFO(" HYMLSBlockPreconditioner setup done.");

    // needs_setup is still kept 'true'. After the submatrices have been extracted
    // for the first time their column maps will be adjusted, THEN needs_setup will
    // be false.

}//Setup2()

///////////////////////////////////////////////////////////////////////////////
// splits the jacobian into the various subblocks
///////////////////////////////////////////////////////////////////////////////

void HYMLSBlockPreconditioner::extract_submatrices(const Epetra_CrsMatrix& Jac)
{
    if (verbose>5)
    {
        INFO("Extract submatrices..." << std::endl);
    }
    {

        // construct all submatrices
        for (int i = 0; i < _NUMSUBM; i++)
        {
            DEBUG("extract submatrix " << i);
            CHECK_ZERO(SubMatrix[i]->Export(Jac, *Importer[i], Zero));

            // the first time we do this we have to adjust the column maps
            if (needs_setup)
            {
                // at first the col maps contain all possible nodes. Once we know
                // what the submatrices look like (they always have the same structure)
                // replace the col map by one that contains only cols actually present
                // on the processor:
                CHECK_ZERO(SubMatrix[i]->FillComplete(*SubMatrixDomainMap[i],
                                                      *SubMatrixRangeMap[i]));
                    
                SubMatrixColMap[i] = Utils::CompressColMap(*SubMatrix[i]);
                SubMatrix[i]=Utils::ReplaceColMap(SubMatrix[i],*SubMatrixColMap[i]);
            }
            CHECK_ZERO(SubMatrix[i]->FillComplete(*SubMatrixDomainMap[i],
                                                  *SubMatrixRangeMap[i]));
            CHECK_ZERO(SubMatrix[i]->OptimizeStorage());
            //DEBVAR(*SubMatrix[i]);
        }
    }



    // all operations that have to be done once the Jacobian
    // is available have now been performed:
    needs_setup = false;

    // since we replace the matrices Auv and ATS by new ones (see next comment/commands),
    // there occurs a problem in ML (as of Trilinos 10), a segfault if we do not delete the
    // solver before the matrix. This is a bug in Trilinos and will probably be fixed soon.
    AuvPrecond=Teuchos::null;
    ATSPrecond=Teuchos::null;

    // Auv/ATS have to be Ifpack-safe. This is definitely
    // the case if we don't give any col map.
    // (I believe the 'LocalFilter' in Ifpack is buggy)
    Auv = Utils::RemoveColMap(SubMatrix[_Auv]);
    CHECK_ZERO(Auv->FillComplete(*mapUV,*mapUV));

    Epetra_Vector diag(*mapUV);
    CHECK_ZERO(Auv->ExtractDiagonalCopy(diag));
    for (int i = 0; i < diag.MyLength(); i++)
        if (mapUV->GID64(i) % dof_ == 2 && std::abs(diag[i]) < 1e-12) // W row
            diag[i] = 1e-12;
    CHECK_ZERO(Auv->ReplaceDiagonalValues(diag));

    Epetra_Vector left(*mapUV);
    Epetra_Vector right(*mapUV);
    left.PutScalar(1.0);
    right.PutScalar(1.0);

    int nx = domain->GlobalN();
    double dy = (domain->Ymax() - domain->Ymin()) / (double)nx;
    for (int i = 0; i < Auv->NumMyRows(); i++)
      {
      int gid = mapUV->GID(i);
      int j = (gid / dof_ / nx) % nx;
      double theta = domain->Ymin() + (j + 1.0) * dy;
      double theta2 = domain->Ymin() + (j + 0.5) * dy;
      if (gid % dof_ == 1)
        right[i] = 1. / cos(theta);
      if (gid % dof_ == 2)
        right[i] = 1. / cos(theta2);
      if (gid % dof_ == 0)
        left[i] = cos(theta);
      if (gid % dof_ == 3)
        left[i] = cos(theta2);
      }
    Auv->LeftScale(left);
    Auv->RightScale(right);

    DEBUG("Adjust diagonal block ATS...");
    ATS = Utils::RemoveColMap(SubMatrix[_ATS]);
    CHECK_ZERO(ATS->FillComplete(*mapTS,*mapTS));

//    DEBVAR(*Duv1);
    DEBUG("All submatrices have been extracted");
}

///////////////////////////////////////////////////////////////////////////////
// another setup function: build block systems, preconditioners and solvers
///////////////////////////////////////////////////////////////////////////////

void HYMLSBlockPreconditioner::build_preconditioner(void)
{
    TIMER_SCOPE("BlockPrec: build preconditioner");
    if (verbose>5)
    {
        INFO("Prepare preconditioner...");
    }

    bool rho_mixing = true; // TODO: for the moment this is hard-coded here
    bool rhomu = lsParams.get("ATS: rho/mu Transform", rho_mixing);
    if (rhomu) this->setup_rhomu();
    if (verbose>5)
    {
        INFO("*** Construct Krylov solvers...");
    }

    Teuchos::RCP<Teuchos::ParameterList> AuvSolverList =
        Teuchos::rcp(new Teuchos::ParameterList(lsParams.sublist("Auv Precond")));

    if (AuvPrecond == Teuchos::null)
    {
        TIMER_SCOPE("BlockPrec: HYMLS Compute");
        AuvPrecond = Teuchos::rcp(
            new HYMLS::Preconditioner(Auv, AuvSolverList,
                                      HYMLS::MainUtils::create_testvector(
                                          AuvSolverList->sublist("Problem"), *Auv)));
        // AuvSolver = Teuchos::rcp(new HYMLS::Solver(Auv, AuvPrecond, AuvSolverList));
        CHECK_ZERO(AuvPrecond->Compute());
    }

#ifdef DEBUGGING
    comm->Barrier();
#endif

    if (verbose>5)
    {
        INFO("*** Create Preconditioners...");
    }

    if (ATSSolver==Teuchos::null)
    {
        Teuchos::ParameterList& solverlist = lsParams.sublist("ATS Solver");
        ATSSolver = SolverFactory::CreateKrylovSolver(solverlist,verbose);
    }

    if (ATSSolver!=Teuchos::null)
    {
        if (rhomu)
        {
            CHECK_ZERO(ATSSolver->SetUserMatrix(Arhomu.get()));
        }
        else
        {
            CHECK_ZERO(ATSSolver->SetUserMatrix(ATS.get()));
        }
    }

    // ATS Precond has to be rebuilt. See comment for Auv Precond.
    {

        if (rhomu)
        {
            DEBUG("Create Preconditioner for Arhomu");
            ATSPrecond =
                SolverFactory::CreateAlgebraicPrecond(*Arhomu,lsParams.sublist("ATS Precond"));
        }
        else
        {
            DEBUG("Create Preconditioner for ATS");
            ATSPrecond =
                SolverFactory::CreateAlgebraicPrecond(*ATS,lsParams.sublist("ATS Precond"));
        }
        DEBUG("Compute ATSPrecond...");
        SolverFactory::ComputeAlgebraicPrecond(ATSPrecond,lsParams.sublist("ATS Precond"));
    }

    // tell the solvers which preconditioners to use:
    DEBUG("Set Preconditioner Operators...");

    if (ATSSolver!=Teuchos::null)
    {
        // set outer stream for Aztec solver
        ATSSolver->SetOutputStream(*OuterStream);
        ATSSolver->SetErrorStream(*OuterErrorStream);
        if (lsParams.sublist("ATS Precond").get("Method","None")!="None")
        {
            CHECK_ZERO(ATSSolver->SetPrecOperator(ATSPrecond.get()));
        }
    }

    DEBUG("leave build_preconditioner");
}//build_preconditioner


///////////////////////////////////////////////////////////////////////////////
// Apply preconditioner matrix (not available)
///////////////////////////////////////////////////////////////////////////////

int HYMLSBlockPreconditioner::
Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
    // Not implemented: Throw an error!
    ERROR("HYMLSBlockPreconditioner::Apply() not implemented",__FILE__,__LINE__);
    return -1;
}

///////////////////////////////////////////////////////////////////////////////
// apply inverse preconditioner step
///////////////////////////////////////////////////////////////////////////////

int HYMLSBlockPreconditioner::
ApplyInverse(const Epetra_MultiVector& input,
             Epetra_MultiVector& result) const
{
    bool noisy = (verbose>=8);
    if (noisy) INFO("Apply Block-Preconditioner...");

//  DEBVAR(input);

    if ((input.NumVectors()>1) || (input.NumVectors()!=result.NumVectors()))
    {
        ERROR("Ocean Preconditioner not implemented for multiple RHS, yet!",__FILE__,__LINE__);
    }

// check if input vectors are multivectors or standard vectors
    const Epetra_Vector *input_vec = dynamic_cast<const Epetra_Vector*>(&input);
    Epetra_Vector *result_vec = dynamic_cast<Epetra_Vector*>(&result);

    if (input_vec==NULL)
    {
        input_vec = input(0);
        result_vec= result(0);
    }

    // cast blockvectors into vectors b=input, x=outpu
    const Epetra_Vector& b = *(input_vec);
    Epetra_Vector& x       = *(result_vec);

    // make the solvers report to our own files
    // (note that Aztec uses a static stream
    // because it is based on old C code)

    // note: it is not 100% clear if we have to
    // set it for all solvers, so we just do it
    if (ATSSolver!=Teuchos::null)
    {
        ATSSolver->SetOutputStream(*InnerStream);
        ATSSolver->SetErrorStream(*InnerErrorStream);
    }

    // (0) PREPROCESSING

    if (noisy)  INFO("(0) Split rhs vector ...");

    // split b = [buv,bTS]' and x = [xuv,xTS]'  // ++scales++
    Epetra_Vector buv(*mapUV);
    Epetra_Vector bTS(*mapTS);

    Epetra_Vector xuv(*mapUV);
    Epetra_Vector xTS(*mapTS);

    CHECK_ZERO(buv.Export(b,*importUV,Zero));
    CHECK_ZERO(bTS.Export(b,*importTS,Zero));

    CHECK_ZERO(xuv.Export(x,*importUV,Zero));
    CHECK_ZERO(xTS.Export(x,*importTS,Zero));

    Epetra_Vector yuv(*mapUV);
    Epetra_Vector yTS(*mapTS);

    if (zero_init)
    {
        CHECK_ZERO(xuv.PutScalar(0.0));
        CHECK_ZERO(xTS.PutScalar(0.0));
    }

    if (scheme=="ILU")
    {
        // ERROR("Block-ILU is no longer supported!!!",__FILE__,__LINE__);
        //solve y = L\b
        if (noisy) INFO("(1) Solve Ly=b...");
        SolveLower(buv,bTS,yuv,yTS);
        // solve x = U\y
        if (noisy) INFO("(2) Solve Ux=y...");
        SolveUpper(yuv,yTS,xuv,xTS);
    }
    else if (scheme=="Gauss-Seidel")
    {
        //solve y = (D+wL)\b
        if (noisy) INFO("(1) Solve (D+wL)x=b...");
        SolveLower(buv,bTS,xuv, xTS);
    }
    else if (scheme=="symmetric Gauss-Seidel")
    {
        ERROR("symmetric GS is no longer supported!!!",__FILE__,__LINE__);
        /* this method has not proved useful and is no longer supported
        //solve x = (D+wL)\b
        if (noisy) INFO("(1) Solve (D+wL)x=b...");
        SolveLower(buv,bTS,xuv, xTS);
        if (noisy) INFO("(2) apply (D+wU)\\D (BuvTS correction)...");
        CHECK_ZERO(SubMatrix[_BuvTS]->Multiply(false,xTS,yw));
        yp.PutScalar(0.0);
        Ap->ApplyInverse(yw,yp);
        CHECK_ZERO(xp.Update(-DampingFactor,yp,1.0));
        if (noisy) INFO("(4) Scale with w(2-w)...");
        double fac = DampingFactor*(2-DampingFactor);
        CHECK_ZERO(xuv.Scale(fac));
        CHECK_ZERO(xw.Scale(fac));
        CHECK_ZERO(xp.Scale(fac));
        CHECK_ZERO(xTS.Scale(fac));
        */
    }
    else
    {
        ERROR("Unsupported Scheme: "+scheme,__FILE__,__LINE__);
    }

    // (3) Postprocessing

    if (noisy) INFO("(3) Postprocess: construct final result...");

    // (3.1) fill result vector x
    CHECK_ZERO(x.PutScalar(0.0));
    CHECK_ZERO(x.Import(xuv,*importUV,Add));
    CHECK_ZERO(x.Import(xTS,*importTS,Add));

    // Epetra_Vector tmp = x;
    // jacobian->Apply(x, tmp);
    // std::cout << tmp<<std::endl;
    // std::cout << b<<std::endl;
    // tmp.Update(1.0, b, -1.0);
    // std::cout << tmp<<std::endl;

    // reset the static Aztec stream for the outer iteration
    // as it may happen that there is no Auv Solver, we do
    // it for all three solvers to make sure the stream is
    // reset
    if (ATSSolver!=Teuchos::null)
    {
        ATSSolver->SetOutputStream(*OuterStream);
        ATSSolver->SetErrorStream(*OuterErrorStream);
    }
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
// compute orthogonal transform QTS so that QTS*ATS*QTS is easier to solve   //
///////////////////////////////////////////////////////////////////////////////

void HYMLSBlockPreconditioner::setup_rhomu()
{
    if (QTS==Teuchos::null)
    {
        QTS=Teuchos::rcp(new Epetra_CrsMatrix(Copy,*mapTS,*mapTS,2,true));
        int indices[2];
        double values[2];
        double alphaT = 1.8e-4; //TODO: hard-coded for the moment
        double alphaS = 7.6e-4;
        double lambda = alphaS/alphaT; // par(LAMB) in THCM
        //double idet = 1.0/sqrt(1+lambda*lambda);
        double idet = 1.0/sqrt(2.0);
        // we choose Q such that it is orthonormal, Q^2=I
        int imax = QTS->NumMyRows();

        for (int i=0;i<imax;i+=2)
        {
            int gidT=QTS->GRID(i);
            int gidS=QTS->GRID(i+1);
            indices[0]=gidT; indices[1]=gidS;
            values[0] =-idet; values[1]=lambda*idet;
            CHECK_ZERO(QTS->InsertGlobalValues(gidT,2,values,indices));
            values[0] = idet/lambda; values[1]=idet;
            CHECK_ZERO(QTS->InsertGlobalValues(gidS,2,values,indices));
        }

        for (int i=imax;i<QTS->NumMyRows();i+=2)
        {
            int gidT=QTS->GRID(i);
            int gidS=QTS->GRID(i+1);

            indices[0]=gidT;
            values[0] =1.0;
            CHECK_ZERO(QTS->InsertGlobalValues(gidT,1,values,indices));
            indices[0]=gidS;
            CHECK_ZERO(QTS->InsertGlobalValues(gidS,1,values,indices));
        }

        CHECK_ZERO(QTS->FillComplete());
    }

    Arhomu = Utils::TripleProduct(false,*QTS,false,*ATS,false,*QTS);

    Arhomu->SetLabel("A_(rho,mu)");
#ifdef LINEAR_ARHOMU_MAPS
    //{ use linear map for Arhomu (to allow use of SuperLU)
    Arhomu_linearmap = Teuchos::rcp(new Epetra_Map(Arhomu->NumGlobalRows(),0,Arhomu->Comm()));
    Teuchos::RCP<Epetra_Map> lincolmap = Utils::AllGather(*Arhomu_linearmap);

    int maxlen = Arhomu->MaxNumEntries();
    int len;
    int *ind = new int[maxlen];
    double *val = new double[maxlen];
    int nloc = Arhomu->NumMyRows();
    int *row_lengths = new int[nloc];
    for (int i=0;i<nloc;i++) row_lengths[i]=Arhomu->NumMyEntries(i);
    Teuchos::RCP<Epetra_CrsMatrix> tmpmat;
    tmpmat = Teuchos::rcp(new Epetra_CrsMatrix(Copy,*Arhomu_linearmap,*lincolmap,row_lengths) );

    int rowA,rowNew;

    for (int i=0;i<Arhomu->NumMyRows();i++)
    {
        rowA = Arhomu->GRID(i);
        rowNew = Arhomu_linearmap->GID(i);
        CHECK_ZERO(Arhomu->ExtractGlobalRowCopy(rowA,maxlen,len,val,ind));
        for (int j=0;j<len;j++)
        {
            int rem=MOD(ind[j],_NUN_);
            int newind=(ind[j]-rem)/_NUN_ + MOD(ind[j],2);
            ind[j] = newind;
        }
        CHECK_ZERO(tmpmat->InsertGlobalValues(rowNew, len, val, ind));
    }
    //Arhomu=Utils::RemoveColMap(tmpmat);
    Arhomu=tmpmat;
    CHECK_ZERO(Arhomu->FillComplete());
#endif
#ifdef STORE_MATRICES
    Utils::Dump(*QTS,"QTS");
    Utils::Dump(*Arhomu,"Arhomu");
#endif
}

///////////////////////////////////////////////////////////////////////////////
// inf-norm (n/a)
///////////////////////////////////////////////////////////////////////////////


double HYMLSBlockPreconditioner::NormInf() const
{
    // Not implemented: Throw an error!
    std::cout << "ERROR: HYMLSBlockPreconditioner::NormInf() - "
              << "method is NOT implemented!!  " << std::endl;
    throw "Error: method not implemented";
}




//////////////////////////////////////////////////////////////////////////////
// solve Ly = b for y:                                                      //
//////////////////////////////////////////////////////////////////////////////
void HYMLSBlockPreconditioner::SolveLower(const Epetra_Vector& buv,
                                     const Epetra_Vector& bTS,
                                     Epetra_Vector& yuv,
                                     Epetra_Vector& yTS) const
{
    TIMER_SCOPE("BlockPrec: solve lower");
    CHECK_ZERO(AuvPrecond->ApplyInverse(buv, yuv));
    // CHECK_ZERO(AuvSolver->ApplyInverse(buv, yuv));
    CHECK_ZERO(SubMatrix[_BTSuv]->Multiply(false,yuv,yTS));
    Epetra_Vector yTS2 = yTS;
    // yTS2 = bTS - yTS
    CHECK_ZERO(yTS2.Update(1.0,bTS,-DampingFactor));
    SolveATS(yTS2,yTS,tolATS,nitATS);
}

// apply x=U\y
void HYMLSBlockPreconditioner::SolveUpper(const Epetra_Vector& yuv, const Epetra_Vector& yTS,
                                     Epetra_Vector& xuv, Epetra_Vector& xTS) const

{
    TIMER_SCOPE("BlockPrec: solve upper");

    // temporary vector
    Epetra_Vector z = yuv;

    // (2) Apply x = U\y
    DEBUG("(3) Solve Ux=y for x");

    // (2.1) compute xTS

    // xTS = yTS
    xTS = yTS;

    // (2.2) compute xw

    // apply zw1 = BuvTS*yTS
    CHECK_ZERO(SubMatrix[_BuvTS]->Multiply(false,yTS,z));

    // apply zp=Auv\(BuvTS*yTS)
    CHECK_ZERO(AuvPrecond->ApplyInverse(z,xuv));
    // CHECK_ZERO(AuvSolver->ApplyInverse(z,xuv));
    CHECK_ZERO(xuv.Update(1.0,yuv,-1.0));
}

void HYMLSBlockPreconditioner::SolveATS(Epetra_Vector& rhs,
                                   Epetra_Vector& sol,
                                   double tol, int maxit) const
{
    if (zero_init)
    {
        CHECK_ZERO(sol.PutScalar(0.0));
    }
    Teuchos::RCP<Epetra_Vector> rhs_ptr = Teuchos::rcp(&rhs,false);
    Teuchos::RCP<Epetra_Vector> sol_ptr = Teuchos::rcp(&sol,false);
    if (QTS!=Teuchos::null)
    {
        rhs_ptr = Teuchos::rcp(new Epetra_Vector(*mapTS));
        sol_ptr = Teuchos::rcp(new Epetra_Vector(*mapTS));
        CHECK_ZERO(QTS->Multiply(false,sol,*sol_ptr));
        CHECK_ZERO(QTS->Multiply(false,rhs,*rhs_ptr));
    }
// TODO: This is for direct solvers for A_(rho/mu) and irrelevant in practice
#ifdef LINEAR_ARHOMU_MAPS
    if (Arhomu_linearmap!=Teuchos::null)
    {
        CHECK_ZERO(rhs_ptr->ReplaceMap(*Arhomu_linearmap));
        CHECK_ZERO(sol_ptr->ReplaceMap(*Arhomu_linearmap));
    }
#endif

    // Solve system with ATS or Arhomu iteratively or apply preconditioner once
    if (ATSSolver!=Teuchos::null)
    {
        TIMER_START("BlockPrec: solve ATS");
        ATSSolver->SetRHS(rhs_ptr.get());
        ATSSolver->SetLHS(sol_ptr.get());
        CHECK_NONNEG(ATSSolver->Iterate(maxit,tol));
        TIMER_STOP("BlockPrec: solve ATS");
    }
    else
    {
        CHECK_ZERO(ATSPrecond->ApplyInverse(*rhs_ptr,*sol_ptr));
    }
#ifdef LINEAR_ARHOMU_MAPS
    if (Arhomu_linearmap!=Teuchos::null)
    {
        CHECK_ZERO(rhs_ptr->ReplaceMap(*Arhomu_linearmap));
        CHECK_ZERO(sol_ptr->ReplaceMap(*Arhomu_linearmap));
    }
#endif
    if (QTS!=Teuchos::null)
    {
        CHECK_ZERO(QTS->Multiply(false,*sol_ptr,sol));
    }
}

// we need a simple search for column indices since it seems that in parallel
// the notion of 'Sorted()' is different from the serial case (?)
bool HYMLSBlockPreconditioner::find_entry(int col, int* indices, int numentries,int& pos)
{
    pos = 0;
    while (pos<numentries)
    {
        if (indices[pos]==col) break;
        pos++;
    }
    return (pos<numentries);
}

// store Jacobian, rhs, start guess and all the preconditioner 'hardware'
// (i.e. depth-averaging operators etc) in an HDF5 file
void HYMLSBlockPreconditioner::dumpLinSys(const Epetra_Vector& x, const Epetra_Vector& b) const
{
#ifndef HAVE_XDMF
    INFO("WARNING: cannot dump linear system, hdf5 is not available!");
#else
    Teuchos::RCP<EpetraExt::HDF5> hdf5 = Teuchos::rcp(new EpetraExt::HDF5(*comm));
    hdf5->Create("linsys.h5");
// write linear system:
    hdf5->Write("x",x);
    hdf5->Write("rhs",b);
    hdf5->Write("jacobian",*jacobian);

// write preconditioner hardware:
    hdf5->Write("mzp1",*Mzp1);
    hdf5->Write("mzp2",*Mzp2);
    hdf5->Write("mapuv",*mapUV);
    hdf5->Write("mapw",*mapW1);
    hdf5->Write("mapp",*mapP1);
    hdf5->Write("mappbar",*mapPbar);
    hdf5->Write("mapts",*mapTS);

    hdf5->Write("svp1",*svp1);
    hdf5->Write("svp2",*svp2);

    hdf5->Close();
    comm->Barrier();

// it seems that the HDF5 lib on my notebook doesn't like our singular vectors...
// we store them in ASCII for the moment:
/*
  if (comm->NumProc()>1) ERROR("dump linsys doesn't support parallel runs presently!",__FILE__,__LINE__);
  std::ofstream ofs1("svp1.txt");
  std::ofstream ofs2("svp2.txt");
  ofs1 << *svp1;
  ofs2 << *svp2;
  ofs1.close();
  ofs2.close();
*/
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATION of Ifpack_Preconditioner interface. This allows us to use an                                                 //
// HYMLSBlockPreconditioner inside an Ifpack_AdditiveSchwarz, i.e. for the Seasonal Cycle problem.                                  //
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

HYMLSBlockPreconditioner::HYMLSBlockPreconditioner(Epetra_RowMatrix* RowMat)
    : label_("Ocean Preconditioner"),
      needs_setup(true), IsComputed_(false)
{
    INFO("HYMLSBlockPreconditioner, Ifpack constructor");
    Epetra_CrsMatrix* CrsMat = dynamic_cast<Epetra_CrsMatrix*>(RowMat);
    if (CrsMat==NULL) ERROR("HYMLSBlockPreconditioner needs CrsMatrix!",__FILE__,__LINE__);
    jacobian = Teuchos::rcp(CrsMat,false);

    // we also need a 'domain' object. The only way to get that right now is
    // from the global THCM instance:
    domain = TRIOS::Static::GetDomain();
    if (domain==Teuchos::null)
    {
        ERROR("could not get static domain pointer, required for Ifpack constructor.",__FILE__,__LINE__);
    }

    comm = domain->GetComm();

    // no params given: set defaults by passing in an empty list
    Teuchos::RCP<Teuchos::ParameterList> List =
        Teuchos::rcp(new Teuchos::ParameterList);
    this->SetParameters(*List);

    // finish constructor
    this->Setup1();
}

int HYMLSBlockPreconditioner::SetParameters(Teuchos::ParameterList &List)
{
    lsParams = List;
    Teuchos::RCP<std::ostream> defaultStream =
        Teuchos::rcp(new Teuchos::FancyOStream(Teuchos::rcp(&std::cout,false)));

    OuterStream      = lsParams.get("Outer Output Stream", defaultStream);
    defaultStream    = Teuchos::rcp_dynamic_cast<Teuchos::FancyOStream>(OuterStream);
    OuterErrorStream = lsParams.get("Outer Error Stream", defaultStream);
    InnerStream      = lsParams.get("Inner Output Stream",defaultStream);
    InnerErrorStream = lsParams.get("Inner Error Stream", defaultStream);

    nitAuv = lsParams.sublist("Auv Solver").get("Max Num Iter",1);
    nitSpp = lsParams.sublist("Saddlepoint Solver").get("Max Num Iter", 5);

    lsParams.sublist("Auv Precond").sublist("Preconditioner").set("Separator Length (z)", domain->GlobalL());
    lsParams.sublist("Auv Precond").sublist("Preconditioner").set("Coarsening Factor (z)", 1);
    lsParams.sublist("Auv Precond").sublist("Problem").set("nx", domain->GlobalN());
    lsParams.sublist("Auv Precond").sublist("Problem").set("ny", domain->GlobalM());
    lsParams.sublist("Auv Precond").sublist("Problem").set("nz", domain->GlobalL());

    tolAuv = lsParams.sublist("Auv Solver").get("Tolerance",1e-4);
    DEBVAR(tolAuv);
    tolSpp = lsParams.sublist("Saddlepoint Solver").get("Tolerance",1e-8);

    scheme = lsParams.get("Scheme","Gauss-Seidel");
    permutation = lsParams.get("Permutation",1);
    verbose = lsParams.get("Verbosity",10);
    zero_init = lsParams.get("Zero Initial Guess",true);

    DampingFactor = lsParams.get("Relaxation: Damping Factor",1.0);

    // for B-grid
    DoPresCorr = lsParams.get("Subtract Spurious Pressure Modes", true);

    // if (scheme=="ILU") //need to solve Schur-complement instead of ATS
    // {
    //     ERROR("BILU Preconditioner is no longer supported!",__FILE__,__LINE__);
    // }

    nitATS = lsParams.sublist("ATS Solver").get("Max Num Iter",25);
    tolATS = lsParams.sublist("ATS Solver").get("Tolerance",1e-10);
    return 0;
}


int HYMLSBlockPreconditioner::Initialize()
{
    // this concept is not clearly implemented here,
    // the ocean preconditioner keeps track of its
    // state using the needs_setup flag
    return 0;
}

bool HYMLSBlockPreconditioner::IsInitialized() const
{
    return true;
}

int HYMLSBlockPreconditioner::Compute()
{
    INFO("  Compute Ocean Preconditioner for " << jacobian->Label());

    if (needs_setup) Setup2(); // allocate memory, build submaps...
    // This has to be done exactly once,
    // but not before the Jacobian is there.

    // Extract Submatrices:
    extract_submatrices(*jacobian);


#ifdef STORE_MATRICES // this is only for debugging, and only for moderate dimensions
    for (int i=0;i<_NUMSUBM;i++)
    {
        Utils::Dump(*SubMatrix[i],SubMatrixLabel[i]);
    }
    Utils::Dump(*Mzp1,"Mzp1");
    Utils::Dump(*Mzp2,"Mzp2");
    Utils::Dump(*Aw,    "Aw");
    Utils::Dump(*Duv1,"Duv1");
#endif

    // build blocksystems, preconditioners and solvers
    build_preconditioner();
    IsComputed_=true;
    return 0;
}

bool HYMLSBlockPreconditioner::IsComputed() const
{
    // if needs_setup==false, it has certainly been computed once.
    // but not necessarily for this matrix, so I'm not sure here.
    return !needs_setup;
}

double HYMLSBlockPreconditioner::Condest() const
{
    // we can't compute that right now
    return -1.0;
}

double HYMLSBlockPreconditioner::Condest(const Ifpack_CondestType CT,
                                    const int MaxIters,
                                    const double Tol,
                                    Epetra_RowMatrix* Matrix)
{
    return -1.0;
}



const Epetra_RowMatrix& HYMLSBlockPreconditioner::Matrix() const
{
    return *jacobian;
}

int HYMLSBlockPreconditioner::NumInitialize() const
{
    return -1; // no idea, we don't need this counting business
    // (THCM class takes care of the important counting/timing)
}

int HYMLSBlockPreconditioner::NumCompute() const
{
    return -1;
}

int HYMLSBlockPreconditioner::NumApplyInverse() const
{
    return -1;
}

double HYMLSBlockPreconditioner::InitializeTime() const
{
    return 0.0;
}
double HYMLSBlockPreconditioner::ComputeTime() const
{
    return 0.0;
}

double HYMLSBlockPreconditioner::ApplyInverseTime() const
{
    return 0.0;
}

double HYMLSBlockPreconditioner::InitializeFlops() const
{
    return 0.0;
}

double HYMLSBlockPreconditioner::ComputeFlops() const
{
    return 0.0;
}

double HYMLSBlockPreconditioner::ApplyInverseFlops() const
{
    return 0.0;
}

std::ostream& HYMLSBlockPreconditioner::Print(std::ostream& os) const
{
    os << Label()<<std::endl;
    os << "Scheme: " << scheme << ", ordering: " << permutation << std::endl;
    return os;
}

#ifdef TESTING
void HYMLSBlockPreconditioner::Test()
{
    Setup2();
    extract_submatrices(*jacobian);
    test_svp();
    Teuchos::RCP<Epetra_CrsMatrix> T =
        Teuchos::rcp(new Epetra_CrsMatrix(Copy, (*SubMatrix[_Guv]).RowMap(),
                                          (*SubMatrix[_Guv]).MaxNumEntries()));
    INFO("Build Mzp1");
    Mzp1 = build_singular_matrix(SubMatrix[_Gw]);

    INFO("Multiply Guv with Mzp1");
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(*SubMatrix[_Guv], false,
                                                 *Mzp1, true, *T));
    INFO("transpose Dw...");
    Epetra_RowMatrixTransposer Trans(SubMatrix[_Dw].get());
    Epetra_CrsMatrix* tmpGw;
    CHECK_ZERO(Trans.CreateTranspose(true, tmpGw, SubMatrixRowMap[_Gw].get()));

    INFO("build Mzp2...");
    Mzp2 = build_singular_matrix(Teuchos::rcp(tmpGw));
    Teuchos::RCP<Epetra_CrsMatrix> T2 =
        Teuchos::rcp(new Epetra_CrsMatrix(Copy, (*Mzp2).RowMap(),
                                          (*SubMatrix[_Duv]).ColMap(),
                                          (*Mzp2).MaxNumEntries()));

    DEBVAR(*Mzp2);
    DEBVAR(*SubMatrix[_Duv]);
    DEBVAR((*SubMatrix[_Duv]).Importer());
    DEBVAR((*SubMatrix[_Duv]).ColMap().SameAs(T2->ColMap()));
    INFO("Multiply Mzp2 with Duv");
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(*Mzp2, false,
                                                 *SubMatrix[_Duv], false, *T2));
}
#endif

}//namespace TRIOS
