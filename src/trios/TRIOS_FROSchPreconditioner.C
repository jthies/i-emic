
#include "TRIOS_FROSchPreconditioner.H"

#ifdef HAVE_FROSCH

// note: The whole FROSch stack of headers requires a certain
// order and completeness, do not try this at home
#include "Xpetra_MapUtils.hpp"
#include "FROSch_Tools_def.hpp"
#include "FROSch_SchwarzPreconditioners_fwd.hpp"
#include "FROSch_OneLevelPreconditioner_def.hpp"
#include "FROSch_TwoLevelPreconditioner_def.hpp"

// conversion tools for Epetra->Xpetra
#include "Xpetra_EpetraMap.hpp"
#include "Xpetra_EpetraCrsMatrix.hpp"

using Xpetra_EpetraMap = Xpetra::EpetraMapT<GO,EpetraNode>;

namespace TRIOS
{

  FROSchPreconditioner::FROSchPreconditioner(Teuchos::RCP<Epetra_CrsMatrix> jac,
                                             Teuchos::RCP<const Domain> domain,
                                             Teuchos::ParameterList &pList)
        :
        epetraMatrix_(*jac),
        epetraMap_(jac->DomainMap()),
        epetraComm_(jac->Comm()),
        xpetraMap_(Xpetra::toXpetra<GO,EpetraNode>(epetraMap_)),
        domain_(domain),
        isInitialized_(false),
        isComputed_(false),
        pList_(pList)
  {
    // Note: The FROSch solvers take an Xpetra::Matrix object, which is
    //       implemented by Xpetra::CrsMatrixWrap, which is
    //       a wrapper of an Xpetra::CrsMatrix object, which is
    //         implemented by Xpetra::EpetraCrsMatrixT, which is
    //         a wrapper of an Epetra_CrsMatrix, which is
    //         the actual class we use in the rest of TRIOS.
    // Welcome to the world of C++ abstraction layers!
    //
    Teuchos::RCP<Xpetra::EpetraCrsMatrixT<GO,EpetraNode>> s0(new Xpetra::EpetraCrsMatrixT<GO,EpetraNode>(Teuchos::rcpFromRef(epetraMatrix_)));
    Teuchos::RCP<Xpetra::CrsMatrix<double,LO,GO,EpetraNode>> s1 = s0;
    Teuchos::RCP<Xpetra::CrsMatrixWrap<double,LO,GO,EpetraNode>> A(new Xpetra::CrsMatrixWrap<double,LO,GO,EpetraNode>(s1));

    this->SetParameters(pList);
    // let's start with a one-level preconditioner and see how that works out
    frosch_ = Teuchos::rcp(new OneLevelFROSch(A, Teuchos::rcpFromRef(pList_)));
    //frosch_ = Teuchos::rcp(new TwoLevelFROSch(A, Teuchos::rcpFromRef(pList_)));
    INFO("FROSch parameters after constructor:");
    INFO(pList_);
    INFO("END FROSch parameters");
  }

  int FROSchPreconditioner::SetParameters(Teuchos::ParameterList& paramList)
  {
    pList_.setParameters(paramList);
    return 0;
  }

    int FROSchPreconditioner::Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
    {
      return -99;
    }

    int FROSchPreconditioner::ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
    {
      Teuchos::RCP<const Xpetra_MultiVector> rhs = Teuchos::rcp(new
        Xpetra::EpetraMultiVectorT<GO, EpetraNode>(Teuchos::rcpFromRef(const_cast<Epetra_MultiVector&>(X))));
      Teuchos::RCP<Xpetra_MultiVector> sol = Teuchos::rcp(new
        Xpetra::EpetraMultiVectorT<GO, EpetraNode>(Teuchos::rcpFromRef(Y)));

      frosch_->apply(*rhs, *sol);
      return 0;
    }

    int FROSchPreconditioner::Initialize()
    {
      // We use the 'assembly map' from the ocean model,
      // it has an overlap of two grid cells. Alternatively,
      // we could create a map that has only one velocity node
      // of overlap. This would require some manual construction,
      // though, whereas the assembly map is already available.
      int overlap = 2;
      Teuchos::RCP<const Xpetra_Map> repeatedMap =
        Teuchos::rcp(new Xpetra_EpetraMap(domain_->GetAssemblyMap()));
      int result = frosch_->initialize(overlap, repeatedMap);
      INFO("FROSch parameters after initialize:");
      INFO(pList_);
      INFO("END FROSch parameters");
      return result;
    }

    int FROSchPreconditioner::Compute()
    {
      int result = frosch_->compute();
      INFO("FROSch parameters after constructor:");
      INFO(pList_);
      INFO("END FROSch parameters");
      return result;
    }
  



} //namespace TRIOS

#if 0

// This is Alex' example from the Trilnos repository (tutorial in ShyLU).

int main (int argc, char *argv[])
{

    // Create communicator
    auto comm = Tpetra::getDefaultComm ();

    // Initialize stacked timers
    comm->barrier();
    RCP<StackedTimer> stackedTimer = rcp(new StackedTimer("FROSch Example"));
    TimeMonitor::setStackedTimer(stackedTimer);

    // Set verbosity
    const bool verbose = (myRank == 0);
    EVerbosityLevel verbosityLevel = static_cast<EVerbosityLevel>(V);

    // Determine the number of subdomains per direction (if numProcs != N^dim stop)
    int N = 0;
    if (dimension == 2) {
        N = pow(comm->getSize(),1/2.) + 100*numeric_limits<double>::epsilon();
        FROSCH_ASSERT(N*N==numProcs,"#MPI ranks != N^2")
    } else if (dimension == 3) {
        N = pow(comm->getSize(),1/3.) + 100*numeric_limits<double>::epsilon();
        FROSCH_ASSERT(N*N*N==numProcs,"#MPI ranks != N^3")
    } else {
        assert(false);
    }

    // Set the linear algebra framework
    UnderlyingLib xpetraLib = UseTpetra;
    if (useEpetra) {
        xpetraLib = UseEpetra;
    } else {
        xpetraLib = UseTpetra;
    }

    if (verbose) cout << endl;
    if (verbose) cout << "###########################" << endl;
    if (verbose) cout << "# Demonstration of FROSch #" << endl;
    if (verbose) cout << "###########################" << endl;
    if (verbose) cout << "Solving " << dimension << "D ";
    if (!equation.compare("laplace")) {
        if (verbose) cout << "Laplace";
    } else {
        if (verbose) cout << "linear elasticity";
    }
    if (verbose) cout << " model problem using ";
    if (useEpetra) {
        if (verbose) cout << "Epetra";
    } else {
        if (verbose) cout << "Tpetra";
    }
    if (verbose) cout << " linear algebra throug Xpetra." << endl;
    if (verbose) cout << "\tNumber of MPI ranks: \t\t\t\t" << numProcs << endl;
    if (verbose) cout << "\tNumber of subdomains per dimension: \t\t" << N << endl;
    if (verbose) cout << "\tNumber of nodes per subdomains and dimension: \t" << M << endl;

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    if (verbose) cout << endl;
    if (verbose) cout << ">> I. Assemble the system matrix using Galeri\n";
    if (verbose) cout << endl;

    // Assemble the system matrix (Laplace / linear elasticity)
    RCP<matrix_type> A;
    RCP<multivector_type> coordinates;
    assembleSystemMatrix(comm,xpetraLib,equation,dimension,N,M,A,coordinates);

    A->describe(*out,verbosityLevel);

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    if (verbose) cout << endl;
    if (verbose) cout << ">> II. Construct iteration and right hand side vectors\n";
    if (verbose) cout << endl;

    // solution / iteration vector
    RCP<const map_type> rowMap = A->getRowMap();
    RCP<multivector_type> x = multivectorfactory_type::Build(rowMap,1);
    x->putScalar(0.0);
    x->describe(*out,verbosityLevel);

    // right hand side vector
    RCP<multivector_type> b = multivectorfactory_type::Build(rowMap,1);
    b->putScalar(1.0);
    b->describe(*out,verbosityLevel);

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    if (verbose) cout << endl;
    if (verbose) cout << ">> III. Construct Schwarz preconditioner\n";
    if (verbose) cout << endl;

    // FROSch preconditioner for Belos
    RCP<operatort_type> belosPrec;
        belosPrec = rcp(new xpetraop_type(prec));
    } else if (!preconditioner.compare("none")) {
    } else {
        FROSCH_ASSERT(false,"Preconditioner type unkown!")
    }

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    if (verbose) cout << endl;
    if (verbose) cout << ">> IV. Solve the linear equation system using GMRES\n";
    if (verbose) cout << endl;

    // Set up the linear equation system for Belos
    RCP<operatort_type> belosA = rcp(new xpetraop_type(A));
    RCP<linear_problem_type> linear_problem (new linear_problem_type(belosA,x,b));
    linear_problem->setProblem(x,b);
    if (preconditioner.compare("none")) {
        linear_problem->setRightPrec(belosPrec); // Specify the preconditioner
    }

    // Build the Belos iterative solver
    solverfactory_type solverfactory;
    RCP<solver_type> solver = solverfactory.create(parameterList->get("Belos Solver Type","GMRES"),belosList);
    solver->setProblem(linear_problem);
    solver->solve(); // Solve the linear system

    x->describe(*out,verbosityLevel);

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    if (verbose) cout << endl;
    if (verbose) cout << ">> V. Test solution\n";
    if (verbose) cout << endl;

    // Compute the 2-norm of the residual
    A->apply(*x,*b,Teuchos::NO_TRANS,static_cast<scalar_type> (-1.0),static_cast<scalar_type> (1.0));
    double normRes = b->getVector(0)->norm2();
    if (verbose) cout << "2-Norm of the residual = " << normRes << endl;

    if (verbose) cout << "Finished!" << endl;

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    // Print timers
    comm->barrier();
    stackedTimer->stop("FROSch Example");
    StackedTimer::OutputOptions options;
    options.output_fraction = options.output_minmax = true;
    if (timers) stackedTimer->report(*out,comm,options);

    return 0;
}
#endif

#endif //HAVE_FROSCH
