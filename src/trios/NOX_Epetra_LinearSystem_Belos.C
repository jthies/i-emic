#include "NOX_Config.h"

#include "NOX_Epetra_LinearSystem_Belos.H"	// class definition

// NOX includes
#include "NOX_Epetra_Interface_Required.H"
#include "NOX_Epetra_Interface_Jacobian.H"
#include "NOX_Epetra_Interface_Preconditioner.H"
#include "NOX_Epetra_MatrixFree.H"
#include "NOX_Epetra_FiniteDifference.H"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "NOX_Epetra_Scaling.H"
#include "NOX_Utils.H"

// External include files for Epetra, Belos, and Ifpack
#include "Epetra_Map.h"
#include "Epetra_Vector.h" 
#include "Epetra_Operator.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_VbrMatrix.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_LinearProblem.h"
#include "BelosEpetraAdapter.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosBlockGmresSolMgr.hpp"
#include "Ifpack.h"
#include "Ifpack_IlukGraph.h"
#include "Ifpack_CrsRiluk.h"
#include "Teuchos_ParameterList.hpp"

// EpetraExt includes for dumping a matrix
//#ifdef HAVE_NOX_DEBUG
#ifdef HAVE_NOX_EPETRAEXT
#include "EpetraExt_BlockMapOut.h"
#include "EpetraExt_MultiVectorOut.h"
#include "EpetraExt_RowMatrixOut.h"
#endif
//#endif

#ifdef HAVE_NOX_ML_EPETRA
#include "Teuchos_ParameterList.hpp"
#endif

#include <typeinfo>



// ***********************************************************************
NOX::Epetra::LinearSystemBelos::
LinearSystemBelos(
 Teuchos::ParameterList& printParams, 
 Teuchos::ParameterList& linearSolverParams, 
 const Teuchos::RCP<NOX::Epetra::Interface::Required>& iReq, 
 const NOX::Epetra::Vector& cloneVector,
 const Teuchos::RCP<NOX::Epetra::Scaling> s):
 LinearSystemAztecOO(printParams, linearSolverParams,
                     iReq, cloneVector, s)
{
  
  reset(linearSolverParams);
}

// ***********************************************************************
NOX::Epetra::LinearSystemBelos::
LinearSystemBelos(
 Teuchos::ParameterList& printParams, 
 Teuchos::ParameterList& linearSolverParams,  
 const Teuchos::RCP<NOX::Epetra::Interface::Required>& iReq, 
 const Teuchos::RCP<NOX::Epetra::Interface::Jacobian>& iJac, 
 const Teuchos::RCP<Epetra_Operator>& jacobian,
 const NOX::Epetra::Vector& cloneVector,
 const Teuchos::RCP<NOX::Epetra::Scaling> s):
  LinearSystemAztecOO(printParams, linearSolverParams,
                      iReq, iJac, jacobian, cloneVector, s)
{
  
  reset(linearSolverParams);
}

// ***********************************************************************
NOX::Epetra::LinearSystemBelos::
LinearSystemBelos(
 Teuchos::ParameterList& printParams, 
 Teuchos::ParameterList& linearSolverParams, 
 const Teuchos::RCP<NOX::Epetra::Interface::Required>& iReq, 
 const Teuchos::RCP<NOX::Epetra::Interface::Preconditioner>& iPrec, 
 const Teuchos::RCP<Epetra_Operator>& preconditioner,
 const NOX::Epetra::Vector& cloneVector,
 const Teuchos::RCP<NOX::Epetra::Scaling> s):
   LinearSystemAztecOO(printParams, linearSolverParams,
            iReq, iPrec, preconditioner, cloneVector, s)
{  
  reset(linearSolverParams);
}

// ***********************************************************************
NOX::Epetra::LinearSystemBelos::
LinearSystemBelos(
 Teuchos::ParameterList& printParams, 
 Teuchos::ParameterList& linearSolverParams,
 const Teuchos::RCP<NOX::Epetra::Interface::Jacobian>& iJac, 
 const Teuchos::RCP<Epetra_Operator>& jacobian,
 const Teuchos::RCP<NOX::Epetra::Interface::Preconditioner>& iPrec, 
 const Teuchos::RCP<Epetra_Operator>& preconditioner,
 const NOX::Epetra::Vector& cloneVector,
 const Teuchos::RCP<NOX::Epetra::Scaling> s):
   LinearSystemAztecOO(printParams, linearSolverParams, 
            iJac, jacobian, iPrec, preconditioner, cloneVector, s)
{  
  reset(linearSolverParams);
}

// ***********************************************************************
NOX::Epetra::LinearSystemBelos::~LinearSystemBelos() 
{
// handled by base class destructor and RCP's
}

// ***********************************************************************

//TODO: reset discards everything, is that what we want?
void NOX::Epetra::LinearSystemBelos::
reset(Teuchos::ParameterList& p)
{  
  // do everything the base class wants to do
  LinearSystemAztecOO::reset(p);

  // retrieve User's Belos list, add more things
  Teuchos::ParameterList& belosList=p.sublist("Belos");
  std::string linearSolver=p.get("Belos Solver","GMRES");

  bool verbose = utils.isPrintType(Utils::LinearSolverDetails);
  bool debug = utils.isPrintType(Utils::Debug);
  
  int verbosity = Belos::Errors + Belos::Warnings;
  if (verbose)
    { //TODO: where to put which option? how do we get readable output?
    verbosity+=Belos::TimingDetails+Belos::IterationDetails;
    verbosity+=Belos::StatusTestDetails+Belos::OrthoDetails+Belos::FinalSummary;
    }
  if (debug) verbosity+=Belos::Debug;

  // User is allowed to override these settings
  if (belosList.isParameter("Verbosity")==false)
    belosList.set("Verbosity",verbosity);

// TODO: not sure how the Output Manager works, probably this has no meaning
//belosOutputPtr=rcp(new Belos::OutputManager<double>(verbosity,rcp(&std::cout,false)));
  
  if (belosList.isParameter("Output Stream")==false)
    belosList.set("Output Stream",Teuchos::rcp(&std::cout, false));
  
  belosList.set("Output Style",(int)Belos::Brief);
  belosList.set("Verbosity",Belos::Errors+Belos::Warnings
                           +Belos::IterationDetails
                           +Belos::StatusTestDetails
                           +Belos::FinalSummary);
                           //+Belos::TimingDetails


  // create vectors
  belosRhs = Teuchos::rcp(new Epetra_Vector(tmpVectorPtr->getEpetraVector()));
  belosSol = Teuchos::rcp(new Epetra_Vector(tmpVectorPtr->getEpetraVector()));
     
  // NOX puts its adaptive choice of tolerance into this place:
  MT tol = (MT)(p.get("Tolerance",0.0));
  // so we use it to override the settings in the Belos list.
  p.sublist("Belos").set("Convergence Tolerance",tol);
    
  // create Belos interface to preconditioner.
  // This is simply an Epetra_Operator with 'Apply' and 'ApplyInverse' switched.
  belosPrecPtr = Teuchos::rcp(new Belos::EpetraPrecOp(solvePrecOpPtr));
  
  // create Belos problem interface
  belosProblemPtr = Teuchos::rcp(new Belos::LinearProblem<ST,MV,OP>(jacPtr,belosSol,belosRhs));

  // set preconditioner
  belosProblemPtr->setRightPrec(belosPrecPtr);

  bool set = belosProblemPtr->setProblem();
  if (set == false) {
    throwError("reset","Belos::LinearProblem failed to set up correctly!");
    }

// create the solver
if (linearSolver=="GMRES")
  {
  Teuchos::RCP<Teuchos::ParameterList> belosListPtr=Teuchos::rcp(&belosList,false);
  belosSolverPtr = Teuchos::rcp(new Belos::BlockGmresSolMgr<ST,MV,OP>(belosProblemPtr,belosListPtr));
  }
else
  {
  throwError("reset","Currently only 'GMRES' is supported as 'Belos Solver'");
  }  
}

// ***********************************************************************

bool NOX::Epetra::LinearSystemBelos::
applyJacobianInverse(Teuchos::ParameterList &p,
		     const NOX::Epetra::Vector& input, 
		     NOX::Epetra::Vector& result)
{
  
  int ierr = 0;

  // AGS: Rare option, similar to Max Iters=1 but twice as fast.
    if ( p.get("Use Preconditioner as Solver", false) ) 
      return applyRightPreconditioning(false, p, input, result);

  double startTime = timer.WallTime();
  
  // Zero out the delta X of the linear problem if requested by user.
  if (zeroInitialGuess)
    result.init(0.0);

  // Create Epetra linear problem object (only used for scaling)

  // Need non-const version of the input vector
  // Epetra_LinearProblem requires non-const versions so we can perform
  // scaling of the linear problem.
  NOX::Epetra::Vector& nonConstInput = const_cast<NOX::Epetra::Vector&>(input);
        
  Epetra_LinearProblem Problem(jacPtr.get(),
                               &(result.getEpetraVector()),
                               &(nonConstInput.getEpetraVector()));

  // ************* Begin linear system scaling *******************
  if ( !Teuchos::is_null(scaling) ) {

    if ( !manualScaling )
      scaling->computeScaling(Problem);
    
    scaling->scaleLinearSystem(Problem);

    if (utils.isPrintType(Utils::Details)) {
      utils.out() << *scaling << std::endl;
    }  
  }
  // ************* End linear system scaling *******************

  // Use EpetraExt to dump linear system if debuggging
#ifdef HAVE_NOX_DEBUG
#ifdef HAVE_NOX_EPETRAEXT

  ++linearSolveCount;
  std::ostringstream iterationNumber;
  iterationNumber << linearSolveCount;
    
  std::string prefixName = p.get("Write Linear System File Prefix", 
				 "NOX_LinSys");
  std::string postfixName = iterationNumber.str();
  postfixName += ".mm";

  if (p.get("Write Linear System", false)) {

    std::string mapFileName = prefixName + "_Map_" + postfixName;
    std::string jacFileName = prefixName + "_Jacobian_" + postfixName;    
    std::string rhsFileName = prefixName + "_RHS_" + postfixName;
    
    Epetra_RowMatrix* printMatrix = NULL;
    printMatrix = dynamic_cast<Epetra_RowMatrix*>(jacPtr.get()); 

    if (printMatrix == NULL) {
      cout << "Error: NOX::Epetra::LinearSystemAztecOO::applyJacobianInverse() - "
	   << "Could not cast the Jacobian operator to an Epetra_RowMatrix!"
	   << "Please set the \"Write Linear System\" parameter to false."
	   << std::endl;
      throw "NOX Error";
    }

    EpetraExt::BlockMapToMatrixMarketFile(mapFileName.c_str(), 
					  printMatrix->RowMatrixRowMap()); 
    EpetraExt::RowMatrixToMatrixMarketFile(jacFileName.c_str(), *printMatrix, 
					   "test matrix", "Jacobian XXX");
    EpetraExt::MultiVectorToMatrixMarketFile(rhsFileName.c_str(), 
					     nonConstInput.getEpetraVector());

  }
#endif
#endif

  // Make sure preconditioner was constructed if requested
  if (!isPrecConstructed && (precAlgorithm != None_)) {
    throwError("applyJacobianInverse", 
       "Preconditioner is not constructed!  Call createPreconditioner() first.");
  }
  

  // do Belos solve
  if (utils.isPrintType(Utils::Debug)) {  
    utils.out() << "**************************************"<<std::endl;
    utils.out() << "* Belos Parameter List               *"<<std::endl;
    utils.out() << "**************************************"<<std::endl;
    utils.out() << p.sublist("Belos");
    utils.out() << "**************************************"<<std::endl;
  }
  //
  // Perform solve
  //
  
  *belosRhs = input.getEpetraVector();
  *belosSol = result.getEpetraVector();
  
  
  belosProblemPtr->setProblem(belosSol,belosRhs);
  
  belosPrecPtr = Teuchos::rcp(new Belos::EpetraPrecOp(solvePrecOpPtr));
  belosProblemPtr->setRightPrec(belosPrecPtr);

  // this may change from solve to solve and is not in the Belos list.
  // it is set by NOX if "Forcing Term Method" is not "Constant, for instance.
  MT tol = (MT)(p.get("Tolerance",0.0));
  Teuchos::ParameterList& belosList = p.sublist("Belos");
  belosList.set("Convergence Tolerance",tol);
  
  belosSolverPtr->setParameters(rcp(&belosList, false));  
  
  Belos::ReturnType ret;
  try {
  ret = belosSolverPtr->solve();
  } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cout,ierr);
  
  

  

  Teuchos::RCP<Epetra_Vector> resultvec = 
        Teuchos::rcp_dynamic_cast<Epetra_Vector>(belosSol);
  if (Teuchos::is_null(resultvec)) throwError("ApplyJacobianInverse",
   "bad cast: unexpected vector type");
  result = *resultvec;

  // check for loss of accuracy
  bool loa = belosSolverPtr->isLOADetected();

  if (loa)
    {
    utils.out() << "WARNING: loss of accuracy in Belos solve!!!" << std::endl;
    utils.out() << "("<<__FILE__<<", line "<<__LINE__<<")"<<std::endl;
    }

// TODO: probably we don't want this??? 
  
  //
  // Compute actual residuals.
  //
  int numrhs=1; //TODO: can we make use of multiple RHS?
  bool badRes = false;
  std::vector<double> actual_resids( numrhs );
  std::vector<double> rhs_norm( numrhs );
  Epetra_MultiVector resid(jacPtr->OperatorRangeMap(), numrhs);
  OPT::Apply( *jacPtr, *belosSol, resid );
  MVT::MvAddMv( -1.0, resid, 1.0, *belosRhs, resid );
  MVT::MvNorm( resid, actual_resids );
  MVT::MvNorm( *belosRhs, rhs_norm );
  if (utils.isPrintType(Utils::Details)) {
    utils.out()<< "---------- Actual Residuals (normalized) ----------"<<std::endl<<std::endl;
    for ( int i=0; i<numrhs; i++) {
      double actRes = actual_resids[i]/rhs_norm[i];
      utils.out()<<"Problem "<<i<<" : \t"<< actRes <<std::endl;
      if (actRes > tol) badRes = true;
    } 
  }

  if (ret!=Belos::Converged || badRes || loa) {
    utils.out() << std::endl << "WARNING:  Belos did not converge!" << std::endl;
    ierr=-1;
  }
else
  {
  //
  // Default return value
  //
  if (utils.isPrintType(Utils::Details))
    utils.out() << std::endl << "SUCCESS:  Belos converged!" << std::endl;
  ierr=0;
  }


  // Unscale the linear system
  if ( !Teuchos::is_null(scaling) )
    scaling->unscaleLinearSystem(Problem);

  // Set the output parameters in the "Output" sublist
  if (outputSolveDetails) {
    Teuchos::ParameterList& outputList = p.sublist("Output");
    int prevLinIters = 
      outputList.get("Total Number of Linear Iterations", 0);
    int curLinIters = 0;
    double achievedTol = -1.0;
    curLinIters = belosSolverPtr->getNumIters();
    for ( int i=0; i<numrhs; i++) {
      double actRes = actual_resids[i]/rhs_norm[i];
      utils.out()<<"Problem "<<i<<" : \t"<< actRes <<std::endl;
      if (actRes > achievedTol) achievedTol = actRes;
    }

    outputList.set("Number of Linear Iterations", curLinIters);
    outputList.set("Total Number of Linear Iterations", 
			    (prevLinIters + curLinIters));
    outputList.set("Achieved Tolerance", achievedTol);
  }

  // Dump solution of linear system
#ifdef HAVE_NOX_DEBUG
#ifdef HAVE_NOX_EPETRAEXT
  if (p.get("Write Linear System", false)) {
    std::string lhsFileName = prefixName + "_LHS_" + postfixName;
    EpetraExt::MultiVectorToMatrixMarketFile(lhsFileName.c_str(), 
					   result.getEpetraVector());
  }
#endif
#endif

  double endTime = timer.WallTime();
  timeApplyJacbianInverse += (endTime - startTime);

  return (ierr==0);
}

// ***********************************************************************

void
NOX::Epetra::LinearSystemBelos::setAztecOOJacobian() const
{  
belosProblemPtr->setOperator(jacPtr);
}

// ***********************************************************************
void
NOX::Epetra::LinearSystemBelos::setAztecOOPreconditioner() const
{  
  if ( !Teuchos::is_null(solvePrecOpPtr))
    {
    belosPrecPtr = Teuchos::rcp(new Belos::EpetraPrecOp(solvePrecOpPtr));
    belosProblemPtr->setRightPrec(belosPrecPtr);
    }   
}

// ***********************************************************************

