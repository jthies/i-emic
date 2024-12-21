/**********************************************************************
 * Copyright by Jonas Thies, Univ. of Groningen 2006/7/8.             *
 * Permission to use, copy, modify, redistribute is granted           *
 * as long as this header remains intact.                             *
 * contact: jonas@math.rug.nl                                         *
 **********************************************************************/
 
/************************************************************************
//OCEAN MODEL 
************************************************************************/

// define this to 
// - read starting solution
// - compute Jacobian A
// - build preconditioner P
// - compute eigenvalues of P\A-I
// - exit the program
//#define ANALYZE_SPECTRUM 1
#include <iostream>

#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_LinearProblem.h"
#include "AztecOO.h"
#include "LOCA_Epetra.H"
#include "LOCA.H"

#include "TRIOS_SolverFactory.H"
#include "TRIOS_BlockPreconditioner.H"

#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

#include <sstream>

// global THCM definitions
#include "THCMdefs.H"

//User's application specific files 
#include "THCM.H"
#include "DefaultParams.H"
#include "OceanModel.H"


#include "NOX_Epetra_LinearSystem_Belos.H"
#include "Epetra_LinearProblem.h"

  // create a linear system with solver for our THCM application.
  // note that you can create a Belos solver as well because the
  // LinearSystem_Belos class is derived from the AztecOO one.
  Teuchos::RCP<NOX::Epetra::LinearSystem> createLinearSystem
           (Teuchos::RCP<OceanModel> model, Teuchos::ParameterList& lsParams,
           Teuchos::ParameterList& printParams, Teuchos::RCP<std::ostream> out)
  {
  DEBUG("enter THCM::createLinearSystem");

    Teuchos::RCP<LOCA::Epetra::Interface::TimeDependent> iReq = model;
    Teuchos::RCP<NOX::Epetra::Interface::Jacobian> iJac = model;
    Teuchos::RCP<NOX::Epetra::Scaling> scaling = model->getScaling();

    lsParams.sublist("Belos").set("Output Stream",out);

    Teuchos::RCP<NOX::Epetra::LinearSystemBelos> linsys;

    Teuchos::RCP<Epetra_CrsMatrix> A = model->getJacobian();
    Teuchos::RCP<Epetra_Vector> soln = model->getSolution();

    std::string PrecType = lsParams.get("Preconditioner","Ifpack");

    if (PrecType == "User Defined")
      {
#ifdef DEBUGGING
      lsParams.set("Verbosity",10);
#endif
      DEBUG("user defined preconditioning");
      Teuchos::RCP<Epetra_Operator> myPrecOperator = model->getPreconditioner();
      // set outer and inner output streams
      Teuchos::RCP<TRIOS::BlockPreconditioner> tmp =
        Teuchos::rcp_dynamic_cast<TRIOS::BlockPreconditioner>(myPrecOperator);
      if (!Teuchos::is_null(tmp)) tmp->setOutputStreams(outFile, out);
      Teuchos::RCP<NOX::Epetra::Interface::Preconditioner> iPrec = model;

      //Create the linear systems with Ocean preconditioner
      linsys = Teuchos::rcp(new NOX::Epetra::LinearSystemBelos(printParams,
                        lsParams, iJac, A, iPrec, myPrecOperator, soln, scaling));
      }
    else
      {
      DEBUG("Trilinos preconditioning");
      //Create the linear system with Trilinos preconditioners
      if (PrecType=="ML")
        {
        Teuchos::ParameterList& mllist=lsParams.sublist("ML");
        if (mllist.get("smoother: type","Aztec")=="Aztec")
          {
          int *az_options = new int[AZ_OPTIONS_SIZE];
          double *az_params = new double[AZ_PARAMS_SIZE];
          AZ_defaults(az_options,az_params);
          Teuchos::ParameterList& azlist = mllist.sublist("smoother: aztec list");

          // some reasonable default options for Krylov smoothers:
          az_options[AZ_solver]=AZ_GMRESR;
          az_options[AZ_scaling]=AZ_none;
          az_options[AZ_precond]=AZ_dom_decomp;
          az_options[AZ_subdomain_solve]=AZ_ilut;
          az_options[AZ_max_iter]=3;
          az_options[AZ_output]=0;
          az_options[AZ_overlap]=0;
          az_options[AZ_print_freq]=0;

          az_params[AZ_tol]=0.0;
          az_params[AZ_drop]=1.0e-12;
          az_params[AZ_ilut_fill]=2.0;

          TRIOS::SolverFactory::ExtractAztecOptions(azlist,az_options,az_params);

          mllist.set("smoother: Aztec options",az_options);
          mllist.set("smoother: Aztec params",az_params);
          }
        }

      linsys = Teuchos::rcp(new NOX::Epetra::LinearSystemBelos(
                   printParams, lsParams, iReq, iJac, A, soln,scaling));
      }
  DEBUG("leave THCM::createLinearSystem");
  return linsys;
  }

int main(int argc, char *argv[])
{

    // Initialize the environment:
    //  - MPI
    //  - output files
    //  - returns Trilinos' communicator Epetra_Comm
    Teuchos::RCP<Epetra_Comm> Comm = initializeEnvironment(argc, argv);


  //Get process ID and total number of processes
  int MyPID = Comm->MyPID();
  int NumProc = Comm->NumProc();

  DEBUG("*********************************************")
  DEBUG("* Debugging output for process "<<MyPID)
  DEBUG("* To prevent this file from being written,  ")
  DEBUG("* omit the -DDEBUGGING flag when compiling. ")
  DEBUG("*********************************************")
  Comm->Barrier();


  try {

  ////////////////////////////////////////////////////////
  // Setup Parameter Lists                              //
  ////////////////////////////////////////////////////////

    // read parameters from files
    // we create one big parameter list and further down we will set some things
    // that are not straight-forwars in XML (verbosity etc.)
    Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new 
    Teuchos::ParameterList);

    paramList->sublist("THCM") = THCM::Instance().getDefaultInitParameters();
    Teuchos::updateParametersFromXmlFile("thcm_params.xml",paramList.ptr());

    DefaultParams::LOCA(*paramList);
    Teuchos::updateParametersFromXmlFile("loca_params.xml",paramList.ptr());

    DefaultParams::NOX(*paramList);
    // create the NOX->Direction->Newton->Linear Solver sublist
    /*TRIOS::DefaultParams::LinearSolver(paramList->sublist("NOX")
                                                 .sublist("Direction")
                                                 .sublist("Newton"));
     */
    // create the "BlockPreconditioner" sublist
    /*TRIOS::DefaultParams::BlockPreconditioner(paramList->sublist("NOX")
                                                 .sublist("Direction")
                                                 .sublist("Newton")
                                                .sublist("Linear Solver"));
     */
    // override default settings with parameters from user input file
    Teuchos::updateParametersFromXmlFile("solver_params.xml",paramList.ptr());

    // extract the final sublists:

    // Get the THCM sublist
    Teuchos::ParameterList& thcmList = paramList->sublist("THCM");

    // Get the LOCA sublist
    Teuchos::ParameterList& locaParamsList = paramList->sublist("LOCA");

    // get the Stepper sublist
    Teuchos::ParameterList& stepperList = locaParamsList.sublist("Stepper");
    std::string cont_param = stepperList.get("Continuation Parameter","Combined Forcing");
    double start_value = stepperList.get("Initial Value",0.0);

    // Get Anasazi Eigensolver sublist (needs --with-loca-anasazi)
//    Teuchos::ParameterList& aList = stepperList.sublist("Eigensolver");
//    aList.set("Verbosity", Anasazi::Errors+Anasazi::Warnings+Anasazi::FinalSummary);

    // Get the "Solver" parameters sublist to be used with NOX Solvers
    Teuchos::ParameterList& nlParams = paramList->sublist("NOX");

    Teuchos::ParameterList& nlPrintParams = nlParams.sublist("Printing");
    nlPrintParams.set("MyPID", MyPID);
    nlPrintParams.set("Output Stream",cdataFile);
    nlPrintParams.set("Error Stream",cdataFile);
    nlPrintParams.set("Output Process",0);
    nlPrintParams.set("Output Information",
			  NOX::Utils::Details + 
			  NOX::Utils::OuterIteration + 
			  NOX::Utils::InnerIteration +
                          NOX::Utils::OuterIterationStatusTest + 
                          NOX::Utils::LinearSolverDetails + 
                          NOX::Utils::Debug +
			  NOX::Utils::Warning +
			  NOX::Utils::StepperDetails +
			  NOX::Utils::StepperIteration +
			  NOX::Utils::StepperParameters); 


    //Create the "Direction" sublist for the "Line Search Based" solver
    Teuchos::ParameterList& dirParams = nlParams.sublist("Direction");

    //Create the "Line Search" sublist for the "Line Search Based" solver
    Teuchos::ParameterList& searchParams = nlParams.sublist("Line Search");

    //Create the "Direction" sublist for the "Line Search Based" solver
    Teuchos::ParameterList& newtParams = dirParams.sublist("Newton");

    //Create the "Linear Solver" sublist for the "Direction' sublist
    Teuchos::ParameterList& lsParams = newtParams.sublist("Linear Solver");

///////////////////////////////////////////////////////////
// Setup the Problem                                     //
///////////////////////////////////////////////////////////

    // put the correct starting value for the continuation parameter
    // in the thcm-list
    thcmList.sublist("Starting Parameters").set(cont_param,start_value);

    // Set up the THCM interface to allow calls to
    // residual (RHS) and Jacobian evaluation routines.
    // THCM is implemented as a Singleton, that means there
    // is only a single instance which should be referenced 
    // using THCM::Instance()
    Teuchos::RCP<THCM> ocean = Teuchos::rcp(new THCM(thcmList, Comm));

    // these LOCA data structures aren't really used by the OceanModel,
    // but are required for the ModelEvaluatorInterface

    // Create Epetra factory
    Teuchos::RCP<LOCA::Abstract::Factory> epetraFactory =
    Teuchos::rcp(new LOCA::Epetra::Factory);

    // Create global data object
    Teuchos::RCP<LOCA::GlobalData> globalData =
    LOCA::createGlobalData(paramList, epetraFactory);

    // the model serves as Preconditioner factory if "User Defined" is selected
    std::string PrecType = lsParams.get("Preconditioner","Ifpack");

    Teuchos::RCP<Teuchos::ParameterList> myPrecList=Teuchos::null;
    if (PrecType=="User Defined")
      {
      myPrecList = Teuchos::rcp(&lsParams,false);
      }

    // for some purposes it's good to know which one is the continuation parameter
    // (i.e. backup in regular intervals)
    thcmList.set("Parameter Name",cont_param);                                 

    // this is the LOCA interface (LOCA::Epetra::Interface::TimeDependent) to
    // our EpetraExt ModelEvaluator class 'OceanModel'.
    Teuchos::RCP<OceanModel> model =
      Teuchos::rcp(new OceanModel(thcmList,globalData,myPrecList));

    // this vector defines what parameters the problem depends on
    // (or at least which parameters may be varied)
    // and gives initial values for them which overwrite
    // the settings made in usrc.F90::stpnt(). 
    Teuchos::RCP<LOCA::ParameterVector> pVector = 
                model->getParameterVector();
for (int i=1;i<36;i++)
      {
     printf("%d th paramater`s value in Contination %f\n",i,pVector->getValue(i));

        }
    //Get the vector from the problem
    Teuchos::RCP<Epetra_Vector> soln = model->getSolution();

    //Initialize solution
    soln->PutScalar(0.0);

    // check for starting solution
    std::string StartConfigFile = thcmList.get("Starting Solution File","None");

    if (StartConfigFile!="None")
      {
      TIMER_START("Read Start File");
      soln=model->ReadConfiguration(StartConfigFile,*pVector);
      try
        {
        start_value = pVector->getValue(cont_param);
        } catch (...) {ERROR("Bad Continuation Parameter",__FILE__,__LINE__);}
      stepperList.set("Initial Value",start_value);
      model->setParameters(*pVector);
      TIMER_STOP("Read Start File");
      }
    //Create the Epetra_RowMatrix for the Jacobian/Preconditioner
    Teuchos::RCP<Epetra_CrsMatrix> A = model->getJacobian();

// NOX/LOCA interface setup

    Teuchos::RCP<NOX::Abstract::PrePostOperator> prepost = model;

    // register pre- and post operations
    nlParams.sublist("Solver Options").set("User Defined Pre/Post Operator",prepost);

    Teuchos::RCP<NOX::Epetra::LinearSystem> linsys = createLinearSystem
                (model, lsParams, nlPrintParams, cdataFile);
    // we use the same linear system for the shift-inverted operator
    Teuchos::RCP<NOX::Epetra::LinearSystem> shiftedLinSys = linsys;

    //Create the loca vector
    NOX::Epetra::Vector locaSoln(soln);

    // Create the Group
    Teuchos::RCP<LOCA::Epetra::Interface::TimeDependent> iReq = model;
    Teuchos::RCP<LOCA::Epetra::Group> grp =
      Teuchos::rcp(new LOCA::Epetra::Group(globalData, nlPrintParams,
                                           iReq, locaSoln, linsys, shiftedLinSys,
                                           *pVector));

    grp->computeF();

    double TolNewton = nlParams.get("Convergence Tolerance",1.0e-9);

    // Set up the Solver Convergence tests
    Teuchos::RCP<NOX::StatusTest::NormF> wrms =
       Teuchos::rcp(new NOX::StatusTest::NormF(TolNewton));
    Teuchos::RCP<NOX::StatusTest::MaxIters> maxiters
      = Teuchos::rcp(new NOX::StatusTest::MaxIters(searchParams.get("Max Iters", 10)));
    Teuchos::RCP<NOX::StatusTest::Combo> comboOR = 
       Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR)); 
    comboOR->addStatusTest(wrms);
    comboOR->addStatusTest(maxiters);


    // Create the stepper  
    LOCA::Stepper stepper(globalData, grp, comboOR, paramList);



paramList->print(*outFile);
//Teuchos::writeParameterListToXmlFile(*paramList,"parameters.xml");

INFO("\n*****************************");
INFO("Start Continuation process...");
INFO("*****************************\n\n");

TIMER_START("entire continuation run");

    // Perform continuation run
    LOCA::Abstract::Iterator::IteratorStatus status = stepper.run();


TIMER_STOP("entire continuation run");

    if (status != LOCA::Abstract::Iterator::Finished) {
      if (globalData->locaUtils->isPrintType(NOX::Utils::Error))
        globalData->locaUtils->out()
          << "Stepper failed to converge!" << std::endl;
    }
    globalData->locaUtils->out() << "Continuation status -> \t"<< status << std::endl;

    // Output the parameter list
    if (globalData->locaUtils->isPrintType(NOX::Utils::StepperParameters)) {
      globalData->locaUtils->out()
        << std::endl << "Final Parameters" << std::endl
        << "****************" << std::endl;
      stepper.getList()->print(globalData->locaUtils->out());
      globalData->locaUtils->out() << std::endl;
    }

    // Get the final solution from the stepper
    Teuchos::RCP<const LOCA::Epetra::Group> finalGroup = 
      Teuchos::rcp_dynamic_cast<const LOCA::Epetra::Group>(stepper.getSolutionGroup());
    const NOX::Epetra::Vector& finalSolutionNOX = 
      dynamic_cast<const NOX::Epetra::Vector&>(finalGroup->getX());
    const Epetra_Vector& finalSolution = finalSolutionNOX.getEpetraVector(); 


    // write final backup file for restarting
    model->setForceBackup(true);
    // also write THCM files (fort.3, fort.44, fort.15)
    model->setThcmOutput(true);
    model->printSolution(finalSolution, model->getContinuationParameterValue());
    Comm->Barrier(); // make sure the root process has written the files
    LOCA::destroyGlobalData(globalData);

  } 
  catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
  catch (const char*s) {
    std::cerr << s << std::endl;
  }
  catch (...) {
    std::cerr << "Caught unknown exception!" << std::endl;
  }

//end main
    return 0;

}

