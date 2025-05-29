#include "TRIOS_LinearSystem.H"

namespace TRIOS {

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

}
