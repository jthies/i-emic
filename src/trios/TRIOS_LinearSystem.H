#pragma once

#include "NOX_Epetra_LinearSystem.H"
#include "OceanModel.H"
#include "Teuchos_ParameterList.hpp"
#include <iostream>

namespace TRIOS {

  // create a linear system with solver for our THCM application.
  // note that you can create a Belos solver as well because the
  // LinearSystem_Belos class is derived from the AztecOO one.
  Teuchos::RCP<NOX::Epetra::LinearSystem> createLinearSystem
           (Teuchos::RCP<OceanModel> model, Teuchos::ParameterList& lsParams,
           Teuchos::ParameterList& printParams, Teuchos::RCP<std::ostream> out);

}
