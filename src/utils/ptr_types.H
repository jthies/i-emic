/**********************************************************************
 * Copyright by Jonas Thies, Univ. of Groningen 2006/7/8.             *
 * Permission to use, copy, modify, redistribute is granted           *
 * as long as this header remains intact.                             *
 * contact: jonas@math.rug.nl                                         *
 **********************************************************************/
#ifndef POINTER_TYPES_H
#define POINTER_TYPES_H

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "NOX_Epetra_Scaling.H"
#include "LOCA_Parameter_Vector.H"


//! can be included to make some useful pointer types available,
//! this is just for convenience as we make frequent use of the 
//! Teuchos::RCP<T> class and don't want to write that  
//! out all the time

//! \note when using pointers to type X you still have to include 
//!       the corresponding header file as this file contains only
//!       forward declarations

//! forward declaration of the most common targets for rcp's:
class Epetra_Map;
class Epetra_Import;
class Epetra_Vector;
class Epetra_IntVector;
class Epetra_CrsMatrix;
class Epetra_Operator;
class Epetra_LinearProblem;
class OceanDomain;
class THCM_Interface;
class Epetra_Comm;
class AztecOO;

typedef Teuchos::RCP<Epetra_Comm> comm_ptr;
typedef Teuchos::RCP<Epetra_Map> map_ptr;
typedef Teuchos::RCP<Epetra_Import> import_ptr;
typedef Teuchos::RCP<Epetra_Vector> vec_ptr;
typedef Teuchos::RCP<Epetra_IntVector> intvec_ptr;
typedef Teuchos::RCP<Epetra_CrsMatrix> mat_ptr;
typedef Teuchos::RCP<Epetra_Operator> op_ptr;
typedef Teuchos::RCP<AztecOO> solver_ptr;
typedef Teuchos::RCP<std::ostream> stream_ptr;
typedef Teuchos::RCP<NOX::Epetra::Scaling> scaling_ptr;

// make some commonly used types directly available
using Teuchos::rcp;
using Teuchos::RCP;
using Teuchos::null;
using Teuchos::ParameterList;
using LOCA::ParameterVector;
#endif
