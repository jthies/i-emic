#ifndef NOX_EPETRA_LINEARSYSTEMBELOS_H
#define NOX_EPETRA_LINEARSYSTEMBELOS_H

#include "BelosConfigDefs.hpp"
#include "NOX_Epetra_LinearSystem_AztecOO.H"
#include "Teuchos_ScalarTraits.hpp"
#include "BelosMultiVecTraits.hpp"
#include "BelosOperatorTraits.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosSolverManager.hpp"
#include "BelosEpetraAdapter.hpp"
#include "BelosOutputManager.hpp"

class Epetra_MultiVector;
class Epetra_Operator;

typedef double ST;
typedef Teuchos::ScalarTraits<ST> SCT;
typedef SCT::magnitudeType MT;
typedef Epetra_MultiVector MV;
typedef Epetra_Operator OP;
typedef Belos::MultiVecTraits<ST,MV> MVT;
typedef Belos::OperatorTraits<ST,MV,OP> OPT;


namespace NOX {
//! Improved version of the Epetra support class.
namespace Epetra {

/*! This class overloads LinearSystem_AztecOO, adding the  
    possibility to use Belos for linear systems instead of 
    AztecOO. Right now the Aztec solver will still be      
    constructed, too. Belos parameters can be set in the   
    "Linear Solver"->"Belos" sublist.                      
 */
class LinearSystemBelos : public LinearSystemAztecOO 
  {

public:
  //! Constructor with no Operators.  
  /*! Jacobian Operator will be constructed internally based on the
    parameter "Jacobian Operator".  Defaults to using a 
    NOX::Epetra::MatrixFree object.
   */
  LinearSystemBelos(
    Teuchos::ParameterList& printingParams, 
    Teuchos::ParameterList& linearSolverParams,
    const Teuchos::RCP<NOX::Epetra::Interface::Required>& iReq, 
    const NOX::Epetra::Vector& cloneVector,
    const Teuchos::RCP<NOX::Epetra::Scaling> scalingObject = 
    Teuchos::null);

  //! Constructor with a user supplied Jacobian Operator only.  
  /*! Either there is no preconditioning or the preconditioner will be 
    used/created internally.  The Jacobian (if derived from an 
    Epetra_RowMatrix class can be used with an internal preconditioner. 
    See the parameter key "Preconditioner Operator" for more details.
   */
  LinearSystemBelos(
    Teuchos::ParameterList& printingParams, 
    Teuchos::ParameterList& linearSolverParams, 
    const Teuchos::RCP<NOX::Epetra::Interface::Required>& iReq, 
    const Teuchos::RCP<NOX::Epetra::Interface::Jacobian>& iJac, 
    const Teuchos::RCP<Epetra_Operator>& J,
    const NOX::Epetra::Vector& cloneVector,
    const Teuchos::RCP<NOX::Epetra::Scaling> scalingObject = 
    Teuchos::null);

  //! Constructor with a user supplied Preconditioner Operator only.  
  /*! Jacobian operator will be constructed internally based on the
    parameter "Jacobian Operator" in the parameter list.  See the 
    parameter key "Jacobian Operator" for more details.  Defaults 
    to using a NOX::Epetra::MatrixFree object.
   */
  LinearSystemBelos(
    Teuchos::ParameterList& printingParams, 
    Teuchos::ParameterList& linearSolverParams, 
    const Teuchos::RCP<NOX::Epetra::Interface::Required>& i, 
    const Teuchos::RCP<NOX::Epetra::Interface::Preconditioner>& iPrec,
    const Teuchos::RCP<Epetra_Operator>& M,
    const NOX::Epetra::Vector& cloneVector,
    const Teuchos::RCP<NOX::Epetra::Scaling> scalingObject = 
    Teuchos::null);
  
  //! Constructor with user supplied separate objects for the
  //! Jacobian (J) and Preconditioner (M).  
  //! linearSolverParams is the "Linear Solver" sublist of parameter list.
  LinearSystemBelos(
    Teuchos::ParameterList& printingParams, 
    Teuchos::ParameterList& linearSolverParams, 
    const Teuchos::RCP<NOX::Epetra::Interface::Jacobian>& iJac, 
    const Teuchos::RCP<Epetra_Operator>& J,  
    const Teuchos::RCP<NOX::Epetra::Interface::Preconditioner>& iPrec, 
    const Teuchos::RCP<Epetra_Operator>& M,
    const NOX::Epetra::Vector& cloneVector,
    const Teuchos::RCP<NOX::Epetra::Scaling> scalingObject = 
    Teuchos::null);

  //! Destructor.
  virtual ~LinearSystemBelos();


  virtual bool applyJacobianInverse(Teuchos::ParameterList& linearSolverParams, 
				    const NOX::Epetra::Vector& input, 
				    NOX::Epetra::Vector& result);

  //! Reset the linear solver parameters.
  virtual void reset(Teuchos::ParameterList& linearSolverParams);

protected:
  
  /*! \brief Sets the epetra Jacobian operator in the Belos object.
     
      It is still called setAztecOO* because NOX/LOCA call that function.
  */

  virtual void setAztecOOJacobian() const;

  /*! \brief Sets the epetra Preconditioner operator in the Belos object.
    
      It is still called setAztecOO* because NOX/LOCA call that function.
  */
  virtual void setAztecOOPreconditioner() const;

protected:

//!\name Belos data structures
//@{

  //! right-hand side vector
  Teuchos::RCP<MV> belosRhs;
  
  //! solution vector
  Teuchos::RCP<MV> belosSol;  
  
  //! Belos preconditioner interface
  mutable Teuchos::RCP<Belos::EpetraPrecOp > belosPrecPtr;

  //! Belos linear problem interface
  Teuchos::RCP<Belos::LinearProblem<ST,MV,OP> > belosProblemPtr;

  //! Belos solver
  Teuchos::RCP<Belos::SolverManager<ST,MV,OP> > belosSolverPtr;

  //! Belos output manager (not sure if this is actually needed)
  Teuchos::RCP<Belos::OutputManager<ST> > belosOutputPtr;
  
//@}

};

} // namespace Epetra
} // namespace NOX


#endif
