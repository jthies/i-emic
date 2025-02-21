#ifndef TRIOS_FROSCH_PRECONDITIONER_H
#define TRIOS_FROSCH_PRECONDITIONER_H

#include "GlobalDefinitions.H"
#include "TRIOS_Domain.H"

#include "Epetra_Operator.h"
#include "Epetra_CrsMatrix.h"
#include "Ifpack_Preconditioner.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_ParameterListAcceptorDefaultBase.hpp"

#ifdef HAVE_FROSCH

#include "Xpetra_Map.hpp"
#include "Xpetra_MultiVector.hpp"
#include "Xpetra_CrsMatrix.hpp"
#include "Xpetra_Operator.hpp"
#include "Xpetra_CrsMatrix.hpp"
#include "Xpetra_CrsMatrixWrap.hpp"

#include "ShyLU_DDFROSch_config.h"
#include "FROSch_SchwarzPreconditioners_fwd.hpp"

class Epetra_Map;
class Epetra_Vector;
class Epetra_MultiVector;

using ST = double;
using LO = int;
using GO = int;
using Xpetra::EpetraNode;
using Xpetra_Map        = Xpetra::Map<LO,GO,EpetraNode>;
using Xpetra_MultiVector= Xpetra::MultiVector<ST,LO,GO,EpetraNode>;
using Xpetra_Operator   = Xpetra::Operator<ST,LO,GO,EpetraNode>;
using Xpetra_CrsMatrixWrap  = Xpetra::CrsMatrixWrap<ST,LO,GO,EpetraNode>;

using OneLevelFROSch = FROSch::OneLevelPreconditioner<ST,LO,GO,EpetraNode>;
using TwoLevelFROSch = FROSch::TwoLevelPreconditioner<ST,LO,GO,EpetraNode>;

namespace TRIOS
{

class FROSchPreconditioner: public Ifpack_Preconditioner
{
public:

  FROSchPreconditioner() = delete;

  FROSchPreconditioner(Teuchos::RCP<Epetra_CrsMatrix> A,
                       Teuchos::RCP<const Domain> domain,
                       Teuchos::ParameterList& froschList);

  ~FROSchPreconditioner() {}

  // \name Epetra_Operator interface
  //!@{ 

  int SetUseTranspose(bool UseTranspose)
  {
    useTranspose_=true;
    return 0;
  }

    int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

    int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

    double NormInf() const
    {
      ERROR("not implemented.", __FILE__, __LINE__);
    }

    const char * Label() const
    {
      return "TRIOS::FROSchPreconditioner";
    }

    bool UseTranspose() const
    {
      return useTranspose_;
    }

    bool HasNormInf() const
    {
      return false;
    }

    //! Returns a pointer to the Epetra_Comm communicator associated with this operator.
    const Epetra_Comm & Comm() const
    {
      return epetraComm_;
    }

    const Epetra_Map & OperatorDomainMap() const
    {
      return epetraMap_;
    }

    //! Returns the Epetra_Map object associated with the range of this operator.
    const Epetra_Map & OperatorRangeMap() const
    {
      return epetraMap_;
    }

    //@}

    //! \name Ifpack_Preconditioner interface
    //@{
    int SetParameters(Teuchos::ParameterList& List);

    //! Computes all it is necessary to initialize the preconditioner.
    int Initialize();

  //! Returns true if the  preconditioner has been successfully initialized, false otherwise.
  bool IsInitialized() const
  {
    return isInitialized_;
  }

  //! Computes all it is necessary to apply the preconditioner.
  int Compute();

  //! Returns true if the  preconditioner has been successfully computed, false otherwise.
  virtual bool IsComputed() const
  {
    return isComputed_;
  }

  //! Computes the condition number estimate, returns its value.
  double Condest(const Ifpack_CondestType CT = Ifpack_Cheap,
                         const int MaxIters = 1550,
                         const double Tol = 1e-9,
                         Epetra_RowMatrix* Matrix = 0)
  {
    ERROR("not implemented", __FILE__, __LINE__);
  }

  //! Returns the computed condition number estimate, or -1.0 if not computed.
  double Condest() const
  {
    ERROR("not implemented", __FILE__, __LINE__);
  }

  //! Returns a pointer to the matrix to be preconditioned.
  const Epetra_RowMatrix& Matrix() const
  {
    return epetraMatrix_;
  }

  //! Returns the number of calls to Initialize().
  int NumInitialize() const
  {
    ERROR("not implemented", __FILE__, __LINE__);
  }

  //! Returns the number of calls to Compute().
  int NumCompute() const
  {
    ERROR("not implemented", __FILE__, __LINE__);
  }

  //! Returns the number of calls to ApplyInverse().
  virtual int NumApplyInverse() const
  {
    ERROR("not implemented", __FILE__, __LINE__);
  }

  //! Returns the time spent in Initialize().
  double InitializeTime() const
  {
    ERROR("not implemented", __FILE__, __LINE__);
  }

  //! Returns the time spent in Compute().
  double ComputeTime() const
  {
    ERROR("not implemented", __FILE__, __LINE__);
  }

  //! Returns the time spent in ApplyInverse().
  double ApplyInverseTime() const
  {
    ERROR("not implemented", __FILE__, __LINE__);
  }

  //! Returns the number of flops in the initialization phase.
  double InitializeFlops() const
  {
    ERROR("not implemented", __FILE__, __LINE__);
  }

  //! Returns the number of flops in the computation phase.
  double ComputeFlops() const
  {
    ERROR("not implemented", __FILE__, __LINE__);
  }

  //! Returns the number of flops in the application of the preconditioner.
  double ApplyInverseFlops() const
  {
    ERROR("not implemented", __FILE__, __LINE__);
  }

  std::ostream& Print(std::ostream& os) const
  {
    os << Label() << std::endl;
    return os;
  }
    //@}

  private:

    //! parameters passed to the FROSch preconditioner
    Teuchos::ParameterList& pList_;

    bool useTranspose_;
    bool isInitialized_, isComputed_;

    //! Geometry and partitioning information
    Teuchos::RCP<const Domain> domain_;

    //! \name Epetra objects inherited from the matrix passed in
    //@{

    const Epetra_Comm& epetraComm_;

    const Epetra_Map& epetraMap_;

    // note: this should be a const ref, but Xpetra
    // does not allow wrapping const CrsMatrices (interface flaw)
    Epetra_CrsMatrix& epetraMatrix_;

    //@}

    //! \name Xpetra objects to wrap Epetra objects and FROSch preconditioner
    //@{

    Teuchos::RCP<const Xpetra_Map> xpetraMap_;

    Teuchos::RCP<const Xpetra_CrsMatrixWrap> xpetraMatrix_;

    //@}

    //! Pointer to the actual preconditioner object.

    //! Note that this can be either a one- or two-level FROSch preconditioner
    //! as the two-level class is derived from the one-level class.
    Teuchos::RCP<OneLevelFROSch> frosch_;

};

} //namespace TRIOS

#endif /* HAVE_FROSCH */

#endif

