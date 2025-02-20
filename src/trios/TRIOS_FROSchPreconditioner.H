#ifndef TRIOS_FROSCH_PRECONDITIONER_H
#define TRIOS_FROSCH_PRECONDITIONER_H

#include "GlobalDefinitions.H"
#include "TRIOS_Domain.H"

#include "Epetra_Operator.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_ParameterListAcceptor.hpp"

#ifdef HAVE_FROSCH

#include "Xpetra_Map.hpp"
#include "Xpetra_MultiVector.hpp"
#include "Xpetra_CrsMatrix.hpp"
#include "Xpetra_Operator.hpp"
#include "Xpetra_CrsMatrix.hpp"

#include "ShyLU_DDFROSch_config.h"
#include "FROSch_SchwarzPreconditioners_fwd.hpp"

class Epetra_Map;
class Epetra_Vector;
class Epetra_MultiVector;
class Epetra_CrsMatrix;

using ST = double;
using LO = int;
using GO = int;
#using serial = Tpetra::KokkosCompat::KokkosSerialWrapperNode;
using Xpetra::EpetraNode;
using Xpetra_Map        = Xpetra::Map<LO,GO,EpetraNode>;
using Xpetra_MultiVector= Xpetra::MultiVector<ST,LO,GO,EpetraNode>;
using Xpetra_Operator   = Xpetra::Operator<ST,LO,GO,EpetraNode>;
using Xpetra_CrsMatrix  = Xpetra::CrsMatrix<ST,LO,GO,EpetraNode>;

using OneLevelFROSch = FROSch::OneLevelPreconditioner<ST,LO,GO,EpetraNode>;
using TwoLevelFROSch = FROSch::TwoLevelPreconditioner<ST,LO,GO,EpetraNode>;

namespace TRIOS
{

class FROSchPreconditioner: public Epetra_Operator,
                            public Epetra_Object,
                            public Teuchos::ParameterListAcceptor
{
public:

  FROSchPreconditioner() = delete;

  FROSchPreconditioner(Teuchos::RCP<const Epetra_CrsMatrix> A,
                       Teuchos::RCP<const Domain> domain,
                       Teuchos::ParameterList& froschList);

  ~FROSchPreconditioner() {}

  int SetUseTranspose(bool UseTranspose)
  {
    useTranspose_=true;
    return 0;
  }

    int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

    int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y);

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

  private:

    bool useTranspose_;
    bool isComputed_;

    //! Geometry and partitioning information
    Teuchos::RCP<const Domain> domain_;

    //! \name Epetra objects inherited from the matrix passed in
    //@{

    const Epetra_Comm& epetraComm_;

    const Epetra_Map& epetraMap_;

    const Epetra_CrsMatrix& epetraMatrix_;

    //@}

    //! \name Xpetra objects to wrap Epetra objects and FROSch preconditioner
    //@{

    Teuchos::RCP<const Xpetra_Map> xpetraMap_;

    Teuchos::RCP<const Xpetra_CrsMatrix> xpetraMatrix_;

    //@}


};

} //namespace TRIOS

#endif /* HAVE_FROSCH */

#endif

