#ifndef TRIOS_FROSCH_PRECONDITIONER_H
#define TRIOS_FROSCH_PRECONDITIONER_H

#include "GlobalDefinitions.H"
#include "TRIOS_Domain.H"

#include "Epetra_Operator.h"
#include "Xpetra_Map.hpp"
#include "Xpetra_MultiVector.hpp"
#include "Xpetra_CrsMatrix.hpp"
#include "Xpetra_Operator.hpp"
#include "Xpetra_CrsMatrix.hpp"

#include "FROSch_OneLevelPreconditioner_decl.hpp"
#include "FROSch_TwoLevelPreconditioner_decl.hpp"

class Epetra_Map;
class Epetra_Vector;
class Epetra_MultiVector;
class Epetra_CrsMatrix;

using ST = double;
using LO = int;
using GO = int;
using node_type = Tpetra::KokkosCompat::KokkosSerialWrapperNode;
using Xpetra_Map        = Xpetra::Map<LO,GO,node_type>;
using Xpetra_MultiVector= Xpetra::MultiVector<ST,LO,GO,node_type>;
using Xpetra_Operator   = Xpetra::Operator<ST,LO,GO,node_type>;
using Xpetra_CrsMatrix  = Xpetra::CrsMatrix<ST,LO,GO,node_type>;


using OneLevelFROSch = FROSch::OneLevelPreconditioner<ST,LO,GO,node_type>;
using TwoLevelFROSch = FROSch::TwoLevelPreconditioner<ST,LO,GO,,node_type>;

namespace TRIOS
{

class FROSchPreconditioner: public Epetra_Operator
{
public:

  FROSchPreconditioner(Teuchos::RCP<const Epetra_CrsMatrix> A,
                       Teuchos::RCP<Domain> domain,
                       Teuchos::ParameterList& froschList);

  ~FROSchPreconditioner() {}

  int SetUseTranspose(bool UseTranspose)
  {
    useTranspose_=true;
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

    //! \name Epetra objects inherited from the matrix passed in
    //@{

    const Epetra_Comm& epetraComm_;

    const Epetra_Map& epetraMap_;

    const Epetra_CrsMatrix& epetraMatrix_;

    //@}

    //! \name Xpetra objects to wrap Epetra objects and FROSch preconditioner
    //@{

    const Xpetra_Comm& xpetraComm_;

    const Xpetra_Map& xpetraMap_;

    const Xpetra_CrsMatrix& xpetraMatrix_;

    //@}


};

#endif

