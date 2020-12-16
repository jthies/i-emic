#ifndef HYMLS_FROSCH_PRECONDITIONER_H
#define HYMLS_FROSCH_PRECONDITIONER_H

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

#include "Ifpack_Preconditioner.h"
#include "Epetra_Time.h"

#include "HYMLS_config.h"
#include "HYMLS_PLA.hpp"

#include "Xpetra_CrsMatrix.hpp"
#include "FROSch_TwoLevelPreconditioner_decl.hpp"
#include "FROSch_TwoLevelBlockPreconditioner_decl.hpp"

#include "TRIOS_Domain.H"
#include "TpetraCore_config.h"


// forward declarations
class Epetra_Comm;
class Epetra_Map;
class Epetra_RowMatrix;
class Epetra_Import;

namespace Teuchos
  {
class ParameterList;
  }

namespace TRIOS {

class Domain;

//! FROSch solver class to use instead of HYMLS

class FROSchPreconditioner : public Ifpack_Preconditioner,
                           public HYMLS::PLA
{
// note: the global index type actually depends on how Tpetra is installed, it is no longer possible
// to just say you want int or whatever. Even though we use only Epetra, I got linker errors with
// just assuming gidx=int
#if defined(HAVE_TPETRA_INST_INT_INT)
typedef int gidx;
#elif defined(HAVE_TPETRA_INST_INT_LONG)
typedef long gidx;
#else
# error "Tpetra must be instantiated with either 'long' or 'int' global indices"
#endif

  // might want to force serial node here to avoid multi-threading where it doesn't make sense
  typedef KokkosClassic::DefaultNode::DefaultNodeType node_type;
  typedef Xpetra::Matrix<double,int,gidx> XMatrix;
  typedef Teuchos::RCP<XMatrix> XMatrixPtr;
  typedef Teuchos::RCP<const XMatrix> ConstXMatrixPtr;
  typedef Xpetra::MultiVector<double,int,gidx> XMultiVector;
  typedef Xpetra::Map<int,gidx> XMap;
  typedef FROSch::TwoLevelBlockPreconditioner<double, int, gidx> FROSchPrecType;
      
public:
  //!
  //! Constructor
  //!
  //! the caller should typically just use
  //!
  //! FROSchPreconditioner(K,domain,params);
  FROSchPreconditioner(Teuchos::RCP<const Epetra_RowMatrix> K,
    Teuchos::RCP<TRIOS::Domain> domain,
    Teuchos::RCP<Teuchos::ParameterList> params);

  //! destructor
  ~FROSchPreconditioner();

  //! Sets all parameters for the preconditioner.
  int SetParameters(Teuchos::ParameterList& List);

  //! Computes all it is necessary to initialize the preconditioner.
  int Initialize();

  //! Returns true if the  preconditioner has been successfully initialized, false otherwise.
  bool IsInitialized() const;

  //! Computes all it is necessary to apply the preconditioner.
  int Compute();

  //! Returns true if the  preconditioner has been successfully computed, false otherwise.
  bool IsComputed() const;
      
  //! Computes the condition number estimate, returns its value.
  double Condest(const Ifpack_CondestType CT = Ifpack_Cheap,
    const int MaxIters = 1550,
    const double Tol = 1e-9,
    Epetra_RowMatrix* Matrix = 0);

  //! Returns the computed condition number estimate, or -1.0 if not computed.
  double Condest() const;

  //! Applies the operator (not implemented)
  int Apply(const Epetra_MultiVector& X,
    Epetra_MultiVector& Y) const {return -1;}

  //! Applies the preconditioner to vector X, returns the result in Y.
  int ApplyInverse(const Epetra_MultiVector& X,
    Epetra_MultiVector& Y) const;

  //! Returns a pointer to the matrix to be preconditioned.
  const Epetra_RowMatrix& Matrix() const;

  //! Returns the number of calls to Initialize().
  int NumInitialize() const;

  //! Returns the number of calls to Compute().
  int NumCompute() const;

  //! Returns the number of calls to ApplyInverse().
  int NumApplyInverse() const;

  //! Returns the time spent in Initialize().
  double InitializeTime() const;

  //! Returns the time spent in Compute().
  double ComputeTime() const;

  //! Returns the time spent in ApplyInverse().
  double ApplyInverseTime() const;

  //! Returns the number of flops in the initialization phase.
  double InitializeFlops() const;

  //! Returns the number of flops in the computation phase.
  double ComputeFlops() const;

  //! Returns the number of flops in the application of the preconditioner.
  double ApplyInverseFlops() const;

  //! Prints basic information on iostream. This function is used by operator<<.
  std::ostream& Print(std::ostream& os) const {return os;};

  int SetUseTranspose(bool UseTranspose)
    {
    useTranspose_=false; // not implemented.
    return -1;
    }
  //! not implemented.
  bool HasNormInf() const {return false;}

  //! infinity norm
  double NormInf() const {return normInf_;}

  //! label
  const char* Label() const {return label_.c_str();}

  //! use transpose?
  bool UseTranspose() const {return useTranspose_;}

  //! communicator
  const Epetra_Comm & Comm() const {return *comm_;}

  //! Returns the Epetra_Map object associated with the domain of this operator.
  const Epetra_Map & OperatorDomainMap() const {return *rangeMap_;}

  //! Returns the Epetra_Map object associated with the range of this operator.
  const Epetra_Map & OperatorRangeMap() const {return *rangeMap_;}

  //@}

  //!\name Teuchos::ParameterListAcceptor
  //@{
  //!
  void setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& list);

  void FillMatrixGlobal();
  
  void FillMatrixLocal();

protected:

  //! function to create the null space information which defines how FROSch will generate
  //! it's coarse problem.
  //!
  //! The in put arguments are those that FROSch also requiress:
  //! - repeatedMaps contain one block of variables, e.g. u|v, w, each
  //!   and may have overlap if we want to apply the coarsening on the separator.
  //! - dofMaps contain one variable each. The array itself has the same number of
  //!   entries as repeatedMaps (the number of blocks), each of them has an array of
  //!   "scalar" maps, e.g. dofMaps[0][0] is the u-map, dofMaps[0][1] the v-map,
  //!   dofMaps[1][0] the w-map, dofMaps[2][0] the p-map, dofMaps[3][0] the T-map, and
  //!   dofMaps[3][1] the S-map. These will also contain overlap, I think.
  //!The dofsPerNodeVecotr contains {2,1,1,2} in our case (u/v, w, p, T/S).
  Teuchos::ArrayRCP<Teuchos::RCP<const XMultiVector> > BuildNullSpaces
    (Teuchos::ArrayRCP<Teuchos::RCP<const XMap> > const repeatedMaps,
     Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<const XMap> > > const dofMaps,
     Teuchos::ArrayRCP<const unsigned> const dofsPerNodeVector) const;

  //! communicator
  Teuchos::RCP<const Epetra_Comm> comm_;

  //! fake communicator
  Teuchos::RCP<const Epetra_Comm> serialComm_;

  //! matrix based on range map
  Teuchos::RCP<const Epetra_RowMatrix> matrix_;
  Teuchos::RCP<Epetra_CrsMatrix> matrixFixedPressure_;
  ConstXMatrixPtr matrix_X_;
  
  //! domain object for creating custom maps
  Teuchos::RCP<TRIOS::Domain> domain_;

  //! range/domain map of matrix_
  Teuchos::RCP<const Epetra_Map> rangeMap_;

  Teuchos::RCP<FROSchPrecType> prec_;

  //@}

  //! use transposed operator?
  bool useTranspose_;

  //! infinity norm
  double normInf_;

  //! label
  std::string label_;

  //! timer
  mutable Teuchos::RCP<Epetra_Time> time_;

  //! has Initialize() been called?
  bool initialized_;

  //! has Compute() been called?
  bool computed_;

  //! how often has Initialize() been called?
  int numInitialize_;

  //! how often has Compute() been called?
  int numCompute_;

  //! how often has ApplyInverse() been called?
  mutable int numApplyInverse_;

  //! flops during Initialize()
  double flopsInitialize_;

  //! flops during Compute()
  double flopsCompute_;

  //! flops during ApplyInverse()
  mutable double flopsApplyInverse_;

  //! time during Initialize()
  mutable double timeInitialize_;

  //! time during Compute()
  mutable double timeCompute_;

  //! time during ApplyInverse()
  mutable double timeApplyInverse_;

  //! time to convert from Epetra to Xpetra and back
  mutable double timeConvert_;

  //!@}


  //! geometric info, used for storing the ordering in MATLAB for visualization
  int dim_,dof_;
  };

}//namespace TRIOS
#endif
