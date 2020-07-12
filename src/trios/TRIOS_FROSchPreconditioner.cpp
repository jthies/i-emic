
#include "TRIOS_FROSchPreconditioner.hpp"
#include "TRIOS_Domain.H"
#include "TRIOS_Macros.H"
#include "Utils.H"

#include "HYMLS_Macros.hpp"
#include "HYMLS_Tools.hpp"

#include <EpetraExt_RowMatrixOut.h>
#include "FROSch_TwoLevelBlockPreconditioner_def.hpp"
#include "Xpetra_CrsMatrixWrap.hpp"
#include <unistd.h>

using namespace ocean_defs;

namespace TRIOS {

// constructor
FROSchPreconditioner::FROSchPreconditioner(Teuchos::RCP<const Epetra_RowMatrix> K,
  Teuchos::RCP<TRIOS::Domain> domain,
  Teuchos::RCP<Teuchos::ParameterList> params)
  :
  domain_(domain),
  PLA(),
  comm_(Teuchos::rcp(&(K->Comm()), false)), matrix_(K),
  rangeMap_(Teuchos::rcp(&(K->RowMatrixRowMap()), false)),
  useTranspose_(false), normInf_(-1.0),
  label_("FROSchPreconditioner"),
  initialized_(false), computed_(false),
  numInitialize_(0), numCompute_(0), numApplyInverse_(0),
  flopsInitialize_(0.0), flopsCompute_(0.0), flopsApplyInverse_(0.0),
  timeInitialize_(0.0), timeCompute_(0.0), timeApplyInverse_(0.0), timeConvert_(0.0)
  {
  HYMLS_PROF3(label_,"Constructor");
      numberComputes_=0;
  time_=Teuchos::rcp(new Epetra_Time(K->Comm()));

  Teuchos::RCP<const Epetra_CrsMatrix> K_crs = Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_);

//      matrixFixedPressure_ = Teuchos::rcp(new Epetra_CrsMatrix( Epetra_DataAccess::Copy, K_crs->RowMap(), K->MaxNumEntries() ) );
  
//  FillMatrixGlobal();
      

      
//      K_crs->Print(std::cout);
  //TODO: don't understand how to create an Xpetra wrapper for a const object
  Teuchos::RCP<Epetra_CrsMatrix> K_crs_nonconst = Teuchos::rcp_const_cast<Epetra_CrsMatrix>(K_crs);
  time_->ResetStartTime();
  // this is a wrapper to turn the object into an Xpetra object
  Teuchos::RCP<Xpetra::CrsMatrix<double, int, hymls_gidx,node_type> > K_x =
        Teuchos::rcp(new Xpetra::EpetraCrsMatrixT<hymls_gidx,node_type>(K_crs_nonconst));
  // this is an Xpetra::Matrix that allows 'viewing' the matrix like a block matrix, for instance
  matrix_X_ = Teuchos::rcp(new Xpetra::CrsMatrixWrap<double,int,hymls_gidx,node_type>(K_x));
  timeConvert_+=time_->ElapsedTime();
//
      Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
      
  setParameterList(params);
  }

// destructor
FROSchPreconditioner::~FROSchPreconditioner()
  {
  HYMLS_PROF3(label_,"Destructor");
  }

//TODO: parameters are not validated

// Sets all parameters for the preconditioner.
int FROSchPreconditioner::SetParameters(Teuchos::ParameterList& List)
  {
  HYMLS_PROF3(label_,"SetParameters");

  Teuchos::RCP<Teuchos::ParameterList> List_ =
    getMyNonconstParamList();

  if (List_==Teuchos::null)
    {
    setMyParamList(Teuchos::rcp(&List, false));
    }
  else if (List_.get()!=&List)
    {
    List_->setParameters(List);
    }

  Teuchos::ParameterList& probList = List_->sublist("Problem");

  dim_ = probList.get("Dimension", -1);
  dof_ = probList.get("Degrees of Freedom", dim_+1);

  if (dim_ == -1 || dof_ == -1)
    {
    HYMLS::Tools::Error("Please set both the Dimension and Degrees of Freedom parameters",
      __FILE__, __LINE__);
    }

  PL().set("Dimension",dim_);
  PL().set("DofsPerNode1",dim_); // u/v/w/p
  PL().set("DofOrdering1", "NodeWise");
  PL().set("DofsPerNode2",2); // T, S
  PL().set("DofOrdering2", "NodeWise");
  // TODO: need to tell FROSch that there's a constant pressure mode,
  // but this is not yet in the 'develop' branch of Trilinos
  PL().set("Null Space Type", "Laplace");

  return 0;
  }

//!
void FROSchPreconditioner::setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& list)
  {
  HYMLS_PROF3(label_,"setParameterList");
  setMyParamList(list);
  CHECK_ZERO(SetParameters(*list));
  }


// Computes all it is necessary to initialize the preconditioner.
int FROSchPreconditioner::Initialize()
{
  HYMLS_PROF(label_,"Initialize");

  time_->ResetStartTime();
  
  if (dim_!=3|| dof_!=6) HYMLS::Tools::Error("currently only implemented for 3D case with dof=6",__FILE__,__LINE__);

  // we start by creating some maps for the separate variables and groups of variables.
  Teuchos::RCP<Epetra_Map> u_map, v_map, w_map, p_map, t_map, s_map, uv_map, ts_map;
  

  // we define for each direction i0 as the first index incl. the separator, and i1 as the last.
  //
  // In 1D: (o is the pressure, | the cell boundary with the velcoity)
  //          i0      i1
  // | o | o || o | o |
  // i0      i1
  int i0=domain_->FirstRealI(), i1=domain_->LastRealI();
  int j0=domain_->FirstRealJ(), j1=domain_->LastRealJ();
  int k0=domain_->FirstRealK(), k1=domain_->LastRealK();

  int N=domain_->GlobalN();
  int M=domain_->GlobalM();
  int L=domain_->GlobalL();

  w_map = Utils::CreateSubMap(*rangeMap_,dof_,WW);
  p_map = Utils::CreateSubMap(*rangeMap_,dof_,PP);
  // TODO: probably T and S should have minimal overlap as well so that
  // a separator is found?
  t_map = Utils::CreateSubMap(*rangeMap_,dof_,TT);
  s_map = Utils::CreateSubMap(*rangeMap_,dof_,SS);
  const int TS[2]={TT,SS};
  ts_map = Utils::CreateSubMap(*rangeMap_,dof_,TS);
  
  // velocities: add separators on `left' side
  if (domain_->FirstI()<domain_->FirstRealI()) i0--;
  if (domain_->FirstJ()<domain_->FirstRealJ()) j0--;
  if (domain_->FirstK()<domain_->FirstRealK()) k0--;

  // a map with all dofs per cell and a single layer of overlap, from which we can extract components as before
  Teuchos::RCP<Epetra_Map> overlappingMap = Utils::CreateMap(i0,i1,j0,j1,k0,k1,
                                                               0, N, 0, M, 0, L,
                                                               dof_,
                                                               rangeMap_->Comm());
  u_map = Utils::CreateSubMap(*overlappingMap,dof_,UU);
  v_map = Utils::CreateSubMap(*overlappingMap,dof_,VV);
  const int UV[2]={UU,VV};
  uv_map = Utils::CreateSubMap(*overlappingMap,dof_,UV);

        
  // data structures to pass to FROSch for describing the problem structure.
  // 'repeated' maps have overlap between partitions, we have that at least for the
  // horizontal velocities, but may want to add overlap for T and S.
    Teuchos::ArrayRCP<unsigned> dofsPerNodeVector(4);
  Teuchos::ArrayRCP<FROSch::DofOrdering> dofOrderings(4, FROSch::NodeWise);
      
  dofsPerNodeVector[0] = 2; // (u,v), located at vertical edge centers
  dofsPerNodeVector[1] = 1; // w, located at the center of the top face
  dofsPerNodeVector[2] = 1; // p, located at cell center
  dofsPerNodeVector[3] = 2; // (T,S), located at cell center

  // All maps are wrapped using Xpetra first.
  Teuchos::ArrayRCP<Teuchos::RCP<const XMap > > velocityMaps(2);
  Teuchos::ArrayRCP<Teuchos::RCP<const XMap > > pressureMaps(1);
  Teuchos::ArrayRCP<Teuchos::RCP<const XMap > > tracerMaps(2);

  Teuchos::ArrayRCP<Teuchos::RCP<const XMap> > repeatedMaps(3);
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<const XMap> > > dofMaps(3);

  const Epetra_MpiComm& tmpComm = dynamic_cast<const Epetra_MpiComm&> (*comm_);
  Teuchos::RCP<const Teuchos::Comm<int> > teuchosComm = Teuchos::rcp(new Teuchos::MpiComm<int> (tmpComm.Comm()));

  repeatedMaps[0] = FROSch::ConvertToXpetra<double,int,hymls_gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *uv_map, teuchosComm );
  repeatedMaps[1] = FROSch::ConvertToXpetra<double,int,hymls_gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *w_map,  teuchosComm );
  repeatedMaps[2] = FROSch::ConvertToXpetra<double,int,hymls_gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *p_map,  teuchosComm );
  repeatedMaps[3] = FROSch::ConvertToXpetra<double,int,hymls_gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *ts_map, teuchosComm );
  
  velocityMaps[0] = FROSch::ConvertToXpetra<double,int,hymls_gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *u_map, teuchosComm );
  velocityMaps[1] = FROSch::ConvertToXpetra<double,int,hymls_gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *v_map, teuchosComm );
  velocityMaps[2] = FROSch::ConvertToXpetra<double,int,hymls_gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *w_map, teuchosComm );
  pressureMaps[0] = FROSch::ConvertToXpetra<double,int,hymls_gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *p_map, teuchosComm ); 
  tracerMaps[0] = FROSch::ConvertToXpetra<double,int,hymls_gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *t_map, teuchosComm ); 
  tracerMaps[1] = FROSch::ConvertToXpetra<double,int,hymls_gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *s_map, teuchosComm ); 

  dofMaps[0] = velocityMaps;
  dofMaps[1] = pressureMaps;
  dofMaps[2] = tracerMaps;

  //Teuchos::RCP<Teuchos::FancyOStream> fancy = fancyOStream(Teuchos::rcpFromRef(std::cout));
  //pressureMaps[0]->describe(*fancy,Teuchos::VERB_EXTREME);
    
  // TODO: construct null spaces for each variable, I assume that we need
  // checkerboard modes (two vectors) for the pressure. u, v, T and S should
  // be treated like standard Laplace problems. For w the null space can probably
  // be constructed from the matrix.
  Teuchos::ArrayRCP< Teuchos::RCP<const XMultiVector > > nodesDummy = Teuchos::null;
  Teuchos::ArrayRCP< Teuchos::RCP<const XMultiVector > > nullSpace = Teuchos::null;

  // create the preconditioner
  prec_ = Teuchos::rcp(new FROSchPrecType(Teuchos::rcp_const_cast<XMatrix>(matrix_X_), Teuchos::rcpFromRef(PL())));
          
  int overlap=1;
          
  prec_->initialize(dim_,dofsPerNodeVector,dofOrderings,overlap,repeatedMaps,nullSpace,nodesDummy,dofMaps);
      
  initialized_ = true;
  numInitialize_++;
  timeInitialize_+=time_->ElapsedTime();
      
  return 0;
}

// Returns true if the  preconditioner has been successfully initialized, false otherwise.
bool FROSchPreconditioner::IsInitialized() const {return initialized_;}

// Computes all it is necessary to apply the preconditioner.
int FROSchPreconditioner::Compute()
  {
  HYMLS_PROF(label_,"Compute");
  //if (!IsInitialized())
    {
    HYMLS::Tools::Warning("FROSchPreconditioner currently has to be re-initialized in every compute(), otherwise nothing happens.",
      __FILE__,__LINE__);
/*
    // the user should normally call Initialize before Compute
    HYMLS::Tools::Warning("FROSchPreconditioner not initialized. I'll do it for you.",
      __FILE__,__LINE__);
 */
    CHECK_ZERO(Initialize());
    }

  time_->ResetStartTime();
//      std::string outName = "hymlsA_compute" + std::to_string(numberComputes_) + ".dat";
//      const char *cstr = outName.c_str();
//      EpetraExt::RowMatrixToMatlabFile(cstr,*matrix_);
//      std::cout << "Exported compute!"<< std::endl;

//      FillMatrixLocal();
      
//    std::string outName = "hymlsA_compute" + std::to_string(numberComputes_) + ".dat";
//    const char *cstr = outName.c_str();
//    EpetraExt::RowMatrixToMatlabFile(cstr,*matrix_);
//    numberComputes_++;
      
  prec_->compute();
 
  computed_ = true;
  timeCompute_ += time_->ElapsedTime();
  numCompute_++;

  return 0;
  }


// Returns true if the  preconditioner has been successfully computed, false otherwise.
bool FROSchPreconditioner::IsComputed() const {return computed_;}

// Applies the preconditioner to vector X, returns the result in Y.
int FROSchPreconditioner::ApplyInverse(const Epetra_MultiVector& B,
  Epetra_MultiVector& X) const
  {
  HYMLS_PROF(label_,"ApplyInverse");

  time_->ResetStartTime();
  Teuchos::RCP<const XMultiVector> _B = Teuchos::rcp(new
        Xpetra::EpetraMultiVectorT<hymls_gidx, node_type>(Teuchos::rcpFromRef(const_cast<Epetra_MultiVector&>(B))));
  Teuchos::RCP<XMultiVector> _X = Teuchos::rcp(new
        Xpetra::EpetraMultiVectorT<hymls_gidx, node_type>(Teuchos::rcpFromRef(X)));
  timeConvert_+=time_->ElapsedTime();

  numApplyInverse_++;
  time_->ResetStartTime();

  prec_->apply(*_B, *_X);
  timeApplyInverse_+=time_->ElapsedTime();
  return 0;
  }

// Returns a pointer to the matrix to be preconditioned.
const Epetra_RowMatrix& FROSchPreconditioner::Matrix() const {return *matrix_;}

//TODO: Implement the functions below

double FROSchPreconditioner::InitializeFlops() const
  {
  return 0;
  }

double FROSchPreconditioner::ComputeFlops() const
  {
  return 0;
  }

double FROSchPreconditioner::ApplyInverseFlops() const
  {
  return 0;
  }

// Computes the condition number estimate, returns its value.
double FROSchPreconditioner::Condest(const Ifpack_CondestType CT,
  const int MaxIters,
  const double Tol,
  Epetra_RowMatrix* Matrix)
  {
  HYMLS::Tools::Warning("not implemented!",__FILE__,__LINE__);
  return -1.0; // not implemented.
  }

// Returns the computed condition number estimate, or -1.0 if not computed.
double FROSchPreconditioner::Condest() const
  {
  HYMLS::Tools::Warning("not implemented!",__FILE__,__LINE__);
  return -1.0;
  }

// Returns the number of calls to Initialize().
int FROSchPreconditioner::NumInitialize() const {return numInitialize_;}

// Returns the number of calls to Compute().
int FROSchPreconditioner::NumCompute() const {return numCompute_;}

// Returns the number of calls to ApplyInverse().
int FROSchPreconditioner::NumApplyInverse() const {return numApplyInverse_;}

// Returns the time spent in Initialize().
double FROSchPreconditioner::InitializeTime() const {return timeInitialize_;}

// Returns the time spent in Compute().
double FROSchPreconditioner::ComputeTime() const {return timeCompute_;}

// Returns the time spent in ApplyInverse().
double FROSchPreconditioner::ApplyInverseTime() const {return timeApplyInverse_;}

//void FROSchPreconditioner::FixPressure(){
//
//    Teuchos::RCP<const Epetra_CrsMatrix> K_crsConst = Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrixFixedPressure_);
//
//    Teuchos::RCP<Epetra_CrsMatrix> K_crs = Teuchos::rcp_const_cast< Epetra_CrsMatrix>(K_crsConst);
//
//    std::cout << "K_crs->IndicesAreGlobal():" << K_crs->IndicesAreGlobal() << std::endl;
//
//    hymls_gidx maxGID = K_crs->Map().MaxAllGID();
//    int lid = K_crs->Map().LID(maxGID);
//    if ( lid > -1 ) {
//        double* values;
//        int* indices;
//        int numEntries;
//        K_crs->ExtractMyRowView(lid, numEntries, values, indices);
//        std::cout << "ExtractMyRowView:"<< std::endl;
//        for (int i=0; i<numEntries; i++){
//            values[i] = 0.;
//            std::cout << i << " " << maxGID << " ind:"<<indices[i] << " indGlob:"<< K_crs->GCID(indices[i])<< " v:" << values[i] << std::endl;
//        }
//
//        double diagValue=1.;
//        int diagIndex=K_crs->LCID(maxGID);
//        std::cout << "indices[0]:" << diagIndex<< std::endl;
//        std::cout << "insert:"<<K_crs->InsertMyValues( lid, 1, &diagValue, &diagIndex ) << std::endl;
//    }
//
//    if ( lid > -1 ) {
//        double* values;
//        int* indices;
//        int numEntries;
//        K_crs->ExtractMyRowView(lid, numEntries, values, indices);
//        for (int i=0; i<numEntries; i++) {
//            std::cout << i << " " << maxGID << " ind:"<<indices[i] << " indGlob:"<< K_crs->GCID(indices[i])<< " v:" << values[i] << std::endl;
//        }
//    }
//
//}

void FROSchPreconditioner::FillMatrixGlobal(){
    
    Teuchos::RCP<const Epetra_CrsMatrix> K_crsConst = Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_);
    
    Teuchos::RCP<Epetra_CrsMatrix> K_crs = Teuchos::rcp_const_cast< Epetra_CrsMatrix>(K_crsConst);
    
    hymls_gidx maxGID = K_crs->Map().MaxAllGID();
    int lidToMaxGID = K_crs->Map().LID(maxGID);
    
    double* values;
    int* indices;
    int numEntries;
    for (int i=0; i<K_crs->NumMyRows(); i++) {
        K_crs->ExtractMyRowView(i, numEntries, values, indices);
        std::vector<hymls_gidx> gids(numEntries);
        for (int j=0; j<numEntries; j++)
            gids[j] = K_crs->GCID(indices[j]);

        if ( !(i==lidToMaxGID) && numEntries>0 )
            matrixFixedPressure_->InsertGlobalValues( K_crs->GRID(i), numEntries, values, &gids[0] );

    }
    if (lidToMaxGID>0) {
        double value = 1.;
        matrixFixedPressure_->InsertGlobalValues( maxGID, 1, &value, &maxGID );
    }
    
    matrixFixedPressure_->FillComplete();
}

void FROSchPreconditioner::FillMatrixLocal(){
    
    Teuchos::RCP<const Epetra_CrsMatrix> K_crsConst = Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_);
    
    Teuchos::RCP<Epetra_CrsMatrix> K_crs = Teuchos::rcp_const_cast< Epetra_CrsMatrix>(K_crsConst);
    
    hymls_gidx maxGID = K_crs->Map().MaxAllGID();
    int lidToMaxGID = K_crs->Map().LID(maxGID);
    
    double* values;
    int* indices;
    int numEntries;
    hymls_gidx gid;
    for (int i=0; i<K_crs->NumMyRows(); i++) {
        K_crs->ExtractMyRowView(i, numEntries, values, indices);
        std::vector<int> lids(numEntries);
        for (int j=0; j<numEntries; j++){
            gid = K_crs->GCID(indices[j]);
            lids[j] = K_crs->LCID( gid );
        }
        
        if (!(i==lidToMaxGID) && numEntries>0)
            matrixFixedPressure_->InsertMyValues( K_crs->GRID(i), numEntries, values, &lids[0] );
        
        if (lidToMaxGID>0) {
            double value = 1.;
            matrixFixedPressure_->InsertMyValues( maxGID, 1, &value, &maxGID );
        }
    }
}

}//namespace TRIOS
