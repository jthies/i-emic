
#include "TRIOS_FROSchPreconditioner.hpp"
#include "TRIOS_Domain.H"
#include "TRIOS_Macros.H"
#include "Utils.H"

#include "HYMLS_Macros.hpp"
#include "HYMLS_Tools.hpp"

#include <EpetraExt_RowMatrixOut.h>
#include "FROSch_TwoLevelPreconditioner_def.hpp"
#include "FROSch_TwoLevelBlockPreconditioner_def.hpp"
#include "Xpetra_CrsMatrixWrap.hpp"
#include "Xpetra_MultiVectorFactory.hpp"
#include <unistd.h>

using namespace ocean_defs;

namespace TRIOS {

//! JT: my own ocean-adaptation of the FROSch-function BuildNullSpace() (in frosch/src/Tools/FROSch_Tools.hpp)
Teuchos::ArrayRCP<Teuchos::RCP<const FROSchPreconditioner::XMultiVector> > 
FROSchPreconditioner::BuildNullSpaces(Teuchos::ArrayRCP<Teuchos::RCP<const XMap> > const repeatedMaps, 
        Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<const XMap> > > const dofMaps,
        Teuchos::ArrayRCP<const unsigned> const dofsPerNodeVector) const
  {
  int numBlocks=repeatedMaps.size();
  // separate multi-vectors for uv, w, p, TS
  if (numBlocks!=4 || dofsPerNodeVector.size()!=numBlocks || dofMaps.size()!=numBlocks)
    {
    HYMLS::Tools::Error("input arrays are expected to represent four blocks: uv,w,p,TS.",
                __FILE__,__LINE__);
    }
  Teuchos::ArrayRCP<Teuchos::RCP<const XMultiVector> > nullSpaces(4);

  // u,v
  nullSpaces[0]=FROSch::BuildNullSpace<double, int, gidx, node_type>
  (dim_, FROSch::LaplaceNullSpace, repeatedMaps[0], 2, dofMaps[0]);
  // w
  // note: for the z-velocity (w) we have only a balance between the vertical pressure gradient
  // and the buoyancy forces. Since we use a 2D domain decomposition, I think we can skip w for
  // the coarse operator altogether, but if I set the nullspace component to null, I get a seg-
  // fault:
//  nullSpaces[1]=Teuchos::null;
  nullSpaces[1]=FROSch::BuildNullSpace<double, int, gidx, node_type>
        (dim_, FROSch::LaplaceNullSpace, repeatedMaps[1], 1, dofMaps[1]);
  // T,S
  nullSpaces[3]=FROSch::BuildNullSpace<double, int, gidx, node_type>
  (dim_, FROSch::LaplaceNullSpace, repeatedMaps[3], 2, dofMaps[3]);

  // p: two checkerboard vectors, we have to construct those ourselves.
  // see also TRIOS::BlockPreconditioner::build_svp
  Teuchos::RCP<XMultiVector> p_null = Xpetra::MultiVectorFactory<double,int,gidx,node_type>::Build(repeatedMaps[2],2);
  p_null->putScalar(0.0);

  int lrow = 0;
  for (int k=domain_->FirstRealK(); k<=domain_->LastRealK(); k++)
    {
    for (int j=domain_->FirstRealJ(); j<=domain_->LastRealJ(); j++)
      {
      for (int i=domain_->FirstRealI(); i<=domain_->LastRealI(); i++)
        {
        p_null->getDataNonConst((i+j)%2)[lrow] = 1.0;
        lrow++;
        }
      }
    }
  nullSpaces[2] =p_null;
  return nullSpaces;
  }

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
  time_=Teuchos::rcp(new Epetra_Time(K->Comm()));

  Teuchos::RCP<const Epetra_CrsMatrix> K_crs = Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_);

//      matrixFixedPressure_ = Teuchos::rcp(new Epetra_CrsMatrix( Epetra_DataAccess::Copy, K_crs->RowMap(), K->MaxNumEntries() ) );
  
//  FillMatrixGlobal();

//      K_crs->Print(std::cout);
  //TODO: don't understand how to create an Xpetra wrapper for a const object
  Teuchos::RCP<Epetra_CrsMatrix> K_crs_nonconst = Teuchos::rcp_const_cast<Epetra_CrsMatrix>(K_crs);
  time_->ResetStartTime();
  // this is a wrapper to turn the object into an Xpetra object
  Teuchos::RCP<Xpetra::CrsMatrix<double, int, gidx,node_type> > K_x =
        Teuchos::rcp(new Xpetra::EpetraCrsMatrixT<gidx,node_type>(K_crs_nonconst));
  // this is an Xpetra::Matrix that allows 'viewing' the matrix like a block matrix, for instance
  matrix_X_ = Teuchos::rcp(new Xpetra::CrsMatrixWrap<double,int,gidx,node_type>(K_x));
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

  dim_ = 3;
  dof_ = 6;
// note: we set these options via the input arguments to initialize
  PL("FROSch").set("Dimension",dim_);
  // we have to construct a specific null space for our
  // ocean preconditioner
  PL("FROSch").set("Null Space Type", "Input");
  PL("FROSch").set("CoarseOperator Type", "IPOUHarmonicCoarseOperator");

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
  // this is mostly because we would have to include overlap between the first and last
  // subdomains in i-direction, and the domain_->CreateMap function doesn't allow that.
  // The AssemblyMap, on the other hand, has too much overlap for the purpose of this  
  // solver class.
  if (domain_->IsPeriodic()) HYMLS::Tools::Error("not yet implemented for periodic B.C.",__FILE__,__LINE__);

  // global map without overlap between partitions
  Teuchos::RCP<const Epetra_Map> standardMap = domain_->GetStandardMap();
  if (!standardMap->SameAs(*rangeMap_)) HYMLS::Tools::Error("unexpeted (repartitioned?) map found",__FILE__,__LINE__);

  int i0=domain_->FirstRealI(), i1=domain_->LastRealI();
  int j0=domain_->FirstRealJ(), j1=domain_->LastRealJ();
  int k0=domain_->FirstRealK(), k1=domain_->LastRealK();

  int I0=0, I1=domain_->GlobalN()-1;
  int J0=0, J1=domain_->GlobalM()-1;
  int K0=0, K1=domain_->GlobalL()-1;

  // a map with all dofs per cell and a single layer of overlap, from which we can extract components as before
  // we define for each direction i0 as the first index incl. the separator, and i1 as the last.
  //
  // In 1D: (o is the pressure, | the cell boundary with the velcoity)
  //          i0      i1
  // | o | o || o | o |
  // i0      i1

  // velocities: add separators on `left' side
  if (domain_->FirstI()<domain_->FirstRealI()) i0--;
  if (domain_->FirstJ()<domain_->FirstRealJ()) j0--;
  if (domain_->FirstK()<domain_->FirstRealK()) k0--;

  Teuchos::RCP<const Epetra_Map> repeatedMap = Utils::CreateMap
        (i0,i1,j0,j1,k0,k1,
         I0,I1,J0,J1,K0,K1,
         dof_, *comm_);

  // we start by creating some maps for the separate variables and groups of variables.
  Teuchos::RCP<Epetra_Map> u_map, v_map, w_map, p_map, t_map, s_map, uv_map, ts_map;
  
  u_map = Utils::CreateSubMap(*repeatedMap,dof_,UU);
  v_map = Utils::CreateSubMap(*repeatedMap,dof_,VV);
  w_map = Utils::CreateSubMap(*standardMap,dof_,WW);
  p_map = Utils::CreateSubMap(*standardMap,dof_,PP);
  t_map = Utils::CreateSubMap(*repeatedMap,dof_,TT);
  s_map = Utils::CreateSubMap(*repeatedMap,dof_,SS);

  const int UV[2]={UU,VV};
  uv_map = Utils::CreateSubMap(*repeatedMap,dof_,UV);

  const int TS[2]={TT,SS};
  ts_map = Utils::CreateSubMap(*repeatedMap,dof_,TS);
        
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
  Teuchos::ArrayRCP<Teuchos::RCP<const XMap > > uvMaps(2);
  Teuchos::ArrayRCP<Teuchos::RCP<const XMap > > wMaps(1);
  Teuchos::ArrayRCP<Teuchos::RCP<const XMap > > pressureMaps(1);
  Teuchos::ArrayRCP<Teuchos::RCP<const XMap > > tracerMaps(2);

  Teuchos::ArrayRCP<Teuchos::RCP<const XMap> > repeatedMaps(4);
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<const XMap> > > dofMaps(4);

  const Epetra_MpiComm& tmpComm = dynamic_cast<const Epetra_MpiComm&> (*comm_);
  Teuchos::RCP<const Teuchos::Comm<int> > teuchosComm = Teuchos::rcp(new Teuchos::MpiComm<int> (tmpComm.Comm()));

  repeatedMaps[0] = FROSch::ConvertToXpetra<double,int,gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *uv_map, teuchosComm );
  repeatedMaps[1] = FROSch::ConvertToXpetra<double,int,gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *w_map,  teuchosComm );
  repeatedMaps[2] = FROSch::ConvertToXpetra<double,int,gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *p_map,  teuchosComm );
  repeatedMaps[3] = FROSch::ConvertToXpetra<double,int,gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *ts_map, teuchosComm );
  
  uvMaps[0] = FROSch::ConvertToXpetra<double,int,gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *u_map, teuchosComm );
  uvMaps[1] = FROSch::ConvertToXpetra<double,int,gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *v_map, teuchosComm );
  wMaps[0] = FROSch::ConvertToXpetra<double,int,gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *w_map, teuchosComm );
  pressureMaps[0] = FROSch::ConvertToXpetra<double,int,gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *p_map, teuchosComm ); 
  tracerMaps[0] = FROSch::ConvertToXpetra<double,int,gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *t_map, teuchosComm ); 
  tracerMaps[1] = FROSch::ConvertToXpetra<double,int,gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *s_map, teuchosComm ); 

  dofMaps[0] = uvMaps;
  dofMaps[1] = wMaps;
  dofMaps[2] = pressureMaps;
  dofMaps[3] = tracerMaps;

  //Teuchos::RCP<Teuchos::FancyOStream> fancy = fancyOStream(Teuchos::rcpFromRef(std::cout));
  //pressureMaps[0]->describe(*fancy,Teuchos::VERB_EXTREME);

  // construct the null spaces used for determining the coarse operator in FROSch.
  Teuchos::ArrayRCP< Teuchos::RCP<const XMultiVector> > nullSpace=BuildNullSpaces
        (repeatedMaps, dofMaps, dofsPerNodeVector);

  // I don't know what the nodes arraay does, I think it's for passing in coordinates,
  // which we don't need.
  Teuchos::ArrayRCP< Teuchos::RCP<const XMultiVector > > nodesDummy = Teuchos::null;

  // create the preconditioner
  prec_ = Teuchos::rcp(new FROSchPrecType(Teuchos::rcp_const_cast<XMatrix>(matrix_X_), 
        Teuchos::rcpFromRef(PL("FROSch"))));
   
  // I think this is irrelevant because we pass in  the 'repeatedMaps' that define how much overlap there is
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
//      std::string outName = "hymlsA_compute" + std::to_string(numCompute_) + ".dat";
//      const char *cstr = outName.c_str();
//      EpetraExt::RowMatrixToMatlabFile(cstr,*matrix_);
//      std::cout << "Exported compute!"<< std::endl;

//      FillMatrixLocal();
      
//    std::string outName = "hymlsA_compute" + std::to_string(numCompute_) + ".dat";
//    const char *cstr = outName.c_str();
//    EpetraExt::RowMatrixToMatlabFile(cstr,*matrix_);
      
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
        Xpetra::EpetraMultiVectorT<gidx, node_type>(Teuchos::rcpFromRef(const_cast<Epetra_MultiVector&>(B))));
  Teuchos::RCP<XMultiVector> _X = Teuchos::rcp(new
        Xpetra::EpetraMultiVectorT<gidx, node_type>(Teuchos::rcpFromRef(X)));
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
//    gidx maxGID = K_crs->Map().MaxAllGID();
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
    
    gidx maxGID = K_crs->Map().MaxAllGID();
    int lidToMaxGID = K_crs->Map().LID(maxGID);
    
    double* values;
    int* indices;
    int numEntries;
    for (int i=0; i<K_crs->NumMyRows(); i++) {
        K_crs->ExtractMyRowView(i, numEntries, values, indices);
        std::vector<gidx> gids(numEntries);
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
    
    gidx maxGID = K_crs->Map().MaxAllGID();
    int lidToMaxGID = K_crs->Map().LID(maxGID);
    
    double* values;
    int* indices;
    int numEntries;
    gidx gid;
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
