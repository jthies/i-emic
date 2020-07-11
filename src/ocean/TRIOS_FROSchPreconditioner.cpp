
#include "FROSchPreconditioner.hpp"
#include "TRIOS_Domain.H"
#include "HYMLS_Macros.hpp"
#include "HYMLS_Tools.hpp"
#include <EpetraExt_RowMatrixOut.h>
#include "FROSch_TwoLevelBlockPreconditioner_def.hpp"
#include "Xpetra_CrsMatrixWrap.hpp"
#include <unistd.h>
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
  PL().set("DofsPerNode1",dim_);
  PL().set("DofOrdering1", "NodeWise");
  PL().set("DofsPerNode2",1);
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
  
  if (dim_!=3|| (dof_!=4 && dof_!=5)) HYMLS::Tools::Error("currently only implemented for 3D case with dof=4 or dof=5",__FILE__,__LINE__);
  
  // create two maps: one with the velocities and the separators replicated across processes,
  // and one with only the local pressures
  Teuchos::RCP<Epetra_Map> u_map = Teuchos::null;
  Teuchos::RCP<Epetra_Map> v_map = Teuchos::null;
  Teuchos::RCP<Epetra_Map> w_map = Teuchos::null;
  Teuchos::RCP<Epetra_Map> p_map = Teuchos::null;
  Teuchos::RCP<Epetra_Map> velocity_map = Teuchos::null;

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
  
  int num_p_elts = (i1-i0+1)*(j1-j0+1)*(k1-k0+1);
  int num_global_p_elts = N*M*L;
  int num_global_u_elts = -1; // there's overlap of the separators, so let Epetra determine it itself.
  int num_global_v_elts = -1;
  int num_global_w_elts = -1;

  // pressures
  int *p_elts = new int[num_p_elts];

  int pos = 0;

  for (int k = k0; k <= k1; k++)
  {
    for (int j = j0; j <= j1; j++)
    {
      for (int i = i0; i <= i1; i++)
      {
      
      p_elts[pos++] = HYMLS::Tools::sub2ind(N, M, L, dof_, i, j, k, dof_-1);
      }
    }
  }

  // velocities: add separators on `left' side
  if (domain_->FirstI()<domain_->FirstRealI()) i0--;
  if (domain_->FirstJ()<domain_->FirstRealJ()) j0--;
  if (domain_->FirstK()<domain_->FirstRealK()) k0--;

  int num_v_elts = (i1-i0+1)*(j1-j0+1)*(k1-k0+1);

  int *u_elts = new int[num_v_elts];
  int *v_elts = new int[num_v_elts];
  int *w_elts = new int[num_v_elts];

  pos = 0;
  for (int k = k0; k <= k1; k++)
  {
    for (int j = j0; j <= j1; j++)
    {
      for (int i = i0; i <= i1; i++)
      {
        u_elts[pos] = HYMLS::Tools::sub2ind(N, M, L, dof_, i, j, k, 0);
        v_elts[pos] = HYMLS::Tools::sub2ind(N, M, L, dof_, i, j, k, 1);
        w_elts[pos] = HYMLS::Tools::sub2ind(N, M, L, dof_, i, j, k, 2);
        if (u_elts[pos]<0)
        {
          std::cout <<"PID " << comm_->MyPID()<< ", pos="<<pos << ", (i,j,k)="<< i<<","<<j<<","<<k<<")"<<", u_idx="<< u_elts[pos]<<std::endl;
        }
        if (v_elts[pos]<0)
        {
          std::cout <<"PID " << comm_->MyPID()<< ", pos="<<pos << ", (i,j,k)="<< i<<","<<j<<","<<k<<")"<<", v_idx="<< v_elts[pos]<<std::endl;
        }
        if (w_elts[pos]<0)
        {
          std::cout <<"PID " << comm_->MyPID()<< ", pos="<<pos << ", (i,j,k)="<< i<<","<<j<<","<<k<<")"<<", w_idx="<< w_elts[pos]<<std::endl;
        }
        pos++;
      }
    }
  }
  u_map = Teuchos::rcp(new Epetra_Map(-1, num_v_elts, u_elts, 0, *comm_));
  v_map = Teuchos::rcp(new Epetra_Map(-1, num_v_elts, v_elts, 0, *comm_));
  w_map = Teuchos::rcp(new Epetra_Map(-1, num_v_elts, w_elts, 0, *comm_));
  p_map = Teuchos::rcp(new Epetra_Map(num_global_p_elts, num_p_elts, p_elts, 0, *comm_));

  hymls_gidx *velocity_elts = new hymls_gidx[3*num_v_elts];
  for (int i=0; i<num_v_elts; i++) 
  {
    velocity_elts[3*i]=u_elts[i];
    velocity_elts[3*i+1]=v_elts[i];
    velocity_elts[3*i+2]=w_elts[i];
  }
  velocity_map = Teuchos::rcp(new Epetra_Map(-1, 3*num_v_elts, velocity_elts, 0, *comm_));

  delete [] u_elts;
  delete [] v_elts;
  delete [] w_elts;
  delete [] p_elts;
      
  delete [] velocity_elts;
      
  Teuchos::ArrayRCP<Teuchos::RCP<const XMap> > repeatedMaps(2);
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<const XMap> > > dofMaps(2);
  Teuchos::ArrayRCP<Teuchos::RCP<const XMap > > velocityMaps(3);

  Teuchos::ArrayRCP<Teuchos::RCP<const XMap > > pressureMaps(1);
  Teuchos::ArrayRCP<unsigned> dofsPerNodeVector(2);
  Teuchos::ArrayRCP<FROSch::DofOrdering> dofOrderings(2);
      
  dofOrderings[0] = FROSch::NodeWise;
  dofOrderings[1] = FROSch::NodeWise;
  dofsPerNodeVector[0] = 3;
  dofsPerNodeVector[1] = 1;
  const Epetra_MpiComm& tmpComm = dynamic_cast<const Epetra_MpiComm&> (*comm_);
  Teuchos::RCP<const Teuchos::Comm<int> > teuchosComm = Teuchos::rcp(new Teuchos::MpiComm<int> (tmpComm.Comm()));

  repeatedMaps[0] = FROSch::ConvertToXpetra<double,int,hymls_gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *velocity_map, teuchosComm );
  repeatedMaps[1] = FROSch::ConvertToXpetra<double,int,hymls_gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *p_map, teuchosComm );
  
  velocityMaps[0] = FROSch::ConvertToXpetra<double,int,hymls_gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *u_map, teuchosComm );
  velocityMaps[1] = FROSch::ConvertToXpetra<double,int,hymls_gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *v_map, teuchosComm );
  velocityMaps[2] = FROSch::ConvertToXpetra<double,int,hymls_gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *w_map, teuchosComm );
  pressureMaps[0] = FROSch::ConvertToXpetra<double,int,hymls_gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *p_map, teuchosComm ); 

  dofMaps[0] = velocityMaps;
  dofMaps[1] = pressureMaps;

  Teuchos::RCP<Teuchos::FancyOStream> fancy = fancyOStream(Teuchos::rcpFromRef(std::cout));

  //pressureMaps[0]->describe(*fancy,Teuchos::VERB_EXTREME);
    
  if (dof_==5)
  {
    Teuchos::RCP<Epetra_Map> t_map = Teuchos::null;
    int num_t_elts = num_v_elts;
  
//  Teuchos::RCP<const Xpetra::Map<int,hymls_gidx,node_type> > repeatedMap = FROSch::MergeMapsCont( repeatedMaps );
//  Teuchos::RCP<Xpetra::Map<int,hymls_gidx,node_type> > uniqueMap = FROSch::BuildUniqueMap( repeatedMap );
//
//
//  Teuchos::RCP<Xpetra::Matrix<double,int,hymls_gidx,node_type> > matrix_X_repartitioned =  Xpetra::MatrixFactory<double,int,hymls_gidx,node_type>::Build(uniqueMap,matrix_X_->getGlobalMaxNumRowEntries() );
//
// Teuchos::RCP<Xpetra::Import<int,hymls_gidx,node_type> > importer = Xpetra::ImportFactory<int,hymls_gidx,node_type>::Build(matrix_X_->getRowMap(),uniqueMap);
//
//      matrix_X_repartitioned->doImport(*matrix_X_,*importer,Xpetra::INSERT);
//      matrix_X_repartitioned->fillComplete();
//      Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
    
  Teuchos::ArrayRCP< Teuchos::RCP<const XMultiVector > > nodesDummy = Teuchos::null;
  Teuchos::ArrayRCP< Teuchos::RCP<const XMultiVector > > nullSpaceDummy = Teuchos::null;

    hymls_gidx *t_elts = new hymls_gidx[num_t_elts];
    pos = 0;
    for (int k = k0; k <= k1; k++)
    {
      for (int j = j0; j <= j1; j++)
      {
        for (int i = i0; i <= i1; i++)
        {
          t_elts[pos++] = HYMLS::Tools::sub2ind(N, M, L, dof_, i, j, k, dof_-2);
        }
      }
    }

    t_map = Teuchos::rcp(new Epetra_Map(-1, num_t_elts, t_elts, 0, *comm_));
    delete [] t_elts;

    // CH 24.01.20: For Boussinesq we actually have the ordering (ux,uy,uz,t,p) per finite volume.
    // Therefore, we should have the pressure map below.
    dofOrderings.resize( 3, FROSch::NodeWise );
    dofsPerNodeVector.resize( 3, 1 );
    Teuchos::ArrayRCP<Teuchos::RCP<const XMap > > tempMaps(1);
    tempMaps[0] = FROSch::ConvertToXpetra<double,int,hymls_gidx,node_type>::ConvertMap( Xpetra::UseEpetra, *t_map, teuchosComm );

    //tempMaps[0]->describe(*fancy,Teuchos::VERB_EXTREME);
        
    repeatedMaps.resize( 3, tempMaps[0] );
    dofMaps.resize( 3, tempMaps);
  }
  Teuchos::ArrayRCP< Teuchos::RCP<const XMultiVector > > nodesDummy = Teuchos::null;
  Teuchos::ArrayRCP< Teuchos::RCP<const XMultiVector > > nullSpaceDummy = Teuchos::null;

  prec_ = Teuchos::rcp(new FROSchPrecType(Teuchos::rcp_const_cast<XMatrix>(matrix_X_), Teuchos::rcpFromRef(PL())));
          
  int overlap=PL().get("Overlap",1);
          
  prec_->initialize(dim_,dofsPerNodeVector,dofOrderings,overlap,repeatedMaps,nullSpaceDummy,nodesDummy,dofMaps);
      
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

