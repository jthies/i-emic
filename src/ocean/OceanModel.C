/**********************************************************************
 * Copyright by Jonas Thies, Univ. of Groningen 2006/7/8.             *
 * Permission to use, copy, modify, redistribute is granted           *
 * as long as this header remains intact.                             *
 * contact: jonas@math.rug.nl                                         *
 **********************************************************************/
#include <iostream>
#include <iomanip>

/* Trilinos */

// Teuchos
#include "Teuchos_oblackholestream.hpp"

// Epetra
#include "Epetra_Util.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_SerialComm.h"
#include "Epetra_Vector.h"

// LOCA
#include "LOCA_Parameter_Vector.H"

/* my own packages */

#include "GlobalDefinitions.H"
#include "TRIOS_MatrixUtils.H"

// TRIOS
#include "TRIOS_Domain.H"
#include "TRIOS_BlockPreconditioner.H"

//trilinos_thcm
#include "GlobalDefinitions.H"
#include "THCM.H"
#include "OceanModel.H"
#include "OceanGrid.H"

#include "OceanOutputXdmf.H"

// thcm

extern "C" {
_SUBROUTINE_(get_grid_data)(int* nn,int* mm,int* ll,double* xx,double* yy,double* zz);
_SUBROUTINE_(write_data)(double*, int*, int*);  // file inout.F90
_SUBROUTINE_(append_data)(double*, int*, int*);  // file inout.F90
_SUBROUTINE_(writematrhs)(double*);  // file matetc.F90
_MODULE_SUBROUTINE_(m_global,compute_flux)(double*);
}//extern

using TRIOS::MatrixUtils;

OceanModelEvaluator::OceanModelEvaluator(ParameterList& plist):
  paramList(plist),pVector(null)
  {
  DEBUG("Create Ocean Model...");

  DEBVAR(paramList);

#ifdef STORE_MATRICES
  store_step_jac = 0;
  store_step_rhs = 0;
#endif

  string probdesc = paramList.get("Problem Description","Unnamed");
  this->SetLabel(("OceanModel ("+probdesc+")").c_str());

// continuation parameter (may be "Time" in transient mode)
// This is typically set by the main program
  cont_param = paramList.get("Parameter Name","Undefined");
  // sign s of exponent if you want to change e in param = 10^{s*e}
  cont_s   = paramList.get("Continuation in Exponent",1.0);
  // if you use "Exponent" as cont. param, the following parameter
  // is given the value 10^{s*e}:
  exp_cont_param = paramList.get("Exp. Cont. Parameter","Undefined");

  pVector = Teuchos::rcp(new LOCA::ParameterVector);
  this->CurrentParameters(*pVector);

  // update parameter names and initial values for ModelEvaluator
  Epetra_SerialComm scomm;
  int npar = _NPAR_+_NPAR_TRILI+1;//+1 because we also have time.
                                 // time is par(0) and some more are added in THCM.C
                                  // (_NPAR_TRILI)
  p_map = Teuchos::rcp(new Epetra_Map(npar,0,scomm));
  p_init = Teuchos::rcp(new Epetra_Vector(*p_map));
  p_names= Teuchos::rcp(new Teuchos::Array<string>(npar));
  double start_value;
  for (int i=0;i<pVector->length();i++)
    {
    (*p_names)[i] = pVector->getLabel(i);
    (*p_init)[i] = pVector->getValue(i);
    }
  try {
    start_value = pVector->getValue(cont_param);
    } catch(...) {
        ERROR("specified Parameter not found in ParameterVector!",__FILE__,__LINE__);
        }

  prec_reuse_policy = paramList.get("Preconditioner Reuse Policy", "None");
  // if this option is not set, leave it to LOCA:
  max_prec_age = paramList.get("Max Age Of Prec", -2);
  prec_age=0;

  backup_interval = paramList.get("Backup Interval",-1.0);
  last_backup = start_value-1e-12;// we subtract eps because of the way

                                  // backuping is treated in XYZT mode.
                                  // otherwise the initial solution would
                                  // be bckuped in that case.
  output_interval=paramList.get("Output Frequency",-1.0);
  last_output=last_backup-1.1*output_interval;
  ParameterList& xdmfParams = paramList.sublist("Xdmf Output");
  Teuchos::RCP<TRIOS::Domain> domain = THCM::Instance().GetDomain();
  bool print = (output_interval>0);
  xdmfWriter = Teuchos::rcp(new OceanOutputXdmf(domain,xdmfParams,print));
  gridPtr = xdmfWriter->getGrid();

  // pressure correction not implemented correctly
  pres_corr = "never";
  }

void OceanModelEvaluator::CurrentParameters(LOCA::ParameterVector& pvec) const
  {
  double val;
  std::string label;
  try {
    // get all _NPAR_ THCM params, time (0), exp (_NPAR_+1) and seas (_NPAR_+2)
    for (int i=0;i<=_NPAR_+_NPAR_TRILI;i++)
      {
      label = THCM::int2par(i);
      val=1.0; // default value for params not yet set.
               // note that all THCM params (1-30) are always set
      THCM::Instance().getParameter(label,val);
      if (pvec.isParameter(label))
        {
        pvec.setValue(label,val);
        }
      else
        {
        pvec.addParameter(label, val);
        }
      }
    } catch (...) {ERROR("failed to set parameters",__FILE__,__LINE__);}
  }

OceanModelEvaluator::~OceanModelEvaluator()
  {
  DEBUG("Destroy Ocean Model...");
  }

Teuchos::RCP<Epetra_Vector> OceanModelEvaluator::ReadConfiguration(std::string filename ,LOCA::ParameterVector& pVec)
  {

  Teuchos::RCP<std::istream> in;
  in = Teuchos::rcp(new std::ifstream(filename.c_str()) );
  std::string s1,s2,s3;
  (*in) >> s1;
  DEBVAR(s1);
  if (s1!="LOCA::ParameterVector")
    {
    ERROR("Error reading start config",__FILE__,__LINE__);
    }

  // read THCM Parameter vector
  int npar;
  (*in) >> s1 >> s2 >> npar >>s3;
  DEBVAR(npar)

  int j;
  string key;
  double value;
  for (int i=0;i<npar;i++)
    {
    read_parameter_entry(in,key,value);
    if (pVec.isParameter(key))
      {
      pVec.setValue(key, value);
      }
    else
      {
      pVec.addParameter(key,value);
      }
    }

  // read current solution
  Teuchos::RCP<Epetra_Vector> dsoln = THCM::Instance().getSolution();
  Teuchos::RCP<Epetra_Map> dmap = THCM::Instance().GetDomain()->GetSolveMap();

  Teuchos::RCP<Epetra_Vector> gsoln = MatrixUtils::Gather(*dsoln,0);

  if (THCM::Instance().GetComm()->MyPID()==0)
    {
    (*in) >> s1;
    if (s1!="Epetra::Vector")
      {
      INFO("Bad Vector label: should be Epetra::Vector, found "<<s1<<std::endl);
      ERROR("Error reading start config",__FILE__,__LINE__);
      }
    (*in) >> s1 >> s2 >> s3;
    if (s1+s2+s3!="MyPIDGIDValue")
      {
      ERROR("Error reading start config",__FILE__,__LINE__);
      }
    int pid,gid;
    double val;
    for (int i=0;i<gsoln->GlobalLength();i++)
      {
      (*in) >> pid >> gid >> val;
      (*gsoln)[gid]=val;
      }
    }

  dsoln = MatrixUtils::Scatter(*gsoln,*dmap);

  try {
    last_backup=pVec.getValue(cont_param);
    } catch (...) {
    ERROR("Missing continuation parameter in starting file!",__FILE__,__LINE__);
    }
  return dsoln;
  }

void OceanModelEvaluator::read_parameter_entry(RCP<std::istream> in, string& key, double& value)
  {
  int j;
  string tmp;
  (*in) >> j;
  (*in) >> key;
  while (1)
    {
    (*in) >> tmp;
    if (tmp=="=")
      {
      (*in) >> value;
      break;
      }
    else
      {
      key = key + " "+tmp;
      }
    }
  DEBVAR(j);
  DEBVAR(key);
  DEBVAR(value);
  }


void OceanModelEvaluator::WriteConfiguration(std::string filename , const LOCA::ParameterVector&
                   pVector, const Epetra_Vector& soln)
  {
  Teuchos::RCP<std::ostream> out;
  if (THCM::Instance().GetComm()->MyPID()==0)
    {
    out = Teuchos::rcp(new std::ofstream(filename.c_str()) );
    }
  else
    { // dummy stream
    out = Teuchos::rcp(new Teuchos::oblackholestream());
    }
  (*out) << std::setw(15) << std::setprecision(15);
  out->setf(ios::scientific);
  (*out) << pVector;
  (*out) << *(MatrixUtils::Gather(soln,0));
  }

// compute and store streamfunction in 'fort.7'
void OceanModelEvaluator::Monitor(double conParam)
  {
  // data in grid-object is assumed to be current solution

  // some constants
  double hdim, r0dim, udim;
  THCM::Instance().getModelConstants(r0dim,udim,hdim);
  double transc = r0dim*hdim*udim*1e-6;


  double psimmin = transc*gridPtr->psimMin();
  double psimmax = transc*gridPtr->psimMax();
  double psibmin = transc*gridPtr->psibMin();
  double psibmax = transc*gridPtr->psibMax();

  int itp = 0; // bifurcation point? Can't say up to now!
  int icp = THCM::Instance().par2int(cont_param); // continuation parameter
  double xl = conParam;

  if (THCM::Instance().GetComm()->MyPID()==0)
    {
    string filename="fort.7";
    // this is for the periodic orbit problem, where multiple instances
    // of THCM may be running on different parts of the global comm:
#ifdef HAVE_MPI
    Epetra_MpiComm globcomm(MPI_COMM_WORLD);
    int gpid = globcomm.MyPID();
    int gnp = globcomm.NumProc();
    if (gnp!= THCM::Instance().GetComm()->NumProc())
      {
      std::stringstream ss;
      ss << filename <<"."<<gpid;
      filename = ss.str();
      }
#endif
    //TODO: find a smart way of handling append or no append
    std::ofstream fort7(filename.c_str(),ios::app);

    fort7 << itp << " ";
    fort7 << icp << " ";
    fort7 << xl  << " ";
    fort7 << psimmin << " ";
    fort7 << psimmax << " ";
    fort7 << psibmin << " ";
    fort7 << psibmax << std::endl;
    fort7.close();
    }

  }

//////////// EpetraExt::ModelEvaluator interface //////////////////////

  // get the map of our variable vector [u,v,w,p,T,S]'
  Teuchos::RCP<const Epetra_Map> OceanModelEvaluator::get_x_map() const
    {
    return THCM::Instance().GetDomain()->GetSolveMap();
    }

///////////////////////////////////////////////////////////////////////

  // get the map of our 'model response' F(u)
  Teuchos::RCP<const Epetra_Map> OceanModelEvaluator::get_f_map() const
    {
    return THCM::Instance().GetDomain()->GetSolveMap();
    }

///////////////////////////////////////////////////////////////////////

  // get initial guess (all zeros)
  Teuchos::RCP<const Epetra_Vector> OceanModelEvaluator::get_x_init() const
    {
    return THCM::Instance().getSolution();
    }

///////////////////////////////////////////////////////////////////////

  // create the Jacobian
  Teuchos::RCP<Epetra_Operator> OceanModelEvaluator::create_W() const
    {
    return THCM::Instance().getJacobian();
    }

///////////////////////////////////////////////////////////////////////

  EpetraExt::ModelEvaluator::InArgs OceanModelEvaluator::createInArgs() const
    {
    EpetraExt::ModelEvaluator::InArgsSetup inArgs;
    inArgs.setModelEvalDescription("Ocean Model");
    inArgs.setSupports(IN_ARG_x,true);
    inArgs.setSupports(IN_ARG_x_dot,true);
    inArgs.setSupports(IN_ARG_alpha,true);
    inArgs.setSupports(IN_ARG_beta,true);
    inArgs.setSupports(IN_ARG_t,true);
    inArgs.set_Np(1); // note: there are actually _NPAR_+2 parameters,
                      // but we store them in a single Epetra_Vector
    return inArgs;
    }

///////////////////////////////////////////////////////////////////////

  EpetraExt::ModelEvaluator::OutArgs OceanModelEvaluator::createOutArgs() const
    {
    EpetraExt::ModelEvaluator::  OutArgsSetup outArgs;
    outArgs.setModelEvalDescription(this->description());
    outArgs.setSupports(OUT_ARG_f,true);
    outArgs.setSupports(OUT_ARG_W,true);
    // TODO: is this correc? I think I just copied it and left it there...
    outArgs.set_W_properties(
    DerivativeProperties(DERIV_LINEARITY_NONCONST
                        ,DERIV_RANK_FULL,true // supportsAdjoint
                    ));
    return outArgs;
    }

///////////////////////////////////////////////////////////////////////

void OceanModelEvaluator::evalModel( const InArgs& inArgs, const OutArgs& outArgs ) const
  {
  using Teuchos::dyn_cast;
  using Teuchos::rcp_dynamic_cast;

  TIMER_START("Compute F and/or Jacobian");
  DEBUG("enter OceanModelEvaluator::evalModel");
  //
  // Get the input arguments
  //
  const Epetra_Vector &x = *inArgs.get_x();
  const Epetra_Vector &xdot = *inArgs.get_x_dot();
  double alpha = inArgs.get_alpha();
  double beta = inArgs.get_beta();
  // there are two ways of setting the time t:
  // * using inArgs.set_t(),
  // * using parameter p0.
  // if inArgs.get_t() has the default value 0,
  // we get it from the parameter vector.
  double t = inArgs.get_t();
  DEBVAR(t)
//  INFO("Model evaluated at t="<<t);
  Teuchos::RCP<const Epetra_Vector> p_values = inArgs.get_p(0);

  if (t==0.0) t = (*p_values)[0];

  // note: p_values[0] is time: we get it from InArgs instead

  // if we do continuation in the exponent, we have to put a different
  // value into THCM, namely par(exp_cont_par)*10^{s*par(exp)}.
  double factor = 1.0;
  int index = -1; // no param should be multiplied by this factor
  if (cont_param=="Exponent")
    {
    index = THCM::Instance().par2int(exp_cont_param);
    // this has to be set by the user in thcm_params.xml as "Exp. Comp. Parameter"
    if (index<0) ERROR("Invalid Exp. Cont. Parameter",__FILE__,__LINE__);
    int exp_idx = THCM::Instance().par2int("Exponent");
    double cont_e = (*p_values)[exp_idx];
    factor = std::pow(10.0, cont_s*cont_e);
    }

  for (int i=1;i<p_values->MyLength();i++)
    {
    string label = (*p_names)[i];
    double value = (*p_values)[i];
    if (index==i) value*=factor;
    THCM::Instance().setParameter(label, value);
    }

  //
  // Get the output arguments
  //
  Teuchos::RCP<Epetra_Vector> f_out = outArgs.get_f();
  Teuchos::RCP<Epetra_Operator>     W_out = outArgs.get_W();
//  if (showGetInvalidArg_)
//    {
//    Epetra_Vector *g_out = outArgs.get_g(0).get();
//    }

  // note that setting "Time" in THCM will switch to monthly data instead of
  // the default (annual mean) as soon as t>0.
  THCM::Instance().setParameter("Time",t);

  bool want_A = ((W_out!=null)&&(beta!=0.0));
  // compute Jacobian and/or RHS
  bool result=THCM::Instance().evaluate(x, f_out, want_A);
  if (!result) ERROR("Error evaluating model!",__FILE__,__LINE__);

  //
  // Compute the functions
  //

  if (f_out!=null)
    {/*
    NOTE: B-matrix not implemented in this version.
    This does not matter for continuation of steady states (xdot=0),
    and tht's the only application for this class here.
    const Epetra_Vector& B = THCM::Instance().DiagB();
    // add diagonal matrix B times xdot
    for (int i=0;i<f_out->MyLength();i++)
      {
      (*f_out)[i] = B[i]*xdot[i] + (*f_out)[i];
      }
    */
    }
  if (W_out!=null)
    {
    // after the evaluate call above, THCM contains an Teuchos::RCP to the matrix A, which
    // we hereby extract:
    Teuchos::RCP<Epetra_CrsMatrix> W = THCM::Instance().getJacobian();

    DEBUG("construct THCM Matrix alpha*B + beta*A");
    DEBVAR(alpha);
    DEBVAR(beta);

    // scale to get beta*A.
    CHECK_ZERO(W->Scale(beta));

  // get diagonal of beta*A in a vector
  Teuchos::RCP<Epetra_Vector> diag = Teuchos::rcp(new Epetra_Vector(*get_x_map()));
  CHECK_ZERO(W->ExtractDiagonalCopy(*diag));

  // add -alpha*B to diagonal. Note that the sign is reversed as compared to the LOCA interface
  /* TODO: this doesn't work yet because THCM.DiagB is not available here.
  const Epetra_Vector& B = THCM::Instance().DiagB();
  CHECK_ZERO(diag->Update(alpha,B,1.0));
  */
  // replace diagonal in Jacobian:
  CHECK_ZERO(W->ReplaceDiagonalValues(*diag));

  // pass it on to the OutArgs:
  Teuchos::RCP<Epetra_CrsMatrix> W_out_crs =
    rcp_dynamic_cast<Epetra_CrsMatrix>(W_out,true);
    if (W_out_crs.get()!=W.get())
      {
      *W_out_crs = *W;
      }

#ifdef STORE_MATRICES
    std::stringstream ss;
    ss << "Jac_"<<store_step_jac<<".txt";
    MatrixUtils::DumpMatrix(*W_out_crs,ss.str());
    store_step_jac++;
#endif

  // compute new scaling
  // TODO: This function uses THCM's internal matrix, which now
  // does not contain beta*A+alpha*B, but only A. Is that a problem?
  // If not, it has already been called in the THCM::evaluate function.
  //THCM::Instance().RecomputeScaling();
  }
TIMER_STOP("Compute F and/or Jacobian");
DEBUG("leave OceanModelEvaluator::evalModel");
}

///////////////////////////////////////////////////////////////////////


// implementation of NOX::Abstract::PrePostOperator
// (functions that should be called before and after each nonlinear solve
// and nonlinear solver iteration, respectively)

// executed at the start of a call to iterate()
void OceanModelEvaluator::runPreIterate(const NOX::Solver::Generic& solver)
  {
  DEBUG("NOX pre-iteration function called");
  if (pres_corr=="pre-iter")
    {
    ERROR("Pressure Correction not implemented",__FILE__,__LINE__);
    }
  }

// executed at the end of a call to iterate()
void OceanModelEvaluator::runPostIterate(const NOX::Solver::Generic& solver)
  {
  DEBUG("NOX post-iteration function called");
  if (pres_corr=="post-iter")
    {
    ERROR("Pressure Correction not implemented",__FILE__,__LINE__);
    }
  }

// executed at the start of a call to solve()
void OceanModelEvaluator::runPreSolve(const NOX::Solver::Generic& solver)
  {
  DEBUG("NOX pre-solve function called");
  if (pres_corr=="pre-solve")
    {
    ERROR("Pressure Correction not implemented",__FILE__,__LINE__);
    }
  }

// executed at the end of a call to solve()
void OceanModelEvaluator::runPostSolve(const NOX::Solver::Generic& solver)
  {
  DEBUG("NOX post-solve function called");
  if (pres_corr=="post-solve")
    {
    ERROR("Pressure Correction not implemented",__FILE__,__LINE__);
    }
  }

///////////////////////////////////////////////////////////////////////

// enhanced interface: OceanModel (for NOX/LOCA)

OceanModel::OceanModel(ParameterList& plist, const Teuchos::RCP<LOCA::GlobalData>& globalData,
    Teuchos::RCP<ParameterList> lsParams)
      : OceanModelEvaluator(plist),
        LOCA::Epetra::ModelEvaluatorInterface(globalData,rcp(this,false)),
        backup_filename("IntermediateConfig.txt"), force_backup(false),
        thcm_output(false), thcm_label(2)
  {
  if (lsParams!=null)
    {
#ifdef DEBUGGING
    lsParams->sublist("Block Preconditioner").set("Verbosity",10);
#endif
    Teuchos::RCP<TRIOS::Domain> domainPtr = THCM::Instance().GetDomain();
    Teuchos::RCP<Epetra_CrsMatrix> jacPtr = THCM::Instance().getJacobian();

    string prec_type=lsParams->get("User Defined Preconditioner","Block Preconditioner");
    if (prec_type=="Block Preconditioner")
      {
      precPtr = Teuchos::rcp(new TRIOS::BlockPreconditioner(jacPtr,domainPtr,lsParams->sublist("Block Preconditioner")));
      }
    else
      {
      ERROR("unkown 'User Defined Preconditioner': '"+prec_type+"'.",__FILE__,__LINE__);
      }

    }
  }

///////////////////////////////////////////////////////////////////////


// compute preconditioner, which can then be retrieved by getPreconditioner()
bool OceanModel::computePreconditioner(const Epetra_Vector& x,
                                         Epetra_Operator& Prec,
                                         ParameterList* p)
  {
  DEBUG("enter OceanModel::computePreconditioner");
  if (precPtr == null)
    {
    // no preconditioner parameters passed to constructor
    ERROR("No Preconditioner available!",__FILE__,__LINE__);
    }
  bool result=precPtr->Initialize();
  if (result==0)
    {
    result=precPtr->Compute();
    }
  DEBUG("leave OceanModel::computePreconditioner");
  return result;
  }

///////////////////////////////////////////////////////////////////////

// for XYZT output
void OceanModel::dataForPrintSolution(const int conStep, const int timeStep,
                                  const int totalTimeSteps)
  {
  // figure out 'what time it is'
  double T = paramList.get("Time Period",0.0);
  double dt = T/totalTimeSteps;
  double t = timeStep*dt;
  THCM::Instance().setParameter("Time",t);
  std::stringstream ss;
  ss << "Config"<<timeStep<<".txt";
  backup_filename = ss.str();
  thcm_label = 2+timeStep;
  INFO("Data for printSolution:");
  INFO(conStep<<" "<<timeStep<<" "<<totalTimeSteps);
  INFO(" backup file \""<<backup_filename<<"\"");
  }

  Teuchos::RCP<NOX::Epetra::Scaling> OceanModelEvaluator::getScaling() const
    {
    Teuchos::RCP<NOX::Epetra::Scaling> scaling = Teuchos::rcp(new NOX::Epetra::Scaling());
    Teuchos::RCP<Epetra_Vector> row_scaling = THCM::Instance().getRowScaling();
    Teuchos::RCP<Epetra_Vector> col_scaling = THCM::Instance().getColScaling();
    scaling->addUserScaling(NOX::Epetra::Scaling::Right,col_scaling);
    scaling->addUserScaling(NOX::Epetra::Scaling::Left,row_scaling);
    return scaling;
    }



  //! return the preconditioner operator. Will only be non-null
  //! if you passed a preclist to the constructor. Before using
  //! the preconditioner, computePreconditioner() should be called.
  Teuchos::RCP<Epetra_Operator> OceanModel::getPreconditioner()
    {
    return Teuchos::rcp_dynamic_cast<Epetra_Operator>(precPtr);
    }


////////////////
// Call user's own print routine for vector-parameter pair
void OceanModel::printSolution(const Epetra_Vector& x,
                   double conParam)
  {
  INFO("#################################");
  INFO("Backup Interval: "<<backup_interval);
  INFO("Last Backup: "<<last_backup);
  INFO("Current value: "<<conParam);
  INFO("Force Backup: "<<force_backup);
  INFO("THCM Output: "<<thcm_output);
  INFO("THCM Label: "<<thcm_label);

  bool xmf_out = false; // false: only import solution to grid

  if (output_interval>=0)
    {
    if (conParam-last_output>output_interval)
      {
      TIMER_START("Store Solution (XDMF)");
      INFO("Writing Xdmf File...");
      last_output = conParam;
      xmf_out = true; //true: store Xdmf file (if available)
      }
    }

    // either just import solution t grid or
    // also write HDF5/XML files
    xdmfWriter->Store(x,conParam,xmf_out);
    if (xmf_out) TIMER_STOP("Store Solution (XDMF)");

  if ((backup_interval>=0)||force_backup)
    {

    if ((conParam-last_backup>backup_interval)||force_backup||(conParam==last_backup))
      {
      //three cases where we write a backup:
      // - some time has passed since last backup (backup_interval)
      // - the user forces us to (i.e. at the end of a continuation, force_backup)
      // - last_backup indicates that this function is being called repeatedly,
      // - which probably means that we're in LOCA XYZT mode
      TIMER_START("Store Solution (Backup)");
      INFO("Writing Backup at param value "<<conParam<<"...");
      WriteConfiguration(backup_filename,*pVector,x);
      last_backup = conParam;
      TIMER_STOP("Store Solution (Backup)");

     if (thcm_output)
       {
       // write solution in native THCM format
       int filename = 3; // write to 'fort.3'
       int label = thcm_label;

       // gather the solution on the root process (pid 0):
       Teuchos::RCP<Epetra_Vector> fullSol = MatrixUtils::Gather(x,0);

       if (THCM::Instance().GetComm()->MyPID() == 0 )
         {
         int ndim = fullSol->GlobalLength();
         double *solutionbig;

         fullSol->ExtractView(&solutionbig);
         (*outFile) << " Store solution in THCM format..."<<std::endl;
         if (label==2)
           {
           INFO("writing data, label = "<<label);
           FNAME(write_data)(solutionbig, &filename, &label);
           }
         else
           {
           INFO("appending data, label = "<<label);
           FNAME(append_data)(solutionbig, &filename, &label);
           }
         //fort.15...
         F90NAME(m_global,compute_flux)(solutionbig);
         (*outFile) << "done!"<<std::endl;
         }
       }

     }
   }

  // in every step, we compute the meridional and barotropic streamfunctions
  // and store their maximum in fort.7 (in the old THCM format)
  // At this point, 'grid' contains the complete solution in 3D array format
  this->Monitor(conParam);

  // invalidate preconditioner after Time-/Continuation step
  if (prec_reuse_policy=="Step")
    {
    prec_age++;
    if (prec_age>=max_prec_age)
      {
      //! This is a bit fiddly and for now we just do not allow any preconditioner reuse.
      ERROR("Recomputing preconditioner not implemented here.",__FILE__,__LINE__);
      prec_age=0;
      }
    }
  }


///////////////////////////////////////////////////////////////////////
