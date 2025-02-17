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
#include "Teuchos_StrUtils.hpp"

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
#include "OceanModelIO.H"
#include "OceanGrid.H"

// thcm

extern "C" {
_SUBROUTINE_(get_grid_data)(int* nn,int* mm,int* ll,double* xx,double* yy,double* zz);
_SUBROUTINE_(write_data)(double*, int*, int*);  // file inout.F90
_SUBROUTINE_(append_data)(double*, int*, int*);  // file inout.F90
_SUBROUTINE_(writematrhs)(double*);  // file matetc.F90
_MODULE_SUBROUTINE_(m_global,compute_flux)(double*);
}//extern

using TRIOS::MatrixUtils;

OceanModelEvaluator::OceanModelEvaluator(Teuchos::ParameterList& plist):
  paramList(plist),pVector(null)
  {
  DEBUG("Create Ocean Model...");

  DEBVAR(paramList);

#ifdef STORE_MATRICES
  store_step_jac = 0;
  store_step_rhs = 0;
#endif

  std::string probdesc = paramList.get("Problem Description","Unnamed");
  this->SetLabel(("OceanModel ("+probdesc+")").c_str());

// continuation parameter (may be "Time" in transient mode)
// This is typically set by the main program
  cont_param = paramList.get("Parameter Name","Undefined");
  // scale continuation parameter received from LOCA by cont_s before
  // passing it to THCM. This is useful because LOCA doesn't straight-
  // forwardly allow continuation e.g. from 0 to -1. cont_s is also used
  // to scale the exponent in "Exponent" continuation runs.
  cont_s   = paramList.get("Continuation Parameter Scaling",1.0);
  // if you use "Exponent" as cont. param, the following parameter
  // is given the value 10^{s*e}:
  actual_cont_param = paramList.get("Actual Continuation Parameter",cont_param);

  if (cont_param == "Undefined")
  {
    ERROR("You have to set the parameter 'THCM'->'Parameter Name' to the value of the LOCA 'Continuation Parameter'",__FILE__,__LINE__);
  }
  if (actual_cont_param == "Undefined")
  {
    actual_cont_param = cont_param;
  }
  if (actual_cont_param == "Exponent" || actual_cont_param=="Backward")
  {
    ERROR("You have to set the parameter 'THCM'->'Actual Continuation Parameter' (string) if you want to do continuation in 'Exponent' or 'Backward'.",__FILE__,__LINE__);
  }

  pVector = Teuchos::rcp(new LOCA::ParameterVector);
  this->CurrentParameters(*pVector);

  // update parameter names and initial values for ModelEvaluator
  Epetra_SerialComm scomm;
  int npar = _NPAR_+_NPAR_TRILI+1;//+1 because we also have time.
                                 // time is par(0) and some more are added in THCM.C
                                  // (_NPAR_TRILI)
  p_map = Teuchos::rcp(new Epetra_Map(npar,0,scomm));
  p_init = Teuchos::rcp(new Epetra_Vector(*p_map));
  p_names= Teuchos::rcp(new Teuchos::Array<std::string>(npar));
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

  backup_interval = paramList.get("Backup Interval",20);
  backup_counter  = paramList.get("Backup Index", 0);
  step_counter = 0;
  last_backup = 0;

                                  // backuping is treated in XYZT mode.
                                  // otherwise the initial solution would
                                  // be bckuped in that case.
  Teuchos::RCP<TRIOS::Domain> domain = THCM::Instance().GetDomain();
  gridPtr = Teuchos::rcp(new OceanGrid(domain));

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
      val=0.0; // default value for params not yet set.
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
  int num_dots = std::count(filename.begin(), filename.end(), '.');
  if (num_dots!=1)
  {
    ERROR("Filename '"+filename+"' should contain exactly one '.'",__FILE__,__LINE__);
  }
  Teuchos::RCP<Epetra_Vector> dsoln = THCM::Instance().getSolution();
  std::string file_extension = Teuchos::StrUtils::after(filename, ".");
  if (file_extension=="h5")
  {
    bool loadSalFlux = (THCM::Instance().getSRES()==0);
    bool loadTemFlux = (THCM::Instance().getTRES()==0);
    bool loadMask = false;
    CHECK_ZERO(OceanModelIO::loadStateFromFile(filename, *dsoln, *pVector, loadTemFlux, loadSalFlux, loadMask));
    // make sure initial state satisfies integral condition.
    // If we compute ("diagnose") the salinity flux from a run with SRES=1 (restoring/Dirichlet conitions)
    // and then switch to non-restoring, the Jacobian is singular. To fix this, one of the equations is replaced
    // by an integral conition "normalizing" the salt content of the water. This function computes the constant
    // prescribed by the solution read from the file.
    if (loadSalFlux)
    {
      THCM::Instance().setIntCondCorrection(dsoln);
    }
  }
  else if (file_extension=="txt")
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
    std::string key;
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
  }
  try
  {
    // note: we may use positive or otherwise scaled values in LOCA,
    // via "Continuation Parameter Scaling". By default cont_s=1.
    double pval=cont_s*pVec.getValue(cont_param);
    pVec.setValue(cont_param, pval);
    last_backup=pval-1.0e-12;
  } catch (...) {
    ERROR("Missing continuation parameter in starting file!",__FILE__,__LINE__);
    }
  return dsoln;
  }

void OceanModelEvaluator::read_parameter_entry(RCP<std::istream> in, std::string& key, double& value)
  {
  int j;
  std::string tmp;
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
  out->setf(std::ios::scientific);
  (*out) << pVector;
  (*out) << *(MatrixUtils::Gather(soln,0));
  }

// compute and store streamfunction in 'fort.7'
void OceanModelEvaluator::Monitor(double conParam)
{
  // note: We do not have access to the current solution here,
  //       however, LOCA always calls 'printSolution' beforehand,
  //       so in that function we import the given state into the
  //       gridPtr (OceanGrid) object and use it below.

  // some constants
  double hdim, r0dim, udim;
  THCM::Instance().getModelConstants(r0dim,udim,hdim);
  double transc = r0dim*hdim*udim*1e-6;

  // get maximum and minimum of meridional overturning streamfunction (PsiM) below 1km
  // note: the input parameters are the depth range over which to compute the
  // min and max (ranging from 0 to 1), but they are ignored right now because
  // otherwise we would get inconsistent fort.7 files in a running study.
  double psimmin = transc*gridPtr->psimMin(1000/hdim,1.0);
  double psimmax = transc*gridPtr->psimMax(1000/hdim,1.0);
  // min and max of barotropic streamfunction
  double psibmin = transc*gridPtr->psibMin();
  double psibmax = transc*gridPtr->psibMax();

  int itp = 0; // bifurcation point? Can't say up to now!
  int icp = THCM::Instance().par2int(cont_param); // continuation parameter
  double xl = cont_s*conParam;

  if (icp > _NPAR_)
  {
    // some parameters like "Exponent" and "Backward" are mapping functions
    // for actual (physical) parameters, so use the actual parameter for THCM output instead:
    icp = THCM::Instance().par2int(actual_cont_param);
    THCM::Instance().getParameter(actual_cont_param, xl);
  }
  if (icp<0)
  {
    WARNING("Continuation parameter index for THCM could not be determined correctly, fort.7 may contain errors.",__FILE__,__LINE__);
  }

  if (THCM::Instance().GetComm()->MyPID()==0)
  {
    std::string filename="fort.7";
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
    std::ofstream fort7(filename.c_str(),std::ios::app);

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

   //////////////////////////////////////
  // Set/update THCM model parameters //
 //////////////////////////////////////

  // We may pass a modified value to THCM in these cases:
  // * cont_s*p if cont_s!=1 (parameter "Continuation Parameter Scaling" in "THCM" sublist)
  // * par(actual_cont_param)*10^{cont_s*par("Exponent") if we do continuation in "Exponent",
  // * (1-par("Backward"))*par(actual_cont_param) for continuation in "Backward".
  double factor = cont_s;
  int index = THCM::Instance().par2int(actual_cont_param);
  if (index<0) ERROR("Invalid Actual Continuation Parameter",__FILE__,__LINE__);

  if (cont_param=="Exponent")
  {
    int exp_idx = THCM::Instance().par2int("Exponent");
    double cont_e = (*p_values)[exp_idx];
    factor = std::pow(10.0, cont_s*cont_e);
  }
  else if (cont_param=="Backward")
  {
    int bw_idx = THCM::Instance().par2int("Backward");
    double cont_bw = (*p_values)[bw_idx];
    factor = 1.0 - cont_bw;
  }

  for (int i=1;i<p_values->MyLength();i++)
  {
    std::string label = (*p_names)[i];
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

OceanModel::OceanModel(Teuchos::ParameterList& plist, const Teuchos::RCP<LOCA::GlobalData>& globalData,
    Teuchos::RCP<Teuchos::ParameterList> lsParams)
      : OceanModelEvaluator(plist),
        LOCA::Epetra::ModelEvaluatorInterface(globalData,rcp(this,false)),
        backup_filename("State_"), force_backup(false)
  {
  if (lsParams!=null)
    {
#ifdef DEBUGGING
    lsParams->sublist("Block Preconditioner").set("Verbosity",10);
#endif
    Teuchos::RCP<TRIOS::Domain> domainPtr = THCM::Instance().GetDomain();
    Teuchos::RCP<Epetra_CrsMatrix> jacPtr = THCM::Instance().getJacobian();

    std::string prec_type=lsParams->get("User Defined Preconditioner","Block Preconditioner");
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
                                         Teuchos::ParameterList* p)
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
  INFO("Current step: "<<step_counter);
  INFO("Current value: "<<conParam);
  INFO("Force Backup: "<<force_backup);
  INFO("THCM Label: "<<thcm_label);

  // make sure we have the current parameter values in pVector
  this->CurrentParameters(*pVector);

  // update solution in "OceanGrid" object, we use this
  // object to compute integrals for the stream functions.
  gridPtr->ImportData(x);

  if ((backup_interval>=0)||force_backup)
  {
    if (((step_counter-last_backup)>backup_interval)||force_backup)
    {
      //two cases where we write a backup:
      // - some time has passed since last backup (backup_interval)
      // - the user forces us to (i.e. at the end of a continuation, force_backup)
      TIMER_START("Store Solution (Backup)");
      INFO("Writing Backup at param value "<<conParam<<"...");
      std::stringstream fs;
      fs << backup_filename << std::setw(4) << std::setfill('0') << backup_counter << "_par" << THCM::Instance().par2int(cont_param)
                            << "_"    << cont_s*conParam;
      backup_counter++;
#ifdef HAVE_HDF5
      auto filename = fs.str() + ".h5";
      CHECK_ZERO(OceanModelIO::saveStateToFile(filename, x, *pVector));
#else
      auto filename = fs.str() + ".txt";
      WriteConfiguration(filename,*pVector,x);
#endif
      last_backup = step_counter;
      TIMER_STOP("Store Solution (Backup)");
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
  step_counter++;
}


///////////////////////////////////////////////////////////////////////
