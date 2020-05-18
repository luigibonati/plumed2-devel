/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2020 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "bias/Bias.h"
#include "core/PlumedMain.h"
#include "core/ActionRegister.h"
#include "core/Atoms.h"
#include "tools/Communicator.h"
#include "tools/File.h"

namespace PLMD {
namespace opes {

//+PLUMEDOC BIAS OPES_EXPLORE
/*
On-the-fly probability enhanced sampling (OPES) with a well-tempered target distribution.

\par Examples

OPES_EXPLORE ...
  LABEL=opes
  ARG=cv
  PACE=500
  SIGMA=0.2
  BARRIER=15
... OPES_EXPLORE


*/
//+ENDPLUMEDOC

class OPESexplore : public bias::Bias {

private:
  bool isFirstStep_;
  bool afterCalculate_;
  unsigned NumParallel_;
  unsigned rank_;
  unsigned NumWalkers_;
  unsigned walker_rank_;
  unsigned ncv_;
  unsigned counter_;

  double kbt_;
  double biasfactor_;
  unsigned stride_;
  std::vector<double> sigma0_;
  unsigned adaptive_sigma_stride_;
  long unsigned adaptive_counter_;
  std::vector<double> av_cv_;
  std::vector<double> av_M2_;
  bool fixed_sigma_;
  double epsilon_;
  double sum_weights_;
  double sum_weights2_;
  double current_bias_;

  bool no_Zed_;
  double Zed_;

  double threshold2_;
  bool recursive_merge_;
//kernels for now are diagonal truncated Gaussians
  struct kernel
  {
    double height;
    std::vector<double> center;
    std::vector<double> sigma;

    inline void merge_me_with(const kernel & );
    kernel(double h, const std::vector<double> & c,const std::vector<double> & s):
      height(h),center(c),sigma(s) {}
  };
  double cutoff2_;
  double val_at_cutoff_;
  inline double evaluateKernel(const kernel&,const std::vector<double>&) const;
  inline double evaluateKernel(const kernel&,const std::vector<double>&,std::vector<double>&);
  std::vector<kernel> kernels_;
  OFile kernelsOfile_;

  double work_;
  double old_counter_;
  double old_Zed_;
  std::vector<kernel> delta_kernels_;

  OFile probOfile_;
  int wProbStride_;
  bool storeOldProb_;

public:
  OPESexplore(const ActionOptions&);
  void calculate() override;
  void update() override;
  double getProbAndDerivatives(const std::vector<double>&,std::vector<double>&);
  void addKernel(const kernel&,const bool);
  void addKernel(const double,const std::vector<double>&,const std::vector<double>&,const bool);
  unsigned getMergeableKernel(const std::vector<double>&,const unsigned);
  void dumpProbToFile();
  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(OPESexplore,"OPES_EXPLORE")

void OPESexplore::registerKeywords(Keywords& keys) {
  Bias::registerKeywords(keys);
  keys.use("ARG");
  keys.add("compulsory","TEMP","-1","temperature. If not specified tries to get it from MD engine");
  keys.add("compulsory","PACE","the frequency for kernel addition");
  keys.add("compulsory","SIGMA","0","the initial widths of the kernels. If not set, adaptive sigma will be used");
  keys.add("compulsory","BARRIER","the free energy barrier to be overcome. It is used to set BIASFACTOR, EPSILON, and KERNEL_CUTOFF to reasonable values");
  keys.add("compulsory","COMPRESSION_THRESHOLD","1","merge kernels if closer than this threshold. Set to zero to avoid compression");
//extra options
  keys.add("optional","BIASFACTOR","the \\f$\\gamma\\f$ bias factor used for well-tempered target \\f$p(\\mathbf{s})\\f$."
           " Set to 'inf' for non-tempered flat target");
  keys.add("optional","EPSILON","the value of the regularization constant for the probability");
  keys.add("optional","KERNEL_CUTOFF","truncate kernels at this distance (in units of sigma)");
  keys.add("optional","ADAPTIVE_SIGMA_STRIDE","stride for measuring adaptive sigma. Default is one PACE");
  keys.addFlag("NO_ZED",false,"do not normalize over the explored CV space, \\f$Z_n=1\\f$");
  keys.addFlag("FIXED_SIGMA",false,"do not decrease sigma as simulation goes on");
  keys.addFlag("RECURSIVE_MERGE_OFF",false,"do not recursively attempt kernel merging when a new one is added. Faster, but total number of compressed kernels might grow and slow down");
//kernels file
  keys.add("compulsory","FILE","KERNELS","a file in which the list of added kernels is stored");
  keys.add("optional","FMT","specify format for KERNELS file");
//save probability estimate (compressed kernels)
  keys.add("optional","PROB_WFILE","the file on which to write the estimated probability");
  keys.add("optional","PROB_WSTRIDE","write the estimated probability to a file every N steps");
  keys.addFlag("STORE_PROB",false,"store all the estimated probability files the calculation generates. They will be deleted if this keyword is not present");
//miscellaneous
  keys.addFlag("WALKERS_MPI",false,"Switch on MPI version of multiple walkers");
  keys.addFlag("SERIAL",false,"perform calculations in serial. Might be faster for small number of kernels e.g. 1D systems");
  keys.use("RESTART");

//output components
  componentsAreNotOptional(keys);
  keys.addOutputComponent("work","default","work done by the last kernel added");
  keys.addOutputComponent("rct","default","estimate of \\f$c(t)\\f$: \\f$\\frac{1}{\\beta}\\log \\lange e^{\\beta V} \\rangle\\f$");
  keys.addOutputComponent("zed","default","estimate of \\f$Z_n=\\int_\\Omega_n \\Tilde{P}_n(\\mathbf{s})\\, d\\mathbf{s}\\f$");
  keys.addOutputComponent("neff","default","effective sample size");
  keys.addOutputComponent("nker","default","total number of compressed kernels employed");
}

OPESexplore::OPESexplore(const ActionOptions&ao)
  : PLUMED_BIAS_INIT(ao)
  , isFirstStep_(true)
  , afterCalculate_(false)
  , counter_(0)
  , sum_weights_(0)
  , sum_weights2_(0)
  , Zed_(1)
  , work_(0)
  , old_counter_(0)
  , old_Zed_(1)
{
  ncv_=getNumberOfArguments();
//set kbt_
  const double Kb=plumed.getAtoms().getKBoltzmann();
  double temp=-1;
  parse("TEMP",temp);
  kbt_=Kb*temp;
  if(kbt_<0)
  {
    kbt_=plumed.getAtoms().getKbT();
    plumed_massert(kbt_>0,"your MD engine does not pass the temperature to plumed, you must specify it using TEMP");
  }

//other compulsory input
  parse("PACE",stride_);

  double barrier=0;
  parse("BARRIER",barrier);
  plumed_massert(barrier>=0,"the BARRIER should be greater than zero");

  biasfactor_=barrier/kbt_;
  parse("BIASFACTOR",biasfactor_);
  plumed_massert(biasfactor_>1,"BIASFACTOR must be greater than one");

  adaptive_sigma_stride_=0;
  parse("ADAPTIVE_SIGMA_STRIDE",adaptive_sigma_stride_);
  parseVector("SIGMA",sigma0_);
  if(sigma0_[0]==0 && sigma0_.size()==1)
  {
    sigma0_.clear();
    adaptive_counter_=0;
    if(adaptive_sigma_stride_==0)
      adaptive_sigma_stride_=stride_;
    av_cv_.resize(ncv_,0);
    av_M2_.resize(ncv_,0);
    plumed_massert(adaptive_sigma_stride_>=stride_,"better to chose ADAPTIVE_SIGMA_STRIDE >= PACE");
  }
  else
  {
    plumed_massert(sigma0_.size()==ncv_,"number of SIGMA parameters does not match number of arguments");
    plumed_massert(adaptive_sigma_stride_==0,"if SIGMA is set then it cannot be adaptive, thus ADAPTIVE_SIGMA_STRIDE should not be set");
    for(unsigned i=0; i<ncv_; i++)
      sigma0_[i]*=std::sqrt(biasfactor_); //the sigma of the target is broader F_t(s)=1/gamma*F(s)
  }

  epsilon_=std::exp(-barrier/(biasfactor_-1)/kbt_);
  parse("EPSILON",epsilon_);
  plumed_massert(epsilon_>0,"you must choose a value for EPSILON greater than zero. Is your BARRIER too high?");

  double cutoff=sqrt(2.*barrier/kbt_);
  parse("KERNEL_CUTOFF",cutoff);
  plumed_massert(cutoff>0,"you must choose a value for KERNEL_CUTOFF greater than zero");
  cutoff2_=cutoff*cutoff;
  val_at_cutoff_=std::exp(-0.5*cutoff2_);

  threshold2_=1;
  parse("COMPRESSION_THRESHOLD",threshold2_);
  threshold2_*=threshold2_;
  if(threshold2_!=0)
    plumed_massert(threshold2_>0 && threshold2_<cutoff2_,"COMPRESSION_THRESHOLD cannot be bigger than the KERNEL_CUTOFF");

//optional stuff
  no_Zed_=false;
  parseFlag("NO_ZED",no_Zed_);
  if(no_Zed_)
  {//this makes it more gentle in the initial phase
    counter_=1;
    sum_weights_=1;
    sum_weights2_=1;
  }
  fixed_sigma_=false;
  parseFlag("FIXED_SIGMA",fixed_sigma_);
  bool recursive_merge_off=false;
  parseFlag("RECURSIVE_MERGE_OFF",recursive_merge_off);
  recursive_merge_=!recursive_merge_off;

//kernels file
  std::string kernelsFileName("KERNELS");
  parse("FILE",kernelsFileName);
  std::string fmt;
  parse("FMT",fmt);

//output current probability estimate, as kernels
  std::string probFileName;
  parse("PROB_WFILE",probFileName);
  wProbStride_=0;
  parse("PROB_WSTRIDE",wProbStride_);
  storeOldProb_=false;
  parseFlag("STORE_PROB",storeOldProb_);
  if(wProbStride_!=0 || storeOldProb_)
    plumed_massert(probFileName.length()>0,"filename for estimated probability not specified, use PROB_WFILE");
  if(probFileName.length()>0 && wProbStride_==0)
    wProbStride_=-1;//will print only on CPT events

//multiple walkers //TODO implement also external mw for cp2k
  bool walkers_mpi=false;
  parseFlag("WALKERS_MPI",walkers_mpi);
  if(walkers_mpi)
  {
    if(comm.Get_rank()==0)//multi_sim_comm works on first rank only
    {
      NumWalkers_=multi_sim_comm.Get_size();
      walker_rank_=multi_sim_comm.Get_rank();
    }
    comm.Bcast(NumWalkers_,0); //if each walker has more than one processor update them all
    comm.Bcast(walker_rank_,0);
  }
  else
  {
    NumWalkers_=1;
    walker_rank_=0;
  }

//parallelization stuff
  NumParallel_=comm.Get_size();
  rank_=comm.Get_rank();
  bool serial=false;
  parseFlag("SERIAL",serial);
  if(serial)
  {
    log.printf(" -- SERIAL: running without loop parallelization\n");
    NumParallel_=1;
    rank_=0;
  }

  checkRead();

//restart if needed
  if(getRestart()) //TODO add option to restart from dumped PROB file
  {
    IFile ifile;
    ifile.link(*this);
    if(ifile.FileExist(kernelsFileName))
    {
      ifile.open(kernelsFileName);
      log.printf("  RESTART - make sure all used options are compatible\n");
      if(sigma0_.size()==0)
        log.printf(" +++ WARNING +++ restarting from an adaptive sigma simulation is not perfect\n");
      log.printf("    Restarting from: %s\n",kernelsFileName.c_str());
      double old_biasfactor;
      ifile.scanField("biasfactor",old_biasfactor);
      plumed_massert(std::abs(biasfactor_-old_biasfactor)>1e-3,"restarting form different bias factor is not supported");
      if(old_biasfactor!=biasfactor_)
        log.printf(" +++ WARNING +++ previous bias factor was %g while now it is %g. diff = %g\n",old_biasfactor,biasfactor_,biasfactor_-old_biasfactor);
      double old_epsilon;
      ifile.scanField("epsilon",old_epsilon);
      if(old_epsilon!=epsilon_)
        log.printf(" +++ WARNING +++ previous epsilon was %g while now it is %g. diff = %g\n",old_epsilon,epsilon_,epsilon_-old_epsilon);
      double old_cutoff;
      ifile.scanField("kernel_cutoff",old_cutoff);
      if(old_cutoff!=cutoff)
        log.printf(" +++ WARNING +++ previous kernel_cutoff was %g while now it is %g. diff = %g\n",old_cutoff,cutoff,cutoff-old_cutoff);
      double old_threshold;
      const double threshold=sqrt(threshold2_);
      ifile.scanField("compression_threshold",old_threshold);
      if(old_threshold!=threshold)
        log.printf(" +++ WARNING +++ previous compression_threshold was %g while now it is %g. diff = %g\n",old_threshold,threshold,threshold-old_threshold);
      for(unsigned i=0; i<ncv_; i++)
      {
        if(getPntrToArgument(i)->isPeriodic())
        {
          std::string arg_min,arg_max;
          getPntrToArgument(i)->getDomain(arg_min,arg_max);
          std::string file_min,file_max;
          ifile.scanField("min_"+getPntrToArgument(i)->getName(),file_min);
          ifile.scanField("max_"+getPntrToArgument(i)->getName(),file_max);
          plumed_massert(file_min==arg_min,"mismatch between restart and ARG periodicity");
          plumed_massert(file_max==arg_max,"mismatch between restart and ARG periodicity");
        }
      }
      ifile.allowIgnoredFields(); //this allows for multiple restart, but without checking for consistency between them!
      double time;
      while(ifile.scanField("time",time))
      {
        std::vector<double> center(ncv_);
        std::vector<double> sigma(ncv_);
        double height;
        double logweight;
        for(unsigned i=0; i<ncv_; i++)
          ifile.scanField(getPntrToArgument(i)->getName(),center[i]);
        for(unsigned i=0; i<ncv_; i++)
          ifile.scanField("sigma_"+getPntrToArgument(i)->getName(),sigma[i]);
        ifile.scanField("height",height);
        ifile.scanField("logweight",logweight);
        ifile.scanField();
        addKernel(height,center,sigma,false);
        const double weight=std::exp(logweight);
        sum_weights_+=weight; //FIXME this sum is slightly inaccurate, because when printing some precision is lost
        sum_weights2_+=weight*weight;
        counter_++;
      }
      if(!no_Zed_)
      {
        double sum_uprob=0;
        for(unsigned k=rank_; k<kernels_.size(); k+=NumParallel_)
          for(unsigned kk=0; kk<kernels_.size(); kk++)
            sum_uprob+=evaluateKernel(kernels_[kk],kernels_[k].center);
        if(NumParallel_>1)
          comm.Sum(sum_uprob);
        Zed_=sum_uprob/counter_/kernels_.size();
        old_Zed_=Zed_;
      }
      log.printf("    A total of %d kernels where read, and compressed to %d\n",counter_,kernels_.size());
      ifile.reset(false);
      ifile.close();
    }
    else
      log.printf(" +++ WARNING +++ restart requested, but file '%s' was not found!\n",kernelsFileName.c_str());
  }
//sync all walkers to avoid opening files before reding is over (see also METAD)
  comm.Barrier();
  if(comm.Get_rank()==0 && walkers_mpi)
    multi_sim_comm.Barrier();

//setup output kernels file
  kernelsOfile_.link(*this);
  if(NumWalkers_>1)
  {
    if(walker_rank_>0)
      kernelsFileName="/dev/null"; //only first walker writes on file
    kernelsOfile_.enforceSuffix("");
  }
  kernelsOfile_.open(kernelsFileName);
  if(fmt.length()>0)
    kernelsOfile_.fmtField(" "+fmt);
  kernelsOfile_.setHeavyFlush(); //do I need it?
  //define and set const fields
  kernelsOfile_.addConstantField("biasfactor");
  kernelsOfile_.addConstantField("epsilon");
  kernelsOfile_.addConstantField("kernel_cutoff");
  kernelsOfile_.addConstantField("compression_threshold");
  for(unsigned i=0; i<ncv_; i++)
    kernelsOfile_.setupPrintValue(getPntrToArgument(i));
  kernelsOfile_.printField("biasfactor",biasfactor_);
  kernelsOfile_.printField("epsilon",epsilon_);
  kernelsOfile_.printField("kernel_cutoff",sqrt(cutoff2_));
  kernelsOfile_.printField("compression_threshold",sqrt(threshold2_));

//open file for storing estimated probability
  if(wProbStride_!=0)
  {
    probOfile_.link(*this);
    if(NumWalkers_>1)
    {
      if(walker_rank_>0)
        probFileName="/dev/null"; //only first walker writes on file
      probOfile_.enforceSuffix("");
    }
    probOfile_.open(probFileName);
    if(fmt.length()>0)
      probOfile_.fmtField(" "+fmt);
  }

//add and set output components
  addComponent("work"); componentIsNotPeriodic("work");
  addComponent("rct"); componentIsNotPeriodic("rct");
  getPntrToComponent("rct")->set(kbt_*std::log(sum_weights_/counter_));
  addComponent("zed"); componentIsNotPeriodic("zed");
  getPntrToComponent("zed")->set(Zed_);
  addComponent("neff"); componentIsNotPeriodic("neff");
  getPntrToComponent("neff")->set(std::pow(1+sum_weights_,2)/(1+sum_weights2_));
  addComponent("nker"); componentIsNotPeriodic("nker");
  getPntrToComponent("nker")->set(kernels_.size());

//printing some info
  log.printf("  temperature T = %g\n",kbt_/Kb);
  log.printf("  beta = %g\n",1./kbt_);
  log.printf("  depositing new kernels with PACE = %d\n",stride_);
  log.printf("  expected BARRIER is %g\n",barrier);
  log.printf("  using target distribution with BIASFACTOR gamma = %g\n",biasfactor_);
  if(sigma0_.size()==0)
    log.printf("  adaptive SIGMA will be used, with ADAPTIVE_SIGMA_STRIDE = %d\n",adaptive_sigma_stride_);
  else
  {
    log.printf("  kernels have initial SIGMA = ");
    for(unsigned i=0; i<ncv_; i++)
      log.printf(" %g",sigma0_[i]);
    log.printf("\n");
  }
  if(fixed_sigma_)
    log.printf(" -- FIXED_SIGMA: sigma will not decrease as simulation steps increases\n");
  log.printf("  kernels are truncated with KERNELS_CUTOFF = %g\n",cutoff);
  if(cutoff<3.5)
    log.printf(" +++ WARNING +++ probably kernels are truncated too much\n");
  log.printf("  the value at cutoff is = %g\n",val_at_cutoff_);
  log.printf("  regularization EPSILON = %g\n",epsilon_);
  if(val_at_cutoff_>epsilon_)
    log.printf(" +++ WARNING +++ the KERNEL_CUTOFF might be too small for the given EPSILON");
  log.printf("  kernels will be compressed when closer than COMPRESSION_THRESHOLD = %g\n",sqrt(threshold2_));
  if(threshold2_==0)
    log.printf(" +++ WARNING +++ kernels will never merge, expect slowdowns\n");
  if(!recursive_merge_)
    log.printf(" -- RECURSIVE_MERGE_OFF: only one merge for each new kernel will be attempted. This is faster only if total number of kernels does not grow too much\n");
  if(no_Zed_)
    log.printf(" -- NO_ZED: using fixed normalization factor = %g\n",Zed_);
  if(wProbStride_!=0 && walker_rank_==0)
    log.printf("  probability estimate is written on file %s with stride %d\n",probFileName.c_str(),wProbStride_);
  if(walkers_mpi)
    log.printf(" -- WALKERS_MPI: if present, multiple replicas will communicate\n");
  if(NumWalkers_>1)
  {
    log.printf("  using multiple walkers\n");
    log.printf("    number of walkers: %d\n",NumWalkers_);
    log.printf("    walker rank: %d\n",walker_rank_);
  }
  int mw_warning=0;
  if(!walkers_mpi && comm.Get_rank()==0 && multi_sim_comm.Get_size()>(int)NumWalkers_)
    mw_warning=1;
  comm.Bcast(mw_warning,0);
  if(mw_warning) //log.printf messes up with comm, so never use it without Bcast!
    log.printf(" +++ WARNING +++ multiple replicas will NOT communicate unless the flag WALKERS_MPI is used\n");
  if(NumParallel_>1)
    log.printf("  using multiple threads per simulation: %d\n",NumParallel_);
  log.printf(" Bibliography ");
  log<<plumed.cite("M. Invernizzi and M. Parrinello, J. Phys. Chem. Lett. 11, 2731-2736 (2020)");
  log.printf("\n");
}

void OPESexplore::calculate()
{
  std::vector<double> cv(ncv_);
  for(unsigned i=0; i<ncv_; i++)
    cv[i]=getArgument(i);

  std::vector<double> der_prob(ncv_,0);
  const double prob=getProbAndDerivatives(cv,der_prob);
  current_bias_=kbt_*(biasfactor_-1)*std::log(prob/Zed_+epsilon_);
  setBias(current_bias_);
  for(unsigned i=0; i<ncv_; i++)
    setOutputForce(i,der_prob[i]==0?0:-kbt_*(biasfactor_-1)/(prob/Zed_+epsilon_)*der_prob[i]/Zed_);

//calculate work
  double tot_delta=0;
  for(unsigned d=0; d<delta_kernels_.size(); d++)
    tot_delta+=evaluateKernel(delta_kernels_[d],cv);
  const double old_prob=(prob*counter_-tot_delta)/old_counter_;
  work_+=current_bias_-kbt_*(biasfactor_-1)*std::log(old_prob/old_Zed_+epsilon_);

  afterCalculate_=true;
}

void OPESexplore::update()
{
//dump prob if requested
  if( (wProbStride_>0 && getStep()%wProbStride_==0) || (wProbStride_==-1 && getCPT()) )
    dumpProbToFile();

//update variance if adaptive sigma
  if(sigma0_.size()==0)
  {
    adaptive_counter_++;
    unsigned tau=adaptive_sigma_stride_;
    if(adaptive_counter_<adaptive_sigma_stride_)
      tau=adaptive_counter_;
    for(unsigned i=0; i<ncv_; i++)
    { //Welford's online algorithm for standard deviation
      const double cv_i=getArgument(i);
      const double diff=difference(i,av_cv_[i],cv_i);
      av_cv_[i]+=diff/tau; //exponentially decaying average
      av_M2_[i]+=diff*difference(i,av_cv_[i],cv_i);
    }
  }

//other updates
  if(getStep()%stride_!=0)
    return;
  if(isFirstStep_)//same in MetaD, useful for restarts?
  {
    isFirstStep_=false;
    return;
  }
  plumed_massert(afterCalculate_,"OPESexplore::update() must be called after OPESexplore::calculate() to work properly");
  afterCalculate_=false;

//work done by the bias in one iteration, uses as zero reference a point at inf, so that the work is always positive
  const double min_shift=kbt_*(biasfactor_-1)*std::log(old_Zed_/Zed_*old_counter_/counter_);
  getPntrToComponent("work")->set(work_-stride_*min_shift);
  work_=0;
  delta_kernels_.clear();
  old_counter_=counter_;
  old_Zed_=Zed_;
  unsigned old_nker=kernels_.size();

//get new kernel height
  double height=std::exp(current_bias_/kbt_); //this assumes that calculate() always runs before update()

//update sum_weights_ and neff
  double sum_heights=height;
  double sum_heights2=height*height;
  if(NumWalkers_>1)
  {
    if(comm.Get_rank()==0)
    {
      multi_sim_comm.Sum(sum_heights);
      multi_sim_comm.Sum(sum_heights2);
    }
    comm.Bcast(sum_heights,0);
    comm.Bcast(sum_heights2,0);
  }
  counter_+=NumWalkers_;
  sum_weights_+=sum_heights;
  sum_weights2_+=sum_heights2;
  const double neff=std::pow(1+sum_weights_,2)/(1+sum_weights2_);
  getPntrToComponent("rct")->set(kbt_*std::log(sum_weights_/counter_));
  getPntrToComponent("neff")->set(neff);
//in opes explore the kernel height=1, because it is not multiplied by the weight
  height=1;

//if needed, rescale sigma and height
  std::vector<double> sigma=sigma0_;
  if(sigma0_.size()==0)
  {
    sigma.resize(ncv_);
    for(unsigned i=0; i<ncv_; i++)
      sigma[i]=std::sqrt(av_M2_[i]/adaptive_counter_);
  }
  if(!fixed_sigma_)
  {
   //in opes explore rescaling is based on counter_, not on neff
    const double s_rescaling=std::pow(counter_*(ncv_+2.)/4.,-1./(4+ncv_));
    for(unsigned i=0; i<ncv_; i++)
      sigma[i]*=s_rescaling;
  //the height should be divided by sqrt(2*pi)*sigma,
  //but this overall factor would be canceled when dividing by Zed
  //thus we skip it altogether, but keep the s_rescaling
    height/=std::pow(s_rescaling,ncv_);
  }

//get new kernel center
  std::vector<double> center(ncv_);
  for(unsigned i=0; i<ncv_; i++)
    center[i]=getArgument(i);

//add new kernel(s)
  if(NumWalkers_>1)
  {
    std::vector<double> all_height(NumWalkers_,0.0);
    std::vector<double> all_center(NumWalkers_*ncv_,0.0);
    std::vector<double> all_sigma(NumWalkers_*ncv_,0.0);
    if(comm.Get_rank()==0)
    {
      multi_sim_comm.Allgather(height,all_height); //TODO heights should be communicated only once
      multi_sim_comm.Allgather(center,all_center);
      multi_sim_comm.Allgather(sigma,all_sigma);
    }
    comm.Bcast(all_height,0);
    comm.Bcast(all_center,0);
    comm.Bcast(all_sigma,0);
    for(unsigned w=0; w<NumWalkers_; w++)
    {
      std::vector<double> center_w(all_center.begin()+ncv_*w,all_center.begin()+ncv_*(w+1));
      std::vector<double> sigma_w(all_sigma.begin()+ncv_*w,all_sigma.begin()+ncv_*(w+1));
      addKernel(all_height[w],center_w,sigma_w,true);
    }
  }
  else
    addKernel(height,center,sigma,true);
  getPntrToComponent("nker")->set(kernels_.size());

  //update Zed_
  if(!no_Zed_)
  {
    double sum_uprob=0;
    const unsigned ks=kernels_.size();
    const unsigned ds=delta_kernels_.size();
    const bool few_kernels=(ks*ks<(3*ks*ds+2*ds*ds*NumParallel_+100)); //this seems reasonable, but is not rigorous...
    if(few_kernels) //really needed? Probably is almost always false
    {
      for(unsigned k=rank_; k<kernels_.size(); k+=NumParallel_)
        for(unsigned kk=0; kk<kernels_.size(); kk++)
          sum_uprob+=evaluateKernel(kernels_[kk],kernels_[k].center);
      if(NumParallel_>1)
        comm.Sum(sum_uprob);
    }
    else
    {
    // Here instead of redoing the full summation, we add only the changes, knowing that
    // uprob = old_uprob + delta_uprob
    // and we also need to consider that in the new sum there are some novel centers and some disappeared ones
      double delta_sum_uprob=0;
      for(unsigned k=rank_; k<kernels_.size(); k+=NumParallel_)
      {
        for(unsigned d=0; d<delta_kernels_.size(); d++)
        {
          const double sign=delta_kernels_[d].height<0?-1:1; //take away contribution from kernels that are gone, and add the one from new ones
          delta_sum_uprob+=evaluateKernel(delta_kernels_[d],kernels_[k].center)+sign*evaluateKernel(kernels_[k],delta_kernels_[d].center);
        }
      }
      if(NumParallel_>1)
        comm.Sum(delta_sum_uprob);
      for(unsigned d=0; d<delta_kernels_.size(); d++)
      {
        for(unsigned dd=0; dd<delta_kernels_.size(); dd++)
        { //now subtract the delta_uprob added before, but not needed
          const double sign=delta_kernels_[d].height<0?-1:1;
          delta_sum_uprob-=sign*evaluateKernel(delta_kernels_[dd],delta_kernels_[d].center);
        }
      }
      sum_uprob=Zed_*old_counter_*old_nker+delta_sum_uprob;
    }
    Zed_=sum_uprob/counter_/kernels_.size();
    getPntrToComponent("zed")->set(Zed_);
  }
}

double OPESexplore::getProbAndDerivatives(const std::vector<double> &cv,std::vector<double> &der_prob)
{
  if(kernels_.size()==0) //needed to avoid division by zero, counter_=0
    return 0;

  double prob=0.0;
  for(unsigned k=rank_; k<kernels_.size(); k+=NumParallel_) //TODO add neighbor list
    prob+=evaluateKernel(kernels_[k],cv,der_prob);
  if(NumParallel_>1)
  {
    comm.Sum(prob);
    comm.Sum(der_prob);
  }
  //normalize the estimate
  prob/=counter_;
  for(unsigned i=0; i<ncv_; i++)
    der_prob[i]/=counter_;

  return prob;
}

void OPESexplore::addKernel(const kernel &new_kernel,const bool write_to_file)
{
  addKernel(new_kernel.height,new_kernel.center,new_kernel.sigma,write_to_file);
}

void OPESexplore::addKernel(const double height,const std::vector<double>& center,const std::vector<double>& sigma,const bool write_to_file)
{
  bool no_match=true;
  if(threshold2_!=0)
  {
    unsigned taker_k=getMergeableKernel(center,kernels_.size());
    if(taker_k<kernels_.size())
    {
      no_match=false;
      delta_kernels_.emplace_back(-1*kernels_[taker_k].height,kernels_[taker_k].center,kernels_[taker_k].sigma);
      kernels_[taker_k].merge_me_with(kernel(height,center,sigma));
      delta_kernels_.push_back(kernels_[taker_k]);
      if(recursive_merge_) //the overhead is worth it if it keeps low the total number of kernels
      {
      //TODO this second check could run only through the kernels closer than, say, 2*threshold
      //     the function getMergeableKernel could return a list of such neighbors
        unsigned giver_k=taker_k;
        taker_k=getMergeableKernel(kernels_[giver_k].center,giver_k);
        while(taker_k<kernels_.size())
        {
          delta_kernels_.pop_back();
          delta_kernels_.emplace_back(-1*kernels_[taker_k].height,kernels_[taker_k].center,kernels_[taker_k].sigma);
          if(taker_k>giver_k) //saves time when erasing
            std::swap(taker_k,giver_k);
          kernels_[taker_k].merge_me_with(kernels_[giver_k]);
          delta_kernels_.push_back(kernels_[taker_k]);
          kernels_.erase(kernels_.begin()+giver_k);
          giver_k=taker_k;
          taker_k=getMergeableKernel(kernels_[giver_k].center,giver_k);
        }
      }
    }
  }
  if(no_match)
  {
    kernels_.emplace_back(height,center,sigma);
    delta_kernels_.emplace_back(height,center,sigma);
  }

//write to file
  if(write_to_file)
  {
    kernelsOfile_.printField("time",getTime());
    for(unsigned i=0; i<ncv_; i++)
      kernelsOfile_.printField(getPntrToArgument(i),center[i]);
    for(unsigned i=0; i<ncv_; i++)
      kernelsOfile_.printField("sigma_"+getPntrToArgument(i)->getName(),sigma[i]);
    kernelsOfile_.printField("height",height);
    kernelsOfile_.printField("logweight",current_bias_/kbt_);
    kernelsOfile_.printField();
  }
}

unsigned OPESexplore::getMergeableKernel(const std::vector<double> &giver_center,const unsigned giver_k)
{ //returns kernels_.size() if no match is found
  unsigned min_k=kernels_.size();
  double min_dist2=threshold2_;
  for(unsigned k=rank_; k<kernels_.size(); k+=NumParallel_) //TODO add neighbor list
  {
    if(k==giver_k) //a kernel should not be merged with itself
      continue;
    double dist2=0;
    for(unsigned i=0; i<ncv_; i++)
    { //TODO implement merging through the border for periodic CVs
      const double d=(kernels_[k].center[i]-giver_center[i])/kernels_[k].sigma[i];
      dist2+=d*d;
      if(dist2>=min_dist2)
        break;
    }
    if(dist2<min_dist2)
    {
      min_dist2=dist2;
      min_k=k;
    }
  }
  if(NumParallel_>1)
  {
    std::vector<double> all_min_dist2(NumParallel_);
    std::vector<unsigned> all_min_k(NumParallel_);
    comm.Allgather(min_dist2,all_min_dist2);
    comm.Allgather(min_k,all_min_k);
    const unsigned best=std::distance(std::begin(all_min_dist2),std::min_element(std::begin(all_min_dist2),std::end(all_min_dist2)));
    if(all_min_dist2[best]<threshold2_)
      min_k=all_min_k[best];
  }
  return min_k;
}

void OPESexplore::dumpProbToFile()
{
  if(storeOldProb_)
    probOfile_.clearFields();
  else if(walker_rank_==0)
    probOfile_.rewind();

  probOfile_.addConstantField("biasfactor");
  probOfile_.addConstantField("epsilon");
  probOfile_.addConstantField("kernel_cutoff");
  probOfile_.addConstantField("compression_threshold");
  probOfile_.addConstantField("zed");
  probOfile_.addConstantField("sum_weights");
  probOfile_.addConstantField("sum_weights2");
  for(unsigned i=0; i<ncv_; i++) //print periodicity of CVs
    probOfile_.setupPrintValue(getPntrToArgument(i));
  probOfile_.printField("biasfactor",biasfactor_);
  probOfile_.printField("epsilon",epsilon_);
  probOfile_.printField("kernel_cutoff",sqrt(cutoff2_));
  probOfile_.printField("compression_threshold",sqrt(threshold2_));
  probOfile_.printField("zed",Zed_);
  probOfile_.printField("sum_weights",sum_weights_);
  probOfile_.printField("sum_weights2",sum_weights2_);
  for(unsigned k=0; k<kernels_.size(); k++)
  {
    probOfile_.printField("time",getTime());
    for(unsigned i=0; i<ncv_; i++)
      probOfile_.printField(getPntrToArgument(i),kernels_[k].center[i]);
    for(unsigned i=0; i<ncv_; i++)
      probOfile_.printField("sigma_"+getPntrToArgument(i)->getName(),kernels_[k].sigma[i]);
    probOfile_.printField("height",kernels_[k].height);
    probOfile_.printField();
  }
  if(!storeOldProb_)
    probOfile_.flush();
}

inline double OPESexplore::evaluateKernel(const kernel& G,const std::vector<double>& x) const
{ //NB: cannot be a method of kernel class, because uses external variables (for cutoff)
  double norm2=0;
  for(unsigned i=0; i<ncv_; i++)
  {
    const double diff_i=difference(i,G.center[i],x[i])/G.sigma[i];
    norm2+=diff_i*diff_i;
    if(norm2>=cutoff2_)
      return 0;
  }
  return G.height*(std::exp(-0.5*norm2)-val_at_cutoff_);
}

inline double OPESexplore::evaluateKernel(const kernel& G,const std::vector<double>& x, std::vector<double> & acc_der)
{ //NB: cannot be a method of kernel class, because uses external variables (for cutoff)
  double norm2=0;
  std::vector<double> diff(ncv_);
  for(unsigned i=0; i<ncv_; i++)
  {
    diff[i]=difference(i,G.center[i],x[i])/G.sigma[i];
    norm2+=diff[i]*diff[i];
    if(norm2>=cutoff2_)
      return 0;
  }
  const double val=G.height*(std::exp(-0.5*norm2)-val_at_cutoff_);
  for(unsigned i=0; i<ncv_; i++)
    acc_der[i]-=diff[i]/G.sigma[i]*val; //NB: we accumulate the derivative into der
  return val;
}

inline void OPESexplore::kernel::merge_me_with(const kernel & other)
{
  const double h=height+other.height;
  for(unsigned i=0; i<center.size(); i++)
  {
    const double c_i=(height*center[i]+other.height*other.center[i])/h;
    const double s_my_part=height*(sigma[i]*sigma[i]+center[i]*center[i]);
    const double s_other_part=other.height*(other.sigma[i]*other.sigma[i]+other.center[i]*other.center[i]);
    const double s2_i=(s_my_part+s_other_part)/h-c_i*c_i;
    center[i]=c_i;
    sigma[i]=sqrt(s2_i);
  }
  height=h;
}

}
}
