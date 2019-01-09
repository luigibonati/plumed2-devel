/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2016,2017 The plumed team
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
#include "Bias.h"
#include "core/PlumedMain.h"
#include "core/ActionRegister.h"
#include "core/ActionSet.h"
#include "core/Atoms.h"
#include "tools/Communicator.h"
#include "tools/Dynet_MLP.h"

using namespace std;
using namespace dynet;

namespace PLMD {
namespace bias {

//+PLUMEDOC BIAS NN_VES
/*
 * To be added
*/
//+ENDPLUMEDOC

typedef std::vector< std::vector<float> > matrix;

class NeuralNetworkVes : public Bias {

private:
   
  unsigned dim_;
  vector<int> nodes_;
  
  double beta_;
  double stride_;
  double tau_;
  double lrate_;  

//dynet
  ParameterCollection m_;
  Trainer *trainer_;
  ComputationGraph cg_;
  MLP nn_;

//calculation of omega
  int counter_;
  //matrix old_s_;
  vector<float> old_s_;

//target distribution
  float min_, max_;
  float ds_; 
  int nbins_;
  vector<float> s_grid;
  vector<float> target_;

//auxiliary
  int iter;
  vector<vector<float>> grad_w;
  vector<vector<float>> grad_b;
  vector<vector<float>> ave_grad_w;
  vector<vector<float>> ave_grad_b;
  vector<vector<float>> target_grad_w;
  vector<vector<float>> target_grad_b;


//output values
  Value* valueOmega;
  Value* valueOmegaBias;
  Value* valueOmegaTarget;
  Value* valueForceTot2;

//class private methods
  void update_coeffs();

public:
  NeuralNetworkVes(const ActionOptions&);
  ~NeuralNetworkVes();
  void calculate();
  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(NeuralNetworkVes,"NN_VES")

void NeuralNetworkVes::registerKeywords(Keywords& keys) {
  Bias::registerKeywords(keys);
  keys.use("ARG");
  keys.add("compulsory","NODES","neural network architecture");
  keys.add("compulsory","RANGE","min and max of the range allowed");
  keys.add("optional","NBINS","bins for target distro");

  keys.add("optional","TEMP","temperature of the simulation");
  keys.add("optional","UP_STRIDE","the number of steps between updating coeffs");
  keys.add("optional","TAU","exponentially decaying average");
  keys.add("optional","LRATE","the step used for the minimization of the functional");

//  keys.addFlag("FIXED_BIAS",false,"keep a fixed bias throughout the whole simulation. INITIAL_COEFFS must be given");

//output components
  componentsAreNotOptional(keys);
  useCustomisableComponents(keys); //needed to have an unknown number of components
  keys.addOutputComponent("omega","default","estimate of the omega functional");
  keys.addOutputComponent("force2","default","total force");
}

NeuralNetworkVes::NeuralNetworkVes(const ActionOptions&ao):
  PLUMED_BIAS_INIT(ao),
  nn_(m_)
{
  //get # of inputs (CVs)
  dim_=getNumberOfArguments();
  //parse the NN architecture
  parseVector("NODES",nodes_);

  // parse temp  
  const double Kb=plumed.getAtoms().getKBoltzmann();
  double temp=0;
  parse("TEMP",temp);
  double KbT=Kb*temp;
  if(KbT<=0)
  {
    KbT=plumed.getAtoms().getKbT();
    plumed_massert(KbT>0,"your MD engine does not pass the temperature to plumed, you must specify it using TEMP");
  }
  beta_=1.0/KbT; //remember: with LJ use NATURAL UNITS

  // parse GRID INFO
  // range 
  vector<float> range;
  parseVector("RANGE",range);
  min_=range[0];
  max_=range[1];

  // nbins
  nbins_=50;
  parse("NBINS", nbins_);
  // spacing and grid values
  s_grid.resize(nbins_);
  ds_ = fabs(max_-min_)/nbins_;
  unsigned k = 0;
  for (auto&& s2 : s_grid)
    s2 = min_+(k++)*ds_;

  //define target distribution
  target_.resize(nbins_); 
  for (auto&& t : target_)
    t = 1/fabs(max_-min_);

// debug
  for(unsigned i=0;i<nbins_;i++){
	cerr << s_grid[i] << "  " << target_[i] << endl;
}

  // parse the update stride
  stride_=500;
  parse("UP_STRIDE",stride_);
  
  // reset counter
  counter_=0;
 
  // check whether to use an exponentially decaying average
  tau_=0;
  parse("TAU",tau_);
  if (tau_>0) 
    tau_*=stride_; 

  // parse learning rate
  lrate_=0.001;
  parse("LRATE",lrate_);

  checkRead();

//auxiliary
  iter=0;

//initialize dynet
  int cc=1;
  char pp[]="plumed";
  char *vv[]={pp};
  char** ivv=vv;
  DynetParams params = extract_dynet_params(cc,ivv,true);
  int random_seed=7; //TODO
  params.random_seed=random_seed;
  dynet::initialize(params);

//for debugging purpose
  cg_.set_immediate_compute(true);
  cg_.set_check_validity(true);

//defining the neural network
  string opt_name;
  trainer_ = new_trainer("Adam",m_,opt_name);

  log.printf("  Defining neural network with %d layers",nodes_.size() );
//define layers
  vector<Layer> layers;
  //input to first layer
  log.printf("  - Input: %d --> %d\n",dim_, nodes_[0]);
  layers.push_back( Layer(dim_, nodes_[0], RELU, 0.0) );
  //hidden layers
  for ( unsigned i=0; i<nodes_.size()-1; i++){
    layers.push_back( Layer(nodes_[i], nodes_[i+1], RELU, 0.0) );
   log.printf("  - Layer %d: %d --> %d\n",i,nodes_[i], nodes_[i+1]);
  }
  //last layer to output
  layers.push_back( Layer(nodes_.back(),1, LINEAR, 0.0) );
  log.printf("  - Output: %d --> %d\n",nodes_.back(), 1);

//create nn with specified architecture
  for (auto&& l : layers)
    nn_.append(m_,l,cg_);

//debug
  cg_.print_graphviz();

//define vars to store previous values
  //old_s_.resize(stride_, vector<float>(dim_));
  old_s_.resize(stride_);

//add all the output components
  addComponent("force2"); componentIsNotPeriodic("force2");
  valueForceTot2=getPntrToComponent("force2");
  addComponent("omega"); componentIsNotPeriodic("omega");
  valueOmega=getPntrToComponent("omega");
  addComponent("omegaBias"); componentIsNotPeriodic("omegaBias");
  valueOmegaBias=getPntrToComponent("omegaBias");
  addComponent("omegaTarget"); componentIsNotPeriodic("omegaTarget");
  valueOmegaTarget=getPntrToComponent("omegaTarget");


//printing some info
  log.printf("  Inputs: %d\n",dim_);
  log.printf("  Temperature T: %g\n",1./(Kb*beta_));
  log.printf("  Beta (1/Kb*T): %g\n",beta_);
  log.printf("  Number of arguments: %d\n",dim_);
  log.printf("  Stride for the ensemble average: %d\n",stride_);
  log.printf("  Learning Rate: %d\n",lrate_);
  if (tau_>1)
    log.printf("  Exponentially decaying average with weight=tau*stride=%d\n",tau_);
  // TODO:add nn and opt info
 
}

NeuralNetworkVes::~NeuralNetworkVes()
{
  delete trainer_;
}

void NeuralNetworkVes::calculate()
{
  cg_.clear();

  double bias_pot=0;
  double tot_force2=0;
  std::vector<float> current_S(dim_);  
  float min_bias = 0; 

//bias and forces
  //vector<float> s_input(dim_);
  const Dim d = {dim_};
  Expression s = input(cg_, d, &current_S);
  Expression s_scaled = s / (0.5*float(max_-min_));
  Expression bias_expr = nn_.run(s_scaled);

//shift the potential to the minimum
  Expression shift = input(cg_,  &min_bias);
  Expression bias_shift = bias_expr - shift;

//get currect CVs
  for(unsigned i=0; i<dim_; i++)
    current_S[i]=getArgument(i);

  bool DEBUG =false;

//debug
	if (counter_ % 100 == 0 && DEBUG ) {
	cerr << "ITER: " << iter << " - STEP: " << counter_ << endl;
	cerr << "s\t:" << as_scalar(cg_.forward(s)) << endl; 
  	cerr << "s'\t:" << as_scalar(cg_.forward(s_scaled)) << endl;
  	cerr << "V(s')\t:" << as_scalar(cg_.forward(bias_expr)) << endl;
  	}
//propagate and backprop to get forces
  bias_pot = as_scalar(cg_.forward(bias_expr));
  cg_.backward(bias_expr,true);
  vector<float> force = as_vector(s.gradient());

//set values
  setBias(bias_pot);
  for (unsigned i=0; i<dim_; i++){
    tot_force2+=pow(force[i],2);
    setOutputForce(i,force[i]);
  }
  valueForceTot2->set(tot_force2);

/*
//save cv value for later
  //old_s_[counter_++] = s_input;
  old_s_[counter_++] = current_S[0];
*/
  //accumulate gradients
  grad_w = nn_.get_grad_w();
  grad_b = nn_.get_grad_b();
  sum_grad(ave_grad_w,grad_w);
  sum_grad(ave_grad_b,grad_b);

//update parameters
  if (counter_==stride_){
    	if (DEBUG) cerr << "iter : " << iter << endl;
    	if (DEBUG) cerr << "old_s_vector " << endl;
    	for (unsigned i=0; i<old_s_.size(); i++){
      		if (DEBUG) cerr << old_s_[i] << " - ";
    	}
    	if (DEBUG) cerr << endl << endl;
    iter++;
    //compute loss function

    // -- (2) target distribution
    // open stream to print bias to file
    ofstream biasfile;
    string name;
    int B_STR = 10; 
    if( iter % B_STR == 0) {
	name = "bias.iter-"+to_string(iter); 
	biasfile.open(name.c_str());
        biasfile << "#s\tV(s)" << endl;
    }

    	if (DEBUG) cerr << " -- target_dist -- " << endl;  
    current_S[0] = s_grid[0];
    Expression omega_t1 = nn_.run(bias_expr);
    min_bias = as_scalar(cg_.forward( nn_.run(bias_expr )));
    
    cerr << "min: " << min_bias << endl; 
    
    for (unsigned i=1; i<s_grid.size(); i++){
      current_S[0] = s_grid[i];
      // print stuff
      float aux_bias = as_scalar(cg_.forward(bias_expr));
      if (aux_bias < min_bias ) min_bias = aux_bias;
//      if( iter % B_STR == 0) biasfile << current_S[0] << "\t" << aux_bias << endl;
      omega_t1= omega_t1 + nn_.run(bias_expr);
      	if (DEBUG) cerr << i << " | omega " << as_scalar(cg_.forward(omega_t1)) << endl;
    }

    current_S[0] = s_grid[0];
    Expression omega_t = nn_.run(bias_shift);
    for (unsigned i=1; i<s_grid.size(); i++){
      current_S[0] = s_grid[i];
      if( iter % B_STR == 0) biasfile << current_S[0] << "\t" << as_scalar(cg_.forward(bias_shift)) << endl;
      omega_t = omega_t + nn_.run(bias_shift);
    }

    //close printing
    if( iter % B_STR == 0) biasfile.close();

    omega_t = omega_t * (ds_);
    	if (DEBUG) cerr << " normalize " << as_scalar(cg_.forward(omega_t)) << endl;

    float aux_omega_t = as_scalar( cg_.forward(omega_t) );
    cg_.backward(omega_t);
    valueOmegaTarget->set( aux_omega_t );   

    //(1) biased term
    current_S[0] = old_s_[0];					//update input	
    Expression av_exp_bias = dynet::exp(bias_shift*beta_);	//exp(beta*V(s_i))
    Expression omega = av_exp_bias;				
    for (unsigned i=1; i<stride_; i++){
      current_S[0]=old_s_[i];
      omega = omega + av_exp_bias;				//sum_i exp(beta*V(s_i))
      	if (DEBUG) cerr << i << " | curr_S: " << current_S[0] << "\te(bv)\t:" << as_scalar(cg_.forward(av_exp_bias)) << "\t--> omega\t" << as_scalar(cg_.forward(omega))   << endl;
    }
    	if (DEBUG) cerr << "done" << endl;

    omega = omega * (1/stride_);
    	if (DEBUG) cerr << " normalize " << as_scalar(cg_.forward(omega)) << endl;
    omega = dynet::log(omega);					//omega = -1/beta * ln (sum_i ...)
    	if (DEBUG) cerr << " log " << as_scalar(cg_.forward(omega)) << endl;
    omega = omega * (-1./beta_);
    	if (DEBUG) cerr << " -1/beta " << as_scalar(cg_.forward(omega)) << endl;
    
    	if (DEBUG) cerr << "setting things.. " << endl;
    	if (DEBUG) cerr << "omega" << endl;
    float aux_omega_bias = as_scalar( cg_.forward(omega) );
    	if (DEBUG) cerr << "backward" << endl;
    cg_.backward(omega);
    	if (DEBUG) cerr << "exporting value" << endl;
    valueOmegaBias->set( aux_omega_bias );
    	if (DEBUG) cerr << "ok" << endl;


    //sum contributions and update 
    valueOmega->set( aux_omega_bias + aux_omega_t );
    //cg_.backward(omega);
    trainer_->update();

    //const Dim d_stride = {stride_};
    //Expression av_bias = input(cg_, d_stride, &old_bias);
    //Expression av_exp_bias =   
    
    counter_=0;
  }
/*
//get CVs, update bias and set forces
  double bias_pot=0;
  double tot_force2=0;
  std::vector<double> current_S(dim_);
  for (unsigned i=0; i<dim_; i++)
  {
    current_S[i]=getArgument(i);
    bias_pot+=mean_coeff_[i]*current_S[i];
    const double force_i=-1*mean_coeff_[i];
    tot_force2+=pow(force_i,2);
    setOutputForce(i,force_i);
  }
  setBias(bias_pot);
  valueForceTot2->set(tot_force2);
  if (fixed_bias_) //stop here in case of FIXED_BIAS
    return;

//update ensemble averages
  av_counter_++;
  double current_V=0;
  for (unsigned i=0; i<dim_; i++)
  {
    av_S_[i]+=(current_S[i]-av_S_[i])/av_counter_;
    for (unsigned j=i; j<dim_; j++)
      av_prod_S_[get_index(i,j)]+=(current_S[i]*current_S[j]-av_prod_S_[get_index(i,j)])/av_counter_;
    current_V+=mean_coeff_[i]*current_S[i];
  }
  av_exp_V_+=(exp(beta_*current_V)-av_exp_V_)/av_counter_;

//update coeffs stuff (to be done after the forces are set)
  if (av_counter_==av_stride_)
  {
    update_coeffs();
    //reset the ensemble averages
    av_counter_=0;
    for (unsigned i=0; i<dim_; i++)
      av_S_[i]=0;
    for (unsigned ij=0; ij<av_prod_S_.size(); ij++)
      av_prod_S_[ij]=0;
    av_exp_V_=0;
  }
*/
}
/*
void NeuralNetworkVes::update_coeffs()
{
//combining the averages of multiple walkers
  if(walkers_num_>1)
  {
    if(comm.Get_rank()==0) //sum only once: in the first rank of each walker
    {
      multi_sim_comm.Sum(av_S_);
      multi_sim_comm.Sum(av_prod_S_);
      multi_sim_comm.Sum(av_exp_V_);
      for(unsigned i=0; i<dim_; i++)
        av_S_[i]/=walkers_num_;
      for(unsigned ij=0; ij<av_prod_S_.size(); ij++)
        av_prod_S_[ij]/=walkers_num_; //WARNING: is this the best way to implement mw into this algorithm? Some theoretical work should be done...
      av_exp_V_/=walkers_num_;
    }
    if (comm.Get_size()>1)//if there are more ranks for each walker, everybody has to know
    {
      comm.Bcast(av_S_,0);
      comm.Bcast(av_prod_S_,0);
      comm.Bcast(av_exp_V_,0);
    }
  }

//build the gradient and the Hessian of the functional
  std::vector<double> grad_omega(dim_);
  std::vector<double> hess_omega_increm(dim_);//inner product between the Hessian and the increment
  mean_counter_++;
  unsigned mean_weight=mean_counter_;
  if (mean_weight_tau_>0 && mean_weight_tau_<mean_counter_)
    mean_weight=mean_weight_tau_;
  for (unsigned i=0; i<dim_; i++)
  {
    grad_omega[i]=target_av_S_[i]-av_S_[i];
    if (target_side_[i]*grad_omega[i]<0)
      grad_omega[i]=0;
    if (target_2sigma2_[i]!=0)
      grad_omega[i]*=(1-exp(-pow(grad_omega[i],2)/target_2sigma2_[i]));
    for(unsigned j=0; j<dim_; j++)
      hess_omega_increm[i]+=(av_prod_S_[get_index(i,j)]-av_S_[i]*av_S_[j])*(inst_coeff_[j]-mean_coeff_[j]);
    hess_omega_increm[i]*=beta_;
  }
//update all the coefficients
  double mean_gradOmega2=0;
  double av_V=0;
  for (unsigned i=0; i<dim_; i++)
  {
    inst_coeff_[i]-=minimization_step_[i]*(grad_omega[i]+hess_omega_increm[i]);
    mean_coeff_[i]+=(inst_coeff_[i]-mean_coeff_[i])/mean_weight;
    mean_gradOmega_[i]+=(grad_omega[i]-mean_gradOmega_[i])/mean_weight;
    mean_gradOmega2+=pow(mean_gradOmega_[i],2);
    av_V+=mean_coeff_[i]*av_S_[i];
    //update also the values
    valueInstCoeff[i]->set(inst_coeff_[i]);
    valueMeanCoeff[i]->set(mean_coeff_[i]);
    valueGradOmega[i]->set(grad_omega[i]);
  }
  valueAverGradRMS->set(sqrt(mean_gradOmega2/dim_));
  valueKLbias->set(std::log(av_exp_V_)-beta_*av_V);
}

*/
}
}


