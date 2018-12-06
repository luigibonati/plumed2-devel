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
 * Implementation of VES for experimental target. Very similar to Bussi's MaxEnt.
 * Bias is linear in the CVs
 * It is possible to add Gaussian uncertainties to the target
*/
//+ENDPLUMEDOC

class NeuralNetworkVes : public Bias {

private:
   
  unsigned dim_;
  std::vector<int> nodes_;
  
  double beta_;
  double stride_;
  double tau_;
  double lrate_;  

//dynet
  ParameterCollection m_;
  Trainer *trainer_;
  ComputationGraph cg_;
  MLP nn_;

//output values
  Value* valueOmega;
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
// how to parse flags!!
//  fixed_bias_=false;
//  parseFlag("FIXED_BIAS",fixed_bias_); 

  //get # of inputs (CVs)
  dim_=getNumberOfArguments();
  //parse the NN architecture
  parseVector("NODES",nodes_);

//initialize vectors!!
//  valueGradOmega.resize(dim_);

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

  // parse the update stride
  stride_=500;
  parse("UP_STRIDE",stride_);

  // check whether to use an exponentially decaying average
  tau_=0;
  parse("TAU",tau_);
  if (tau_>0) 
    tau_*=stride_; 

  // parse learning rate
  lrate_=0.001;
  parse("LRATE",lrate_);

  checkRead();

//initialize dynet
  int cc=1;
  char pp[]="plumed";
  char *vv[]={pp};
  char** ivv=vv;
  DynetParams params = extract_dynet_params(cc,ivv,true);
  int random_seed=7; //TODO
  params.random_seed=random_seed;
  dynet::initialize(params);

//defining the neural network
  string opt_name;
  trainer_ = new_trainer("Adam",m_,opt_name);

cerr << dim_ << endl;
for ( unsigned i=0; i<nodes_.size(); i++)
	cerr << nodes_[i] << endl;

//define layers
  vector<Layer> layers;
  //input to first layer
  log.printf("  [NN] First layer: %d - %d\n",dim_, nodes_[0]);
  layers.push_back( Layer(dim_, nodes_[0], RELU, 0.0) );
  //hidden layers
  for ( unsigned i=0; i<nodes_.size()-1; i++){
    layers.push_back( Layer(nodes_[i], nodes_[i+1], RELU, 0.0) );
   log.printf("  [NN] Layer %d: %d - %d\n",i,nodes_[i], nodes_[i+1]);
  }
  //last layer to output
  layers.push_back( Layer(nodes_.back(),1, LINEAR, 0.0) );
  log.printf("  [NN] Last layer: %d - %d\n",nodes_.back(), 1);

//create nn with specified architecture
  for (auto&& l : layers)
    nn_.append(m_,l);

//debug
  cg_.print_graphviz();

//add all the output components
  addComponent("force2"); componentIsNotPeriodic("force2");
  valueForceTot2=getPntrToComponent("force2");
  addComponent("omega"); componentIsNotPeriodic("omega");
  valueOmega=getPntrToComponent("omega");

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
  double bias_pot=0;
  double tot_force2=0;
  std::vector<double> current_S(dim_);  

//bias and forces
  vector<float> s_input(dim_);
  const Dim d = {dim_};
  Expression s = input(cg_, d, &s_input);
  Expression bias_expr = nn_.run(s,cg_);

//get currect CVs
  for (unsigned i=0; i<dim_; i++)
    current_S[i]=getArgument(i);

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


