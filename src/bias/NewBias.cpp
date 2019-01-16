/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2012-2018 The plumed team
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
#include <torch/torch.h>

using namespace std;


namespace PLMD {
namespace bias {

//+PLUMEDOC BIAS NN_VES
/*
Work in progress
*/
//+ENDPLUMEDOC

template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b){
    assert(a.size() == b.size());
    std::vector<T> result;
    result.reserve(a.size());
    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::plus<T>());
    return result;
}
template <class T, class Q>
std::vector <T> operator* (const Q c, std::vector <T> A){
    std::transform (A.begin (), A.end (), A.begin (),
                 std::bind1st (std::multiplies <T> () , c)) ;
    return A ;
}
std::vector<float> tensor_to_vector(const torch::Tensor& x) {
    return std::vector<float>(x.data<float>(), x.data<float>() + x.numel());
}
float tensor_to_scalar(const torch::Tensor& x){
    return x.item<float>();
}

struct Net : torch::nn::Module {
  Net( vector<int> nodes, bool periodic ) : _layers() {
    //get number of hidden layers 
    _hidden=nodes.size() - 2;
    //check wheter to enforce periodicity
    _periodic=periodic;
    if(_periodic)
        nodes[0] *= 2;
    //by default do not normalize TODO change this
    _normalize=false;
    //register modules
    for(int i=0; i<_hidden; i++)
        _layers.push_back( register_module("fc"+to_string(i+1), torch::nn::Linear(nodes[i], nodes[i+1])) );
    //register output layer
    _out = register_module("out", torch::nn::Linear(nodes[_hidden], nodes[_hidden+1]));
  }

  ~Net() {}

  void setRange(float m, float M){
    _normalize=true;
    _min=m;
    _max=M;
  }

  torch::Tensor forward(torch::Tensor x) {
    //enforce periodicity (encode every input x into {cos(x), sin(x)} )
    if(_periodic)
        x = at::stack({at::sin(x),at::cos(x)},1).view({x.size(0),2});
    //normalize (this works only if you have a batch!)
    //x = x.sub_(x.mean()).div_(x.std());
    //normalize (with range) 
    if(_normalize){
        float x_mean = (_max+_min)/2.;
	float x_range = (_max-_min)/2.;
    	x = (x-x_mean)/x_range; 
    }
    //now propagate
    for(unsigned i=0; i<_layers.size(); i++){
        x = torch::relu(_layers[i]->forward(x));
    }
    x = _out->forward(x);
    return x;
  }

  /*--class members--*/
  int _hidden;
  bool _periodic, _normalize;
  float _min, _max;
  vector<torch::nn::Linear> _layers;
  torch::nn::Linear _out = nullptr;
};

class NeuralNetworkVes : public Bias {
private:
/*--neural_network_setup--*/
  unsigned 		nn_input_dim;
  vector<int> 		nn_nodes;
  shared_ptr<Net>	nn_model;
  shared_ptr<torch::optim::Adam> nn_opt; //TODO generalize optimizer
/*--parameters and options--*/
  double 		o_beta;
  int	 		o_stride;
  int	 		o_print;
  int 			o_tau;
  double 		o_lrate;
  bool			o_periodic; //TODO: PARSE PERIODIC CVs and pass it as an array of booleans 
/*--counters--*/
  int			c_iter;
/*--target distribution--*/
  float 		t_min, t_max; 
  float 		t_ds;
  int 			t_nbins;
  vector<float> 	t_grid,t_target;
/*--reweight--*/
  float 		r_ct; 
  float 		r_bias;
/*--gradients vectors--*/
  vector<vector<float>>	g_;
  vector<vector<float>>	g_mean;
  vector<vector<float>>	g_target;
  vector<torch::Tensor> g_tensor;
/*--outputvalues--*/
  Value*		v_Omega;
  Value*		v_OmegaBias;
  Value*		v_OmegaTarget;
  Value*		v_ForceTot2;
/*--methods-*/
  void 			update_coeffs();

public:
  explicit NeuralNetworkVes(const ActionOptions&);
  ~NeuralNetworkVes() {};
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
  keys.add("optional","TAU","exponentially decaying average");
  keys.add("optional","LRATE","the step used for the minimization of the functional");
  keys.add("optional","AVE_STRIDE","the stride for the update of the bias");
  keys.add("optional","PRINT_STRIDE","the stride for printing the bias (iterations)");
   componentsAreNotOptional(keys);
  useCustomisableComponents(keys); //needed to have an unknown number of components
  // Should be _bias below
  keys.addOutputComponent("_bias","default","one or multiple instances of this quantity can be referenced elsewhere in the input file. "
                          "these quantities will named with  the arguments of the bias followed by "
                          "the character string _bias. These quantities tell the user how much the bias is "
                          "due to each of the colvars.");
  keys.addOutputComponent("omega","default","estimate of the omega functional");
  keys.addOutputComponent("rct","default","estimate of the omega functional");
  keys.addOutputComponent("rbias","default","estimate of the omega functional");
  keys.addOutputComponent("force2","default","total force");
}

NeuralNetworkVes::NeuralNetworkVes(const ActionOptions&ao):
  PLUMED_BIAS_INIT(ao)
{
  //for debugging TODO remove
  torch::manual_seed(0);
  /*--NN OPTIONS--*/
  //get # of inputs (CVs)
  nn_input_dim=getNumberOfArguments();
  //parse the NN architecture
  parseVector("NODES",nn_nodes);
  //todo: check dim_ and first dimension
  /*--TEMPERATURE--*/  
  double temp=0;
  parse("TEMP",temp);
  double Kb = plumed.getAtoms().getKBoltzmann();
  double KbT = plumed.getAtoms().getKBoltzmann()*temp;
  if(KbT<=0){
    KbT=plumed.getAtoms().getKbT();
    plumed_massert(KbT>0,"your MD engine does not pass the temperature to plumed, you must specify it using TEMP");
  }
  o_beta=1.0/KbT; //remember: with LJ use NATURAL UNITS
  /*--TARGET DISTRIBUTION--*/
  // range 
  vector<float> range;
  parseVector("RANGE",range);
  t_min=range[0];
  t_max=range[1];
  // nbins
  t_nbins=50;
  parse("NBINS", t_nbins);
  // spacing and grid values
  t_grid.resize(t_nbins);
  t_ds = fabs(t_max-t_min)/t_nbins;
  unsigned k = 0;
  for (auto&& s2 : t_grid)
    s2 = t_min+(k++)*t_ds;
  //define target distribution (UNIFORM)
  t_target.resize(t_nbins);
  for (auto&& t : t_target)
    //t = 1/fabs(t_max-t_min); 
    //t = 1./t_nbins;			//WARNING
  /*--PARAMETERS--*/
  // update stride
  o_stride=500;
  parse("AVE_STRIDE",o_stride);
 // update stride
  o_print=1000;
  parse("PRINT_STRIDE",o_print);
 // check whether to use an exponentially decaying average
  o_tau=0;
  parse("TAU",o_tau);
  if (o_tau>0)
    o_tau*=o_stride;
  // parse learning rate
  o_lrate=0.001;
  parse("LRATE",o_lrate);
  // check if args are periodic TODO IMPROVE TO DEAL WITH MORE CVs
  o_periodic=getPntrToArgument(0)->isPeriodic();
   // reset counters
  c_iter=0;
  /*--PARSING DONE --*/
  checkRead();
  /*--NEURAL NETWORK SETUP --*/
  log.printf("  Defining neural network with %d layers",nn_nodes.size() );
  //nn_model = new Net (nn_nodes, o_periodic);
  nn_model = make_shared<Net>(nn_nodes, o_periodic);
  //normalize setup TODO integrate this in a seamless way
  nn_model->setRange(t_min, t_max);
  nn_opt = make_shared<torch::optim::Adam>(nn_model->parameters(), /*lr=*/o_lrate );
  /*--CREATE AUXILIARY VECTORS--*/
  //dummy backward pass in order to have the grads defined
  vector<torch::Tensor> params = nn_model->parameters();
  torch::Tensor y = nn_model->forward( torch::rand({1}).view({1,nn_input_dim}) );
  y.backward();
  nn_opt->zero_grad();
  //Define auxiliary vectors to store gradients
  for (auto&& p : params ){
    cout << p << endl;
    vector<float> gg = tensor_to_vector( p.grad() );
    g_.push_back(gg);
    g_mean.push_back(gg);
    g_target.push_back(gg);
    g_tensor.push_back(p);
  }
  /*--SET OUTPUT COMPONENTS--*/
  addComponent("force2"); componentIsNotPeriodic("force2");
  v_ForceTot2=getPntrToComponent("force2");
  addComponent("omega"); componentIsNotPeriodic("omega");
  v_Omega=getPntrToComponent("omega");
  addComponent("omegaBias"); componentIsNotPeriodic("omegaBias");
  v_OmegaBias=getPntrToComponent("omegaBias");
  addComponent("omegaTarget"); componentIsNotPeriodic("omegaTarget");
  v_OmegaTarget=getPntrToComponent("omegaTarget");
  /*--LOG INFO--*/
  log.printf("  Inputs: %d\n",nn_input_dim);
  log.printf("  Temperature T: %g\n",1./(Kb*o_beta));
  log.printf("  Beta (1/Kb*T): %g\n",o_beta);
  log.printf("  Stride for the ensemble average: %d\n",o_stride);
  log.printf("  Learning Rate: %d\n",o_lrate);
  if (o_tau>1)
    log.printf("  Exponentially decaying average with weight=tau*stride=%d\n",o_tau);
  // TODO:add nn and opt info
}

void NeuralNetworkVes::calculate() {
  double bias_pot=0;
  double tot_force2=0;
  //get current CVs
  vector<float> current_S(nn_input_dim);
  for(unsigned i=0; i<nn_input_dim; i++)
    current_S[i]=getArgument(i);
  //convert current CVs into torch::Tensor
  torch::Tensor input_S = torch::tensor(current_S).view({1,nn_input_dim});
  input_S.set_requires_grad(true);
  //propagate to get the bias
  nn_opt->zero_grad();
  auto output = nn_model->forward( input_S );
  bias_pot = output.item<float>();
  //backprop to get forces
  output.backward();
  vector<float> force=tensor_to_vector( input_S.grad() );
  //set bias
  setBias(bias_pot);
  //set forces
  for (unsigned i=0; i<nn_input_dim; i++){
    tot_force2+=pow(force[i],2);
    setOutputForce(i,-force[i]);
  }
  v_ForceTot2->set(tot_force2);
  //accumulate gradients
  vector<torch::Tensor> p = nn_model->parameters();
  for (unsigned i=0; i<p.size(); i++){
    vector<float> gg = tensor_to_vector( p[i].grad() );
    g_mean[i] = g_mean[i] + gg;
  }

  /*--UPDATE PARAMETERS--*/
  if(getStep()%o_stride==0){
    c_iter++; 
    //TODO use internal routines
    ofstream biasfile;
    if( c_iter % o_print == 0){
        biasfile.open(("bias.iter-"+to_string(c_iter)).c_str());
        biasfile << "#s\tV(s)" << endl;
    }
    /**Biased ensemble contribution**/
    //normalize average gradient
    for (unsigned i=0; i<g_mean.size(); i++)
      g_mean[i] = -(1./o_stride) * g_mean[i];

    /**Target distribution contribution**/
/*
    for (unsigned i=0; i<t_grid.size(); i++){
      //scan over grid
      current_S[0] = t_grid[i];
      torch::Tensor input_S_target = torch::tensor(current_S).view({1,nn_input_dim});
      nn_opt->zero_grad();
      output = nn_model->forward( input_S_target );
      bias_pot = output.item<float>();
      output.backward();
      p = nn_model->parameters();
      for (unsigned i=0; i<p.size(); i++){
        vector<float> gg = tensor_to_vector( p[i].grad() );
        gg = t_target[i] * gg;
        g_target[i] = g_target[i] + gg;
      }
      //print bias on file
      if( c_iter % o_print == 0) biasfile << current_S[0] << "\t" << bias_pot << endl;
    }
*/
    /**alternative with minibatch**/
    torch::Tensor batch_S = torch::tensor(t_grid).view({t_nbins,nn_input_dim});    
    nn_opt->zero_grad();
    output = nn_model->forward( batch_S );
    output.backward();
    p = nn_model->parameters();
    for (unsigned i=0; i<p.size(); i++){
        vector<float> gg = tensor_to_vector( p[i].grad() );
        gg = t_target[i] * gg;
        g_target[i] = g_target[i] + gg;
    } 
    vector<float> bias_grid = tensor_to_vector(output);
    if( c_iter % o_print == 0)
      for(int i=0; i<t_grid.size(); i++)
	biasfile << t_grid[i] << "\t" << bias_grid[i] << endl;
    //reset gradients
    nn_opt->zero_grad();
    //normalize target gradient (if different from uniform) TODO
    //for (unsigned i=0; i<g_target.size(); i++)
    //  g_target[i] = ( (t_max-t_min)/t_nbins ) * g_target[i];
    //close the ostream
    if( c_iter % o_print == 0) biasfile.close(); 
    /**Assign new gradient and update coefficients**/
    for (unsigned i=0; i<g_.size()-1; i++){  //until size-1 since we do not want to update the bias of the output layer
	//bias-target
	g_[i]=g_mean[i]+g_target[i];
        //vector to Tensor
        g_tensor[i] = torch::tensor(g_[i]).view( nn_model->parameters()[i].sizes() );
        //assign tensor to derivatives
        nn_model->parameters()[i].grad() = g_tensor[i].detach();
        //reset mean grads
        std::fill(g_[i].begin(), g_[i].end(), 0.);
        std::fill(g_mean[i].begin(), g_mean[i].end(), 0.);
        std::fill(g_target[i].begin(), g_target[i].end(), 0.);
      }
      //update the parameters
      nn_opt->step();
  }
}

}
}


