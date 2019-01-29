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
// exp_added(expsum,expvalue)
// expsum=log(exp(expsum)+exp(expvalue)
inline void exp_added(double& expsum,double expvalue)
{
    if(expsum>expvalue)
	expsum=expsum+std::log(1.0+exp(expvalue-expsum));
    else
	expsum=expvalue+std::log(1.0+exp(expsum-expvalue));
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
    for(unsigned i=0; i<_layers.size(); i++)
        x = torch::elu(_layers[i]->forward(x));
    
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
  float 		o_beta;
  int	 		o_stride;
  int	 		o_print;
  int			o_target;
  int 			o_tau;
  float 		o_lrate;
  float			o_gamma;
  bool			o_periodic; //TODO: PARSE PERIODIC CVs and pass it as an array of booleans 
/*--counters--*/
  int			c_iter;
/*--target distribution--*/
  float 		t_min, t_max; 
  float 		t_ds;
  int 			t_nbins;
  vector<float> 	t_grid,t_target_ds,t_bias_hist,t_fes;
/*--reweight--*/
  float 		r_ct; 
  float 		r_bias;
/*--gradients vectors--*/
  vector<vector<float>>	g_;
  vector<vector<float>>	g_mean;
  vector<vector<float>>	g_target;
  vector<torch::Tensor> g_tensor;
/*--outputvalues--*/
  Value*		v_kl;
  Value*		v_rct;
  Value*		v_rbias;
  Value*		v_ForceTot2;
/*--methods-*/
  void 			update_coeffs();
/*--extra--*/
  float			bias_min;

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
  keys.add("optional","TAU","exponentially decaying average for KL");
  keys.add("optional","LRATE","the step used for the minimization of the functional");
  keys.add("optional","GAMMA","gamma value for well-tempered distribution");
  keys.add("optional","AVE_STRIDE","the stride for the update of the bias");
  keys.add("optional","PRINT_STRIDE","the stride for printing the bias (iterations)");
  keys.add("optional","TARGET_STRIDE","the stride for updating the iterations (iterations)");
   componentsAreNotOptional(keys);
  useCustomisableComponents(keys); //needed to have an unknown number of components
  // Should be _bias below
  keys.addOutputComponent("_bias","default","one or multiple instances of this quantity can be referenced elsewhere in the input file. "
                          "these quantities will named with  the arguments of the bias followed by "
                          "the character string _bias. These quantities tell the user how much the bias is "
                          "due to each of the colvars.");
  keys.addOutputComponent("kl","default","kl divergence between bias and target");
  keys.addOutputComponent("rct","default","c(t) term");
  keys.addOutputComponent("rbias","default","bias-c(t)");
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
  //define target distribution (UNIFORM): we call p(s)*ds=t_target_ds
  t_target_ds.resize(t_nbins);
  for (auto&& t : t_target_ds)
    t = 1./t_nbins;			//normalization
    //t = 1/fabs(t_max-t_min); 
  //define histogram for computing biased distribution
  t_bias_hist.resize(t_nbins);
  std::fill(t_bias_hist.begin(), t_bias_hist.end(), 0.000001);
  //define grid for fes
  t_fes.resize(t_nbins);
  std::fill(t_fes.begin(), t_fes.end(), 0.);
  /*--PARAMETERS--*/
  // update stride
  o_stride=500;
  parse("AVE_STRIDE",o_stride);
 // print stride
  o_print=1000;
  parse("PRINT_STRIDE",o_print);
 // update stride
  o_target=100;
  parse("TARGET_STRIDE",o_target);
 
  // check whether to use an exponentially decaying average for the calculation of KL
  o_tau=0;
  parse("TAU",o_tau);
  if (o_tau>0)
    o_tau*=o_stride;
  // parse learning rate
  o_lrate=0.001;
  parse("LRATE",o_lrate); 
  // parse gamma
  o_gamma=0;
  parse("GAMMA",o_gamma);
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
  nn_opt = make_shared<torch::optim::Adam>(nn_model->parameters(), /*lr=*/o_lrate);
  //torch::optim::AdamOptions opt(o_lrate);
  //opt.amsgrad(true);
  //opt.beta1(0.99);
  //nn_opt = make_shared<torch::optim::Adam>(nn_model->parameters(), opt);
  /*--CREATE AUXILIARY VECTORS--*/
  //dummy backward pass in order to have the grads defined
  vector<torch::Tensor> params = nn_model->parameters();
  torch::Tensor y = nn_model->forward( torch::rand({1}).view({1,nn_input_dim}) );
  y.backward();
  //Define auxiliary vectors to store gradients
  nn_opt->zero_grad();
  for (auto&& p : params ){
    //cout << p << endl;
    vector<float> gg = tensor_to_vector( p.grad() );
    g_.push_back(gg);
    g_mean.push_back(gg);
    g_target.push_back(gg);
    g_tensor.push_back(p);
  }
  //bias_min initialize
  bias_min=0;
  /*--SET OUTPUT COMPONENTS--*/
  addComponent("force2"); componentIsNotPeriodic("force2");
  v_ForceTot2=getPntrToComponent("force2");
  addComponent("kl"); componentIsNotPeriodic("kl");
  v_kl=getPntrToComponent("kl");
  addComponent("rct"); componentIsNotPeriodic("rct");
  v_rct=getPntrToComponent("rct");
  addComponent("rbias"); componentIsNotPeriodic("rbias");
  v_rbias=getPntrToComponent("rbias");

  /*--LOG INFO--*/
  log.printf("  Inputs: %d\n",nn_input_dim);
  log.printf("  Temperature T: %g\n",1./(Kb*o_beta));
  log.printf("  Beta (1/Kb*T): %g\n",o_beta);
  log.printf("  Stride for the ensemble average: %d\n",o_stride);
  log.printf("  Learning Rate: %g\n",o_lrate);
  if (o_tau>0)
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
//  setBias(bias_pot);
  setBias(bias_pot-bias_min);
  //set forces
  for (unsigned i=0; i<nn_input_dim; i++){
    tot_force2+=pow(force[i],2);
    setOutputForce(i,-force[i]); //be careful of minus sign
  }
  v_ForceTot2->set(tot_force2);
  //accumulate gradients
  vector<torch::Tensor> p = nn_model->parameters();
  for (unsigned i=0; i<p.size(); i++){
    vector<float> gg = tensor_to_vector( p[i].grad() );
    g_mean[i] = g_mean[i] + gg;
  }
  //accumulate histogram for biased distribution TODO grid more than 1cv
  int idx=(current_S[0]-t_min)/t_ds;
    //check if outside the grid TODO crash?
  if(idx>=t_nbins) idx=t_nbins-1;
  if(idx<0) idx=0;
  //t_bias_hist[idx]++;
    //get current weight
  float weight=getStep()+1;
  if (o_tau>0 && weight>o_tau)
    weight=o_tau;
  t_bias_hist[idx]+=1./weight;
  /*--UPDATE PARAMETERS--*/
  if(getStep()%o_stride==0){
    c_iter++; 
    //TODO use internal routines
//    ofstream biasfile;
//    if( c_iter % o_print == 0){
//        biasfile.open(("bias.iter-"+to_string(c_iter)).c_str());
//        biasfile << "#\tV(s)" << endl;
//    }
    /**Biased ensemble contribution**/
    //normalize average gradient
    for (unsigned i=0; i<g_mean.size(); i++)
      g_mean[i] = -(1./o_stride) * g_mean[i];

    /**Target distribution contribution**/
    vector<float> bias_grid (t_nbins);
    for (unsigned i=0; i<t_grid.size(); i++){
      //scan over grid
      current_S[0] = t_grid[i];
      torch::Tensor input_S_target = torch::tensor(current_S).view({1,nn_input_dim});
      nn_opt->zero_grad();
      output = nn_model->forward( input_S_target );
      bias_grid[i] = output.item<float>();
      output.backward();
      p = nn_model->parameters();
      for (unsigned j=0; j<p.size()-1; j++){
        vector<float> gg = tensor_to_vector( p[j].grad() );
        gg = t_target_ds[i] * gg;
        g_target[j] = g_target[j] + gg;
      }
      //print bias on file
      //if( c_iter % o_print == 0) biasfile << current_S[0] << "\t" << bias_grid[i] << endl;
    }

    //shift bias to zero
/*
    bias_min = *std::min_element(bias_grid.begin(), bias_grid.end());
    for (unsigned i=0; i<bias_grid.size(); i++)
      bias_grid[i]-=bias_min;
*/ 
    //print bias to file
    if( c_iter % o_print == 0){
      ofstream biasfile;
      biasfile.open(("bias.iter-"+to_string(c_iter)).c_str());
      for (unsigned i=0; i<t_grid.size(); i++)
        biasfile << t_grid[i] << "\t" << bias_grid[i] << endl;
      biasfile.close();
    }
    /**alternative with minibatch**/ //be careful about target distribution normalization
/*
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
*/
    //reset gradients
    nn_opt->zero_grad();
    //normalize target gradient (if different from uniform) TODO or maybe not normalize at all?
    //for (unsigned i=0; i<g_target.size(); i++)
    // g_target[i] = ( (t_max-t_min)/t_nbins ) * g_target[i]; // if not using minibatch
      //g_target[i] = (t_max-t_min) * g_target[i];

    //close the ostream
    //if( c_iter % o_print == 0) biasfile.close(); 

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

    /*--COMPUTE REWEIGHT FACTOR--*/ 
    double log_sumebv=-1.0e38;
    double target_norm=0;
    //loop over grid
    for (unsigned i=0; i<t_grid.size(); i++){
      double log_target = std::log(t_target_ds[i]);	        
      double log_ebv= o_beta * bias_grid[i] + log_target;     	//beta*V(s)+log p(s)
      if(i==0) log_sumebv = log_ebv;				//sum exp with previous ones (see func. exp_added)
      else exp_added(log_sumebv,log_ebv);
      target_norm += t_target_ds[i];
    }
    //compute c(t)
    r_ct = (log_sumebv-std::log(target_norm))/o_beta;
    getPntrToComponent("rct")->set(r_ct); 
    //compute rbias
    r_bias = bias_pot-r_ct;
    getPntrToComponent("rbias")->set(r_bias);

    //--COMPUTE KL--
    //normalize bias histogram
    double bias_norm=0;
    for (auto& n : t_bias_hist)
      bias_norm += n;
    //t_bias_hist = (t_nbins/(t_max-t_min)) * t_bias_hist;
    //t_bias_hist = (1./sum * t_nbins/(t_max-t_min)) * t_bias_hist;
    //t_bias_hist = (1./o_stride * t_nbins/(t_max-t_min)) * t_bias_hist;
    //normalize distributions
    auto biased_dist = (1./bias_norm) * t_bias_hist;
    auto target_dist = (1./target_norm) * t_target_ds;
    //compute kl
    double kl=0;
    for (unsigned i=0; i<target_dist.size(); i++)
      kl+=biased_dist[i]*std::log(biased_dist[i]/target_dist[i]);
    getPntrToComponent("kl")->set(kl);
    //std::fill(t_bias_hist.begin(), t_bias_hist.end(), 0.000001);

    /*--UPDATE TARGET DISTRIBUTION--*/ 
    float sum_exp_beta_F = 0;
    if(o_target > 0 && c_iter % o_target == 0){
      //compute new estimate of the fes
      for (unsigned i=0; i<t_fes.size(); i++){
        t_fes[i] = - bias_grid[i] + (1./o_gamma) * t_fes[i];
        float exp_beta_F = std::exp( (-o_beta/o_gamma) * t_fes[i] ); 
	sum_exp_beta_F += exp_beta_F;
        t_target_ds[i] = exp_beta_F;
      }
      t_target_ds = (1./sum_exp_beta_F) * t_target_ds;

      ofstream file;
      if( c_iter % o_print == 0){
        file.open(("info.iter-"+to_string(c_iter)).c_str());
        for(unsigned i=0; i<t_fes.size(); i++)
          if( c_iter % o_print == 0) file << t_grid[i] << "\t" << bias_grid[i] << "\t" << t_fes[i] << "\t" << t_target_ds[i] << endl;
	file.close();
      }
    }  
    //update parameters
    //nn_opt->step();
 }
}

}
}


