/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Copyright (c) 2021 of Luigi Bonati.

The pytorch module is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The pytorch module is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

/* Use the following preprocessor directive only if plumed 
is configured to look for libtorch */
#ifdef __PLUMED_HAS_LIBTORCH

#include "core/PlumedMain.h"
#include "function/Function.h"
#include "function/ActionRegister.h"

#include <torch/torch.h>
#include <torch/script.h>

#include <cmath>

using namespace std;

namespace PLMD {
namespace function {
namespace mlcvs {

//+PLUMEDOC MLCVS_FUNCTION PYTORCH_MODEL_TEST
/*
Load a Pytorch model compiled with TorchScript. The derivatives are set using native backpropagation in Pytorch.

\par Examples
Define a model that takes as inputs two distances d1 and d2.

\plumedfile
model: PYTORCH_MODEL FILE=model.ptc ARG=d1,d2
\endplumedfile

The N nodes of the neural network are saved as "model.node-0", "model.node-1", ..., "model.node-(N-1)".

*/
//+ENDPLUMEDOC


class PytorchModelTest :
  public Function
{
  unsigned _n_in;
  unsigned _n_out;
  torch::jit::script::Module _model;

public:
  explicit PytorchModelTest(const ActionOptions&);
  void calculate();
  static void registerKeywords(Keywords& keys);

  std::vector<float> tensor_to_vector(const torch::Tensor& x);
};

PLUMED_REGISTER_ACTION(PytorchModelTest,"PYTORCH_MODEL_TEST")

void PytorchModelTest::registerKeywords(Keywords& keys) {
  Function::registerKeywords(keys);
  keys.use("ARG");
  keys.add("optional","FILE","filename of the trained model"); 
  keys.addOutputComponent("node", "default", "NN outputs"); 
}

// Auxiliary function to transform torch tensor in std vector
std::vector<float> PytorchModelTest::tensor_to_vector(const torch::Tensor& x) {
    return std::vector<float>(x.data_ptr<float>(), x.data_ptr<float>() + x.numel());
}

PytorchModelTest::PytorchModelTest(const ActionOptions&ao):
  Action(ao),
  Function(ao)
{ //print pytorch version

  //number of inputs of the model
  _n_in=getNumberOfArguments();

  //parse model name
  std::string fname="model.ptc";
  parse("FILE",fname); 
 
  //deserialize the model from file
  try {
    _model = torch::jit::load(fname);
  }
  catch (const c10::Error& e) {
    error("Cannot load Pytorch model. Check that the model is present and that the version of Pytorch is compatible with the Libtorch linked to PLUMED.");    
  }

  checkRead();

  //check the dimension of the output
  log.printf("Checking output dimension:\n");
  std::vector<float> input_test (_n_in);
  torch::Tensor single_input = torch::tensor(input_test).view({1,_n_in});  
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back( single_input );
  torch::Tensor output = _model.forward( inputs ).toTensor(); 
  vector<float> cvs = this->tensor_to_vector (output);
  _n_out=cvs.size();

  //create components
  for(unsigned j=0; j<_n_out; j++){
    string name_comp = "node-"+std::to_string(j);
    addComponentWithDerivatives( name_comp );
    componentIsNotPeriodic( name_comp );
  }
 
  //print log
  //log.printf("Pytorch Model Loaded: %s \n",fname);
  log.printf("Number of input: %d \n",_n_in); 
  log.printf("Number of outputs: %d \n",_n_out); 
  log.printf("  Bibliography: ");
  log<<plumed.cite("L. Bonati, V. Rizzi and M. Parrinello, J. Phys. Chem. Lett. 11, 2998-3004 (2020)");
  log.printf("\n");

}

void PytorchModelTest::calculate() {

  //retrieve arguments
  vector<float> current_S(_n_in);
  for(unsigned i=0; i<_n_in; i++)
    current_S[i]=getArgument(i);
  //convert to tensor
  torch::Tensor input_S = torch::tensor(current_S).view({1,_n_in});
  input_S.set_requires_grad(true);
  //convert to Ivalue
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back( input_S );
  //calculate output
  torch::Tensor output = _model.forward( inputs ).toTensor();  
  //set CV values
  vector<float> cvs = this->tensor_to_vector (output);
  for(unsigned j=0; j<_n_out; j++){
    string name_comp = "node-"+std::to_string(j);
    getPntrToComponent(name_comp)->set(cvs[j]);
  }
  //derivatives
  for(unsigned j=0; j<_n_out; j++){
    // expand dim to have shape (1,_n_out)
    int batch_size = 1; 
    auto grad_output = torch::ones({1}).expand({batch_size, 1}); 
    // calculate derivatives with automatic differentiation
    auto gradient = torch::autograd::grad({output.slice(/*dim=*/1, /*start=*/j, /*end=*/j+1)},
                                          {input_S},
                                          /*grad_outputs=*/{grad_output},
                                          /*retain_graph=*/true,
                                          /*create_graph=*/false);
    // add dimension
    auto grad = gradient[0].unsqueeze(/*dim=*/1);
    //convert to vector
    vector<float> der = this->tensor_to_vector ( grad );

    string name_comp = "node-"+std::to_string(j);
    //set derivatives of component j
    for(unsigned i=0; i<_n_in; i++)
      setDerivative( getPntrToComponent(name_comp) ,i, der[i] );
    //reset gradients
    //input_S.grad().zero_();
  }
}
}
}
}

#endif
