#ifdef __PLUMED_HAS_DYNET

#ifndef DYNET_MLP_H
#define DYNET_MLP_H

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;
using namespace dynet;

using namespace std;
using namespace dynet;
/**
 * \ingroup ffbuilders
 * Common activation functions used in multilayer perceptrons
 */
enum Activation {
  SIGMOID, /**< `SIGMOID` : Sigmoid function \f$x\longrightarrow \frac {1} {1+e^{-x}}\f$ */
  TANH, /**< `TANH` : Tanh function \f$x\longrightarrow \frac {1-e^{-2x}} {1+e^{-2x}}\f$ */
  RELU, /**< `RELU` : Rectified linear unit \f$x\longrightarrow \max(0,x)\f$ */
  LINEAR, /**< `LINEAR` : Identity function \f$x\longrightarrow x\f$ */
  SOFTMAX /**< `SOFTMAX` : Softmax function \f$\textbf{x}=(x_i)_{i=1,\dots,n}\longrightarrow \frac {e^{x_i}}{\sum_{j=1}^n e^{x_j} })_{i=1,\dots,n}\f$ */
};

/**
 * \ingroup ffbuilders
 * \struct Layer
 * \brief Simple layer structure
 * \details Contains all parameters defining a layer
 *
 */
struct Layer {
public:
  unsigned input_dim; /**< Input dimension */
  unsigned output_dim; /**< Output dimension */
  Activation activation = LINEAR; /**< Activation function */
  float dropout_rate = 0; /**< Dropout rate */
  /**
   * \brief Build a feed forward layer
   *
   * \param input_dim Input dimension
   * \param output_dim Output dimension
   * \param activation Activation function
   * \param dropout_rate Dropout rate
   */
  Layer(unsigned input_dim, unsigned output_dim, Activation activation, float dropout_rate) :
    input_dim(input_dim),
    output_dim(output_dim),
    activation(activation),
    dropout_rate(dropout_rate) {};
  Layer() {};
};

/**
 * \ingroup ffbuilders
 * \struct MLP
 * \brief Simple multilayer perceptron
 *
 */
struct MLP {
protected:
  // Hyper-parameters
  unsigned LAYERS = 0;

  // Layers
  vector<Layer> layers;
  // Parameters
  vector<vector<Parameter>> params;
  // Expressions 
  vector<vector<Expression>> expr;
 
bool dropout_active = true;

public:
  /**
   * \brief Default constructor
   * \details Dont forget to add layers!
   */
  MLP(ParameterCollection & model) {
    LAYERS = 0;
  }
  /**
   * \brief Returns a Multilayer perceptron
   * \details Creates a feedforward multilayer perceptron based on a list of layer descriptions
   *
   * \param model ParameterCollection to contain parameters
   * \param layers Layers description
   */
  MLP(ParameterCollection& model,
      vector<Layer> layers,
      ComputationGraph& cg) {
    // Verify layers compatibility
    for (unsigned l = 0; l < layers.size() - 1; ++l) {
      if (layers[l].output_dim != layers[l + 1].input_dim)
        throw invalid_argument("Layer dimensions don't match");
    }

    // Register parameters in model
    for (Layer layer : layers) {
      append(model, layer, cg);
    } 
  }

  /**
   * \brief Append a layer at the end of the network
   * \details [long description]
   *
   * \param model [description]
   * \param layer [description]
   */
  void append(ParameterCollection& model, Layer layer, ComputationGraph& cg) {
    // Check compatibility
    if (LAYERS > 0)
      if (layers[LAYERS - 1].output_dim != layer.input_dim)
        throw invalid_argument("Layer dimensions don't match");

    // Add to layers
    layers.push_back(layer);
    LAYERS++;
    // Register parameters
    Parameter W = model.add_parameters({layer.output_dim, layer.input_dim});
    Parameter b = model.add_parameters({layer.output_dim});
    params.push_back({W, b});
    // Initialize parameters in computation graph
    Expression W2 = parameter(cg, params[LAYERS-1][0]);
    Expression b2 = parameter(cg, params[LAYERS-1][1]);
    expr.push_back({W2,b2});
  }

  unsigned Layers ( ) const { return LAYERS; }

  Layer getLayer ( unsigned id ) {
    if (id >= LAYERS) 
      throw invalid_argument("Unable to recover the requested layer");    
    // Get id-th layer
    return layers[id];
  }

  /**
   * \brief Run the MLP on an input vector/batch
   *
   * \param x Input expression (vector or batch)
   * \param cg Computation graph
   *
   * \return [description]
   */
  Expression run(Expression x) {
    // Expression for the current hidden state
    Expression h_cur = x;
    for (unsigned l = 0; l < LAYERS; ++l) {
      // Apply affine transform
      Expression a = affine_transform({expr[l][1], expr[l][0], h_cur});
      // Apply activation function
      Expression h = activate(a, layers[l].activation);
      // Assign to current state 
      h_cur = h;
    }
    return h_cur;
  }

  vector<vector<real> > get_grad_w() {
    vector<vector<real>> g;
    for (unsigned l = 0; l < LAYERS; ++l)
      g.push_back( as_vector ( expr[l][0].gradient() ) ); 
   
    return g;
}

  vector<vector<real> > get_grad_b() {
    vector<vector<real>> g;
    for (unsigned l = 0; l < LAYERS; ++l)
      g.push_back( as_vector ( expr[l][1].gradient() ) ); 
   
    return g;
  }

void set_grad ( ComputationGraph& cg, vector<vector<real>> grad_w, vector<vector<real>> grad_b ) {

    for (unsigned l = 0; l < LAYERS; ++l){
      Expression new_gw = input(cg, { grad_w[l].size() }, &grad_w[l]);		//cp the gradients into an expression 
      Expression new_gb = input(cg, { grad_b[l].size() }, &grad_b[l]);		//cp the gradients into an expression
      Expression new_gw_r = reshape(new_gw, {layers[l].output_dim,layers[l].input_dim}); //reshape from vector shape to a matrix
      Expression loss_w = cmult(expr[l][0],new_gw_r);				//element-wise multiplication
      Expression loss_b = cmult(expr[l][1],new_gb);				//element-wise multiplication
      Expression loss = sum_elems(loss_w) + sum_elems(loss_b);			//sum elements
      cg.backward(loss);
    }
  }

  /**
   * \brief Enable dropout
   * \details This is supposed to be used during training or during testing if you want to sample outputs using montecarlo
   */
  void enable_dropout() {
    dropout_active = true;
  }

  /**
   * \brief Disable dropout
   * \details Do this during testing if you want a deterministic network
   */
  void disable_dropout() {
    dropout_active = false;
  }

  /**
   * \brief Check wether dropout is enabled or not
   *
   * \return Dropout state
   */
  bool is_dropout_enabled() {
    return dropout_active;
  }

private:
  inline Expression activate(Expression h, Activation f) {
    switch (f) {
    case LINEAR:
      return h;
      break;
    case RELU:
      return rectify(h);
      break;
    case SIGMOID:
      return logistic(h);
      break;
    case TANH:
      return tanh(h);
      break;
    case SOFTMAX:
      return softmax(h);
      break;
    default:
      throw invalid_argument("Unknown activation function");
      break;
    }
  }
};

Trainer* new_trainer(const std::string& algorithm,ParameterCollection& pc,std::string& fullname)
{
        if(algorithm=="SimpleSGD"||algorithm=="simpleSGD"||algorithm=="simplesgd"||algorithm=="SGD"||algorithm=="sgd")
        {
                fullname="Stochastic gradient descent";
                Trainer *trainer = new SimpleSGDTrainer(pc);
                return trainer;
        }
        if(algorithm=="CyclicalSGD"||algorithm=="cyclicalSGD"||algorithm=="cyclicalsgd"||algorithm=="CSGD"||algorithm=="csgd")
        {
                fullname="Cyclical learning rate SGD";
                Trainer *trainer = new CyclicalSGDTrainer(pc);
                return trainer;
        }
        if(algorithm=="MomentumSGD"||algorithm=="momentumSGD"||algorithm=="momentumSGD"||algorithm=="MSGD"||algorithm=="msgd")
        {
                fullname="SGD with momentum";
                Trainer *trainer = new MomentumSGDTrainer(pc);
                return trainer;
        }
        if(algorithm=="Adagrad"||algorithm=="adagrad"||algorithm=="adag"||algorithm=="ADAG")
        {
                fullname="Adagrad optimizer";
                Trainer *trainer = new AdagradTrainer(pc);
                return trainer;
        }
        if(algorithm=="Adadelta"||algorithm=="adadelta"||algorithm=="AdaDelta"||algorithm=="AdaD"||algorithm=="adad"||algorithm=="ADAD")
        {
                fullname="AdaDelta optimizer";
                Trainer *trainer = new AdadeltaTrainer(pc);
                return trainer;
        }
        if(algorithm=="RMSProp"||algorithm=="rmsprop"||algorithm=="rmsp"||algorithm=="RMSP")
        {
                fullname="RMSProp optimizer";
                Trainer *trainer = new RMSPropTrainer(pc);
                return trainer;
        }
        if(algorithm=="Adam"||algorithm=="adam"||algorithm=="ADAM")
        {
                fullname="Adam optimizer";
                Trainer *trainer = new AdamTrainer(pc);
                return trainer;
        }
        if(algorithm=="AMSGrad"||algorithm=="Amsgrad"||algorithm=="Amsg"||algorithm=="amsg")
        {
                fullname="AMSGrad optimizer";
                Trainer *trainer = new AmsgradTrainer(pc);
                return trainer;
        }
        return NULL;
}


#endif

#endif
