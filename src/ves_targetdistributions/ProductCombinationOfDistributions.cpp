/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2015-2016 The ves-code team
   (see the PEOPLE-VES file at the root of the distribution for a list of names)

   See http://www.ves-code.org for more information.

   This file is part of ves-code, version 1.

   ves-code is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ves-code is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with ves-code.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "TargetDistribution.h"
#include "TargetDistributionRegister.h"

#include "tools/Keywords.h"
#include "tools/Grid.h"

namespace PLMD {

//+PLUMEDOC INTERNAL GAUSSIAN
/*
  Gaussian target distribution
*/
//+ENDPLUMEDOC

class ProductCombinationOfDistributions: public TargetDistribution {
private:
  std::vector<TargetDistribution*> distribution_pntrs_;
  std::vector<Grid*> grid_pntrs_;
  unsigned int ndist_;
  void setupAdditionalGrids(const std::vector<Value*>&, const std::vector<std::string>&, const std::vector<std::string>&, const std::vector<unsigned int>&);
public:
  static void registerKeywords(Keywords&);
  explicit ProductCombinationOfDistributions(const TargetDistributionOptions& to);
  void updateGrid();
  double getValue(const std::vector<double>&) const;
  ~ProductCombinationOfDistributions();
  //
  void linkVesBias(bias::VesBias*);
  void linkAction(Action*);
  void linkBiasGrid(Grid*);
  void linkBiasWithoutCutoffGrid(Grid*);
  void linkFesGrid(Grid*);
};


VES_REGISTER_TARGET_DISTRIBUTION(ProductCombinationOfDistributions,"PRODUCT_COMBINATION")


void ProductCombinationOfDistributions::registerKeywords(Keywords& keys){
  TargetDistribution::registerKeywords(keys);
  keys.add("numbered","DIST_ARG","The one dimensional target distributions to be used in the product combination for each argument");
  keys.addFlag("IGNORE_NORMALIZATION",false,"If the check on the normalization of the distributions should be ignored. Be warned that this can lead to non-normalized distributions and most likely stange results.");
}


ProductCombinationOfDistributions::ProductCombinationOfDistributions( const TargetDistributionOptions& to ):
TargetDistribution(to),
distribution_pntrs_(0),
grid_pntrs_(0),
ndist_(0)
{
  bool normalized = true;
  for(unsigned int i=1;; i++) {
    std::string keywords;
    if(!parseNumbered("DIST_ARG",i,keywords) ){break;}
    std::vector<std::string> words = Tools::getWords(keywords);
    TargetDistribution* dist_pntr_tmp = targetDistributionRegister().create( (words) );
    if(dist_pntr_tmp->isDynamic()){setDynamic();}
    if(dist_pntr_tmp->fesGridNeeded()){setFesGridNeeded();}
    if(dist_pntr_tmp->biasGridNeeded()){setBiasGridNeeded();}
    if(!dist_pntr_tmp->isNormalized()){normalized = false;}
    distribution_pntrs_.push_back(dist_pntr_tmp);
  }
  ndist_ = distribution_pntrs_.size();
  grid_pntrs_.assign(ndist_,NULL);
  setDimension(ndist_);

  bool ignore_normalization_check = false;
  parseFlag("IGNORE_NORMALIZATION",ignore_normalization_check);
  if(normalized){
    setNormalized();
  }
  else{
    if(!ignore_normalization_check){
      plumed_merror("PRODUCT_COMBINATION: one of the one dimensional target distribution is not normalized so the product combination will not be normalized. Use the keyword IGNORE_NORMALIZATION to ignore this check and run regardless.");
    }
    setNotNormalized();
  }

  checkRead();
}


ProductCombinationOfDistributions::~ProductCombinationOfDistributions(){
  for(unsigned int i=0; i<ndist_; i++){
    delete distribution_pntrs_[i];
  }
}


double ProductCombinationOfDistributions::getValue(const std::vector<double>& argument) const {
  plumed_merror("getValue not implemented for ProductCombinationOfDistributions");
  return 0.0;
}


void ProductCombinationOfDistributions::setupAdditionalGrids(const std::vector<Value*>& arguments, const std::vector<std::string>& min, const std::vector<std::string>& max, const std::vector<unsigned int>& nbins) {
  for(unsigned int i=0; i<ndist_; i++){
    std::vector<Value*> arg1d(1);
    std::vector<std::string> min1d(1);
    std::vector<std::string> max1d(1);
    std::vector<unsigned int> nbins1d(1);
    arg1d[0]=arguments[i];
    min1d[0]=min[i];
    max1d[0]=max[i];
    nbins1d[0]=nbins[i];
    distribution_pntrs_[i]->setupGrids(arg1d,min1d,max1d,nbins1d);
    grid_pntrs_[i]=distribution_pntrs_[i]->getTargetDistGridPntr();
    if(distribution_pntrs_[i]->getDimension()!=1 || grid_pntrs_[i]->getDimension()!=1){
      plumed_merror("Error in PRODUCT_COMBINATION: all target distribution need to be one dimensional");
    }
  }
}


void ProductCombinationOfDistributions::updateGrid(){
  for(unsigned int i=0; i<ndist_; i++){
    distribution_pntrs_[i]->update();
  }
  for(Grid::index_t l=0; l<targetDistGrid().getSize(); l++){
    std::vector<unsigned int> indices = targetDistGrid().getIndices(l);
    double value = 1.0;
    for(unsigned int i=0; i<ndist_; i++){
      value *= grid_pntrs_[i]->getValue(indices[i]);
    }
    targetDistGrid().setValue(l,value);
    logTargetDistGrid().setValue(l,-std::log(value));
  }
  logTargetDistGrid().setMinToZero();
}


void ProductCombinationOfDistributions::linkVesBias(bias::VesBias* vesbias_pntr_in){
  TargetDistribution::linkVesBias(vesbias_pntr_in);
  for(unsigned int i=0; i<ndist_; i++){
    distribution_pntrs_[i]->linkVesBias(vesbias_pntr_in);
  }
}


void ProductCombinationOfDistributions::linkAction(Action* action_pntr_in){
  TargetDistribution::linkAction(action_pntr_in);
  for(unsigned int i=0; i<ndist_; i++){
    distribution_pntrs_[i]->linkAction(action_pntr_in);
  }
}


void ProductCombinationOfDistributions::linkBiasGrid(Grid* bias_grid_pntr_in){
  TargetDistribution::linkBiasGrid(bias_grid_pntr_in);
  for(unsigned int i=0; i<ndist_; i++){
    distribution_pntrs_[i]->linkBiasGrid(bias_grid_pntr_in);
  }
}


void ProductCombinationOfDistributions::linkBiasWithoutCutoffGrid(Grid* bias_withoutcutoff_grid_pntr_in){
  TargetDistribution::linkBiasWithoutCutoffGrid(bias_withoutcutoff_grid_pntr_in);
  for(unsigned int i=0; i<ndist_; i++){
    distribution_pntrs_[i]->linkBiasWithoutCutoffGrid(bias_withoutcutoff_grid_pntr_in);
  }
}


void ProductCombinationOfDistributions::linkFesGrid(Grid* fes_grid_pntr_in){
  TargetDistribution::linkFesGrid(fes_grid_pntr_in);
  for(unsigned int i=0; i<ndist_; i++){
    distribution_pntrs_[i]->linkFesGrid(fes_grid_pntr_in);
  }
}


}
