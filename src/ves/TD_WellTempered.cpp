/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2016-2017 The ves-code team
   (see the PEOPLE-VES file at the root of this folder for a list of names)

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
#include "GridIntegrationWeights.h"

#include "tools/Keywords.h"
#include "tools/Grid.h"


namespace PLMD{
namespace ves{

//+PLUMEDOC VES_TARGETDIST WELL_TEMPERED
/*
Well-tempered target distribution (dynamic).

\par Examples

*/
//+ENDPLUMEDOC

class TD_WellTempered: public TargetDistribution {
private:
  double bias_factor_;
public:
  static void registerKeywords(Keywords&);
  explicit TD_WellTempered(const TargetDistributionOptions& to);
  void updateGrid();
  double getValue(const std::vector<double>&) const;
  ~TD_WellTempered(){}
};


VES_REGISTER_TARGET_DISTRIBUTION(TD_WellTempered,"WELL_TEMPERED")


void TD_WellTempered::registerKeywords(Keywords& keys){
  TargetDistribution::registerKeywords(keys);
  keys.add("compulsory","BIASFACTOR","The bias factor to be used for the well tempered distribution");
}


TD_WellTempered::TD_WellTempered( const TargetDistributionOptions& to ):
TargetDistribution(to),
bias_factor_(0.0)
{
  parse("BIASFACTOR",bias_factor_);
  if(bias_factor_<=1.0){
    plumed_merror("WELL_TEMPERED target distribution: the value of the bias factor doesn't make sense, it should be larger than 1.0");
  }
  setDynamic();
  setFesGridNeeded();
  checkRead();
}


double TD_WellTempered::getValue(const std::vector<double>& argument) const {
  plumed_merror("getValue not implemented for TD_WellTempered");
  return 0.0;
}


void TD_WellTempered::updateGrid(){
  double beta_prime = getBeta()/bias_factor_;
  plumed_massert(getFesGridPntr()!=NULL,"the FES grid has to be linked to use TD_WellTempered!");
  std::vector<double> integration_weights = GridIntegrationWeights::getIntegrationWeights(getTargetDistGridPntr());
  double norm = 0.0;
  for(Grid::index_t l=0; l<targetDistGrid().getSize(); l++){
    double value = beta_prime * getFesGridPntr()->getValue(l);
    logTargetDistGrid().setValue(l,value);
    value = exp(-value);
    norm += integration_weights[l]*value;
    targetDistGrid().setValue(l,value);
  }
  targetDistGrid().scaleAllValuesAndDerivatives(1.0/norm);
  logTargetDistGrid().setMinToZero();
}


}
}