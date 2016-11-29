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

#ifndef __PLUMED_ves_TargetDistModifer_h
#define __PLUMED_ves_TargetDistModifer_h

namespace PLMD{
namespace ves{

class TargetDistModifer{
public:
  virtual double getModifedTargetDistValue(const double targetdist_value, const std::vector<double>& cv_values) const = 0;
  virtual ~TargetDistModifer(){}
};

class WellTemperedModifer:public TargetDistModifer{
private:
  double invbiasf_;
public:
  explicit WellTemperedModifer(double biasfactor):invbiasf_(1.0/biasfactor){}
  double getModifedTargetDistValue(const double targetdist_value, const std::vector<double>& cv_values) const {
    return std::pow(targetdist_value,invbiasf_);
  }
};




}
}

#endif
