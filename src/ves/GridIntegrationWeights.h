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

#ifndef __PLUMED_ves_GridIntegrationWeights_h
#define __PLUMED_ves_GridIntegrationWeights_h

#include <vector>
#include <string>


namespace PLMD{


class Grid;


namespace ves{

class GridIntegrationWeights{
private:
  static std::vector<double> getOneDimensionalTrapezoidalWeights(const unsigned int, const double, const bool periodic=false);
public:
  static std::vector<double> getIntegrationWeights(const Grid*, const std::string& fname_weights_grid="", const std::string& weights_type="trapezoidal");
};



}
}

#endif
