/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2018 The plumed team
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
#include "Colvar.h"
#include "ActionRegister.h"
#include "tools/Pbc.h"

#include <string>
#include <cmath>

#include <numeric> 
#include <sstream>
#include <iterator>

using namespace std;

namespace PLMD {
namespace colvar {

//+PLUMEDOC COLVAR MultiDistances
/*
Calculate the MultiDistances between a pair of atoms.

By default the MultiDistances is computed taking into account periodic
boundary conditions. This behavior can be changed with the NOPBC flag.
Moreover, single components in cartesian space (x,y, and z, with COMPONENTS)
or single components projected to the three lattice vectors (a,b, and c, with SCALED_COMPONENTS)
can be also computed.

Notice that Cartesian components will not have the proper periodicity!
If you have to study e.g. the permeation of a molecule across a membrane,
better to use SCALED_COMPONENTS.

\par Examples

The following input tells plumed to print the MultiDistances between atoms 3 and 5,
the MultiDistances between atoms 2 and 4 and the x component of the MultiDistances between atoms 2 and 4.
\plumedfile
d1:  MultiDistances ATOMS=3,5
d2:  MultiDistances ATOMS=2,4
d2c: MultiDistances ATOMS=2,4 COMPONENTS
PRINT ARG=d1,d2,d2c.x
\endplumedfile

The following input computes the end-to-end MultiDistances for a polymer
of 100 atoms and keeps it at a value around 5.
\plumedfile
WHOLEMOLECULES ENTITY0=1-100
e2e: MultiDistances ATOMS=1,100 NOPBC
RESTRAINT ARG=e2e KAPPA=1 AT=5
\endplumedfile

Notice that NOPBC is used
to be sure that if the end-to-end MultiDistances is larger than half the simulation
box the MultiDistances is compute properly. Also notice that, since many MD
codes break molecules across cell boundary, it might be necessary to
use the \ref WHOLEMOLECULES keyword (also notice that it should be
_before_ MultiDistances). The list of atoms provided to \ref WHOLEMOLECULES
here contains all the atoms between 1 and 100. Strictly speaking, this
is not necessary. If you know for sure that atoms with difference in
the index say equal to 10 are _not_ going to be farther than half cell
you can e.g. use
\plumedfile
WHOLEMOLECULES ENTITY0=1,10,20,30,40,50,60,70,80,90,100
e2e: MultiDistances ATOMS=1,100 NOPBC
RESTRAINT ARG=e2e KAPPA=1 AT=5
\endplumedfile
Just be sure that the ordered list provide to \ref WHOLEMOLECULES has the following
properties:
- Consecutive atoms should be closer than half-cell throughout the entire simulation.
- Atoms required later for the MultiDistances (e.g. 1 and 100) should be included in the list

The following example shows how to take into account periodicity e.g.
in z-component of a MultiDistances
\plumedfile
# this is a center of mass of a large group
c: COM ATOMS=1-100
# this is the MultiDistances between atom 101 and the group
d: MultiDistances ATOMS=c,101 COMPONENTS
# this makes a new variable, dd, equal to d and periodic, with domain -10,10
# this is the right choise if e.g. the cell is orthorombic and its size in
# z direction is 20.
dz: COMBINE ARG=d.z PERIODIC=-10,10
# metadynamics on dd
METAD ARG=dz SIGMA=0.1 HEIGHT=0.1 PACE=200
\endplumedfile

Using SCALED_COMPONENTS this problem should not arise because they are always periodic
with domain (-0.5,+0.5).




*/
//+ENDPLUMEDOC

class MultiDistances : public Colvar {
  bool pbc;
  unsigned num_atoms, num_dist;

public:
  static void registerKeywords( Keywords& keys );
  explicit MultiDistances(const ActionOptions&);
// active methods:
  virtual void calculate();
};

PLUMED_REGISTER_ACTION(MultiDistances,"MULTI_DISTANCES")

void MultiDistances::registerKeywords( Keywords& keys ) {
  Colvar::registerKeywords( keys );
  keys.add("atoms","ATOMS","the pair of atom that we are calculating the MultiDistances between");
  keys.addFlag("COMPONENTS",true,"calculate the x, y and z components of the MultiDistances separately and store them as label.x, label.y and label.z");
}

MultiDistances::MultiDistances(const ActionOptions&ao):
  PLUMED_COLVAR_INIT(ao),
  pbc(true)
{
  vector<AtomNumber> atoms;
  parseAtomList("ATOMS",atoms);
  num_atoms = atoms.size();
  num_dist = (num_atoms*(num_atoms-1))/2;

  bool nopbc=!pbc;
  parseFlag("NOPBC",nopbc);
  pbc=!nopbc;
  checkRead();

  //log.printf("  between atoms %d %d\n",atoms[0].serial(),atoms[1].serial());
  if(pbc) log.printf("  using periodic boundary conditions\n");
  else    log.printf("  without periodic boundary conditions\n");

     for(unsigned i=0; i<num_dist; i++)
     {
        char label[3];
        sprintf(label, "%d", i);
        addComponentWithDerivatives(label); componentIsNotPeriodic(label);
     }

  requestAtoms(atoms);
}


// calculator
void MultiDistances::calculate() {

  if(pbc) makeWhole();

  vector<Vector> pos=getPositions();

  Vector distance;
  double invdist, value;

  int ind = 0;
  for(unsigned i=0; i<num_atoms; i++)
  {
     for(unsigned j=i+1; j<num_atoms; j++)
     {
        distance=delta(getPosition(i),getPosition(j));
        value=distance.modulo();
        invdist=1.0/value;

        char label[3];
        sprintf(label, "%d", ind);
	Value* val=getPntrToComponent(label);
	
	setAtomsDerivatives (val,i,-invdist*distance);
        setAtomsDerivatives (val,j,invdist*distance);

        setBoxDerivativesNoPbc(val);
        val->set(value);

        ind++;
     }
  }

}

}
}
