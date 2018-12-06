/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2016 The plumed team
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
#include "tools/NeighborList.h"
#include "tools/Communicator.h"
#include "tools/Tools.h"
#include "tools/SwitchingFunction.h"

#include <string>

using namespace std;

namespace PLMD{
namespace colvar{

//+PLUMEDOC COLVAR TETRA
/*

Mean Tetrahedral Order Parameter (cfr. multicolvar/Tetrabrick.cpp for further references)

\f[
s = 1 - \frac{3}{32} \sum_{i=0} ^{N-1} \sum_{j=i+1} ^{N} (\cos(\theta_{ikj}+1/3))^2
\f]


\par Examples

Usage:
\verbatim
TETRA ...
GROUPA=1-216
LABEL=t
SWITCH={RATIONAL D_0=2.5 R_0=0.5 D_MAX=3.5}
NLIST
NL_CUTOFF=5
NL_STRIDE=10
... TETRA
\endverbatim

*/
//+ENDPLUMEDOC

class Tetra : public Colvar {
  bool pbc;
  bool serial;
  NeighborList *nl;
  bool invalidateList;
  bool firsttime;
  vector<AtomNumber> list_a;

  double rcut2;
  SwitchingFunction switchingFunction;

public:
  explicit Tetra(const ActionOptions&);
  ~Tetra();
// active methods:
  virtual void calculate();
  virtual void prepare();
  static void registerKeywords( Keywords& keys );

};

PLUMED_REGISTER_ACTION(Tetra,"TETRA")

void Tetra::registerKeywords( Keywords& keys ){

  Colvar::registerKeywords(keys);
  keys.addFlag("SERIAL",false,"Perform the calculation in serial - for debug purpose");
  keys.addFlag("NLIST",false,"Use a neighbour list to speed up the calculation");
  keys.addFlag("PAIR",false,"Pair only 1st element of the 1st group with 1st element in the second, etc");
  keys.add("optional","NL_CUTOFF","The cutoff for the neighbour list");
  keys.add("optional","NL_STRIDE","The frequency with which we are updating the atoms in the neighbour list");

  keys.add("atoms","GROUPA","First list of atoms");
  keys.add("atoms","GROUPB","Second list of atoms (if empty, N*(N-1)/2 pairs in GROUPA are counted)");

  keys.add("compulsory","NN","6","The n parameter of the switching function ");
  keys.add("compulsory","MM","0","The m parameter of the switching function; 0 implies 2*NN");
  keys.add("compulsory","D_0","0.0","The d_0 parameter of the switching function");
  keys.add("compulsory","R_0","The r_0 parameter of the switching function");
  keys.add("optional","SWITCH","This keyword is used if you want to employ an alternative to the continuous swiching function defined above. "
                               "The following provides information on the \\ref switchingfunction that are available. "
                               "When this keyword is present you no longer need the NN, MM, D_0 and R_0 keywords.");

}

Tetra::Tetra(const ActionOptions&ao):
PLUMED_COLVAR_INIT(ao),
pbc(true),
serial(false),
invalidateList(true),
firsttime(true)
{

  parseFlag("SERIAL",serial);

  vector<AtomNumber> ga_lista,gb_lista;
  parseAtomList("GROUPA",ga_lista);
  parseAtomList("GROUPB",gb_lista);

  list_a=ga_lista;

  bool nopbc=!pbc;
  parseFlag("NOPBC",nopbc);
  pbc=!nopbc;

// pair stuff
  bool dopair=false;
  parseFlag("PAIR",dopair);

// neighbor list stuff
  bool doneigh=false;
  double nl_cut=0.0;
  int nl_st=0;
  parseFlag("NLIST",doneigh);
  if(doneigh){
   parse("NL_CUTOFF",nl_cut);
   if(nl_cut<=0.0) error("NL_CUTOFF should be explicitly specified and positive");
   parse("NL_STRIDE",nl_st);
   if(nl_st<=0) error("NL_STRIDE should be explicitly specified and positive");
  }

  addValueWithDerivatives(); setNotPeriodic();
  if(gb_lista.size()>0){
    if(doneigh)  nl= new NeighborList(ga_lista,gb_lista,dopair,pbc,getPbc(),nl_cut,nl_st);
    else         nl= new NeighborList(ga_lista,gb_lista,dopair,pbc,getPbc());
  } else {
    if(doneigh)  nl= new NeighborList(ga_lista,pbc,getPbc(),nl_cut,nl_st);
    else         nl= new NeighborList(ga_lista,pbc,getPbc());
  }

  requestAtoms(nl->getFullAtomList());

  log.printf("  between two groups of %u and %u atoms\n",static_cast<unsigned>(ga_lista.size()),static_cast<unsigned>(gb_lista.size()));
  log.printf("  first group:\n");
  for(unsigned int i=0;i<ga_lista.size();++i){
   if ( (i+1) % 25 == 0 ) log.printf("  \n");
   log.printf("  %d", ga_lista[i].serial());
  }
  log.printf("  \n  second group:\n");
  for(unsigned int i=0;i<gb_lista.size();++i){
   if ( (i+1) % 25 == 0 ) log.printf("  \n");
   log.printf("  %d", gb_lista[i].serial());
  }
  log.printf("  \n");
  if(pbc) log.printf("  using periodic boundary conditions\n");
  else    log.printf("  without periodic boundary conditions\n");
  if(dopair) log.printf("  with PAIR option\n");
  if(doneigh){
   log.printf("  using neighbor lists with\n");
   log.printf("  update every %d steps and cutoff %f\n",nl_st,nl_cut);
  }

  // Read in the switching function
  std::string sw, errors; parse("SWITCH",sw);
  if(sw.length()>0){
     switchingFunction.set(sw,errors);
     if( errors.length()!=0 ) error("problem reading SWITCH keyword : " + errors );
  } else {
     double r_0=-1.0, d_0; int nn, mm;
     parse("NN",nn); parse("MM",mm);
     parse("R_0",r_0); parse("D_0",d_0);
     if( r_0<0.0 ) error("you must set a value for R_0");
     switchingFunction.set(nn,mm,r_0,d_0);
  }
  log.printf("  Tetrahedral order parameter calculated computing the angles with the atoms within %s\n",( switchingFunction.description() ).c_str() );

  rcut2 = switchingFunction.get_dmax()*switchingFunction.get_dmax();

  if(doneigh){
    if(nl_cut < switchingFunction.get_dmax() ) error("NL_CUTOFF should be larger than D_MAX");
  }

  checkRead();
}

Tetra::~Tetra(){
  delete nl;
}

void Tetra::prepare(){
  if(nl->getStride()>0){
    if(firsttime || (getStep()%nl->getStride()==0)){
      requestAtoms(nl->getFullAtomList());
      invalidateList=true;
      firsttime=false;
    }else{
      requestAtoms(nl->getReducedAtomList());
      invalidateList=false;
      if(getExchangeStep()) error("Neighbor lists should be updated on exchange steps - choose a NL_STRIDE which divides the exchange stride!");
    }
    if(getExchangeStep()) firsttime=true;
  }
}

// calculator
void Tetra::calculate()
{
  double NumberOfAtoms=list_a.size();
  //mpi vectors (or matrix) for intermediate computations
  vector<double> tetra(NumberOfAtoms);
  Matrix<Vector> deriv(NumberOfAtoms,NumberOfAtoms);
  vector<Tensor> virial(NumberOfAtoms);
  // Define output quantities
  double tetra_mean=0.;
  vector<Vector> deriv_sum(getNumberOfAtoms());
  Tensor virial_sum;
  // Define temp quantities
  unsigned int index_i, index_j;
  double d2i, d2j;								//square distances
  double di_mod, dj_mod;						//modulo of distances
  double di_inv, dj_inv;						//inverse of distances
  double cos, cos2, versors; 							//cosine term, and squared version, and scalar product between versors
  double sw_i, sw_j, df_i, df_j;				//switch functions and derivatives
  double prefactor;							//prefactor of the derivative of the cosine part
  Vector der_sw_i, der_sw_j;					//switch derivative parts
  Vector der_i, der_j;							//derivatives i and j

  // Setup neighbor list and parallelization
  if(nl->getStride()>0 && invalidateList){
    nl->update(getPositions());
  }
  unsigned stride=comm.Get_size();
  unsigned rank=comm.Get_rank();
  if(serial){
    stride=1;
    rank=0;
  }else{
    stride=comm.Get_size();
    rank=comm.Get_rank();
  }

  //loop over atoms
  for(unsigned int n=rank;n<NumberOfAtoms;n+=stride) {
    //get the neighbors of atom[k] within list_a
    unsigned int index_k = list_a[n].index();
    vector<unsigned> neigh_k = nl->getNeighbors(index_k);
    Vector rk = getPosition(index_k);
    Vector di, dj;
    //(1)loop over neighbors
    for(unsigned i=0;i<(neigh_k.size()-1);++i){
      //get relative positions between atom k and neighbors i and j
      index_i=neigh_k[i];
      if(pbc){ di=pbcDistance(rk,getPosition(index_i));
      } else { di=delta(rk,getPosition(index_i)); }
      if(getAbsoluteIndex(index_k)==getAbsoluteIndex(index_i)) continue;
      //check if the distance is within the cutoff
      if ( (d2i=di[0]*di[0])<rcut2 && (d2i+=di[1]*di[1])<rcut2 && (d2i+=di[2]*di[2])<rcut2 ) {
	       //(2)second loop over neighbors
        for(unsigned j=i+1;j<neigh_k.size();++j){
          index_j=neigh_k[j];
          if(pbc){ dj=pbcDistance(rk,getPosition(index_j));
          } else { dj=delta(rk,getPosition(index_j)); }
          if(getAbsoluteIndex(index_k)==getAbsoluteIndex(index_j)) continue;
          if ( (d2j=dj[0]*dj[0])<rcut2 && (d2j+=dj[1]*dj[1])<rcut2 && (d2j+=dj[2]*dj[2])<rcut2) {
            // -- (1) COMPUTE --
            // useful terms
            di_mod=sqrt(d2i);
            dj_mod=sqrt(d2j);
            di_inv=1./di_mod;
            dj_inv=1./dj_mod;
            // cosine term
            versors = dotProduct( di,dj )*di_inv*dj_inv;
            cos = versors + 1./3.;
            cos2 = cos*cos;
            // compute switching functions (derivatives return der/dist)
            sw_i = switchingFunction.calculateSqr( d2i, df_i );
            sw_j = switchingFunction.calculateSqr( d2j, df_j );
            // order parameter (relative to ikj triplet)
            tetra[n] += cos2*sw_i*sw_j;

            if (!doNotCalculateDerivatives()) {
              // -- (2) DERIVATE --
              // useful terms
              der_sw_i = di * df_i * sw_j * cos2;		// sw part relative to i derivative
              der_sw_j = sw_i * dj * df_j * cos2;		// sw part relative to j derivative
              prefactor = 2*sw_i*sw_j*cos;			    // der prefactor

              der_i = der_sw_i + prefactor * ( di_inv*dj_inv * dj - (versors/d2i)* di);
              der_j = der_sw_j + prefactor * ( di_inv*dj_inv * di - (versors/d2j)* dj);

              deriv[n][index_i] += der_i;
              deriv[n][index_j] += der_j;
              deriv[n][index_k] -= der_i + der_j;
              // -- (3) VIRIAL --
              virial[n] -= Tensor(di,-3./8.*der_i)+Tensor(dj,-3./8.*der_j);
            }
          }
        }
      }
    }
  }

  if(!serial){
    comm.Sum(&tetra[0],NumberOfAtoms);
    comm.Sum(&deriv[0][0],NumberOfAtoms*NumberOfAtoms);
    comm.Sum(&virial[0],NumberOfAtoms);
  }

  for(unsigned int k=0;k<NumberOfAtoms;++k){
    tetra_mean += tetra[k];
    virial_sum += virial[k];
    for(unsigned int l=0;l<NumberOfAtoms;++l) deriv_sum[k] += deriv[l][k];
  }

  // mean tetrahedral order parameter
  tetra_mean = 1 - 3./8.*(tetra_mean/NumberOfAtoms);

  // Assign output quantities
  for(unsigned k=0;k<NumberOfAtoms;++k) setAtomsDerivatives(k,-3./8. /NumberOfAtoms *deriv_sum[k]);
  setValue           (tetra_mean);
  setBoxDerivatives  (virial_sum/NumberOfAtoms);
}

}
}
