/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2016,2017 The plumed team
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
#include "core/PlumedMain.h"
#include "tools/Communicator.h"

#include <cmath>
#include <sstream> //std::ostringstream

using namespace std;

namespace PLMD {
namespace colvar {

//+PLUMEDOC COLVAR STRUCTURE_FACTOR_ISO
/*
Use as CVs the instantaneous isotropic structure factor:
\f[
 S_k = 1+2/N \sum_{i<j} sin(k r_{ij})/(k r_{ij})
\f]
where \f$r_{ij}\f$ is the distance between atom \f$i\f$ and \f$j\f$.
There is also the possibility of considering only a sub set of atoms, through the `ATOMS` keyword.

Two are the possible usages:
- manual pick active \f$k\f$ or \f$2\theta\f$ (useful when biasing)
- grid of certain \f$k\f$ values (mainly for building the full structure factor)
The grid spacing is automatically chosen using the given `BOX_EDGE` which doesn't have to be exact, it is just a reference.
This spacing can be fine tuned through the `K_RESOLUTION` keyword.
To convert the frequency \f$k\f$ into the diffraction angle \f$2\theta\f$ the Bragg law (elastic scattering) is used:
\f[
  \frac{2\pi}{\lambda}sin(\theta)=\frac{k}{2}
\f]

\par Examplexs
Some usage examples
\plumedfile
manual: STRUCTURE_FACTOR_ISO LAMBDA=0.4123 ACTIVE_2THETA=13.5,25
grid_k: STRUCTURE_FACTOR_ISO BOX_EDGE=16 K_MIN=1 K_MAX=9
\endplumedfile

NB:
Minimal image convention is NOT used, even if simulation has pbc.
This introduces an arbitrary asymmetry, but reduces noise and gives a more precise Sk.

*/
//+ENDPLUMEDOC

struct indexes_pair //auxiliary class for double loop parallelization
{
  unsigned i,j;
  indexes_pair(unsigned _i,unsigned _j) : i(_i),j(_j) {}
};

class StructureFactor_iso : public Colvar {

private:
  bool grid_mode_;
  unsigned NumParallel_; //number of parallel tasks

  unsigned NumAtom_;
  std::vector<indexes_pair> AtomPair_;

  unsigned n_min_;
  double k_const_;
  std::vector<double> active_k_;

  std::vector<Value*> valueSk;

public:
  StructureFactor_iso(const ActionOptions&);
  virtual void calculate();
  static void registerKeywords(Keywords& );
};

PLUMED_REGISTER_ACTION(StructureFactor_iso,"STRUCTURE_FACTOR_ISO")

void StructureFactor_iso::registerKeywords(Keywords& keys)
{
  Colvar::registerKeywords(keys);
  keys.add("atoms","ATOMS","calculate Fourier components using only these atoms. Default is to use all atoms");

//set considered k by manually choosing them. Useful for biasing
  keys.add("optional","ACTIVE_K","manually set which k frequencies will be considered");
  keys.add("optional","ACTIVE_2THETA","manually set which k frequencies will be considered, by setting the angle in degrees");
  keys.add("optional","LAMBDA","wavelength of incident radiation. Compulsory when using angles instead of k frequencies");

//set considered k through a uniform grid. Used mainly when running the driver
  keys.add("optional","BOX_EDGE","set a reference value for the edge L of the simulation box");
  keys.add("optional","K_MAX","calculate the structure factor up to this k value. Default is reasonable when few primitive cells are simulated");
  keys.add("optional","K_MIN","calculate the structure factor from this k value. Default is reasonable when few primitive cells are simulated");
  keys.add("optional","K_RESOLUTION","default is 3. Actual resolution depends on the size of the simulation box");

//some flags to toggle
  keys.addFlag("SERIAL",false,"perform the calculation in serial even if multiple tasks are available");

//output components
  keys.add("optional","NAME_PRECISION","set the number of digits used for components name");
  keys.addOutputComponent("Sk","default","the instantaneous isotropic structure factor at wavelength k");
  ActionWithValue::useCustomisableComponents(keys); //needed to have an unknown number of components
}

StructureFactor_iso::StructureFactor_iso(const ActionOptions&ao):
  PLUMED_COLVAR_INIT(ao)
{
//parse and initialize:
//- get active k frequencies
  double lambda=-1;
  std::vector<double> active_2theta;
  parse("LAMBDA",lambda);
  parseVector("ACTIVE_2THETA",active_2theta);
  parseVector("ACTIVE_K",active_k_);
  if (active_2theta.size()>0 && lambda!=-1 && active_k_.size()==0)
  {
    active_k_.resize(active_2theta.size());
    for (unsigned k=0; k<active_k_.size(); k++)
    {
      plumed_massert(active_2theta[k]>0 && active_2theta[k]<180,"2theta must be between 0 and 180 degrees");
      active_k_[k]=4*PLMD::pi/lambda*sin(active_2theta[k]*PLMD::pi/360);
    }
    log.printf("  converting 2theta values to k=4*pi/lambda*sin(2theta*pi/360), using LAMBDA = %g\n",lambda);
  }
  else
    plumed_massert(active_2theta.size()==0 && lambda==-1,"must either set only both ACTIVE_2THETA and LAMBDA or none of them");

  if (active_k_.size()>0)
  {
    //print info
    log.printf("  using only %d manually selected k:",active_k_.size());
    for (unsigned k=0; k<active_k_.size(); k++)
      log.printf("  %g",active_k_[k]);
    log.printf("\n");
    grid_mode_=false;
  }
  else
    grid_mode_=true;

//- get k frequency grid
  double box_edge=-1;
  double k_max=-1;
  double k_min=-1;
  unsigned k_resolution=0;
  parse("BOX_EDGE",box_edge);
  parse("K_MAX",k_max);
  parse("K_MIN",k_min);
  parse("K_RESOLUTION",k_resolution);
  if (grid_mode_)
  {
    plumed_massert(box_edge!=-1,"either specific ACTIVE_K or a grid (thus at least BOX_EDGE) are needed");
    if (k_resolution==0)
      k_resolution=3; //default resolution
    //set the grid
    k_const_=PLMD::pi/box_edge/k_resolution;
    if (k_min==-1)
      k_min=4*PLMD::pi/box_edge; //below this is not physical
    n_min_=std::ceil(k_min/k_const_);
    unsigned n_max;
    if (k_max==-1)
      n_max=n_min_+149; //good guess for small simulations
    else
      n_max=std::floor(k_max/k_const_);
    active_k_.resize(n_max-n_min_+1,k_const_);
    for (unsigned k=0; k<active_k_.size(); k++)
      active_k_[k]*=(k+n_min_);
    //print grid info
    log.printf("  using a grid on k space:\n");
    log.printf("    reference BOX_EDGE L = %g\n",box_edge);
    log.printf("    K_MIN = %g [should be greater than 2pi/L=%g]\n",k_const_*n_min_,2*PLMD::pi/box_edge);
    log.printf("    K_MAX = %g\n",k_const_*n_max);
    log.printf("    K_RESOLUTION = %d --> %d grid points\n",k_resolution,active_k_.size());
  }
  else
    plumed_massert(box_edge==-1 && k_max==-1 && k_min==-1 && k_resolution==0,"if specific ACTIVE_K are given, no grid parameter (BOX_EDGE,K_MAX,K_MIN,K_RESOLUTION) should be set");

//add colvar components
  valueSk.resize(active_k_.size());
  std::ostringstream oss;
  unsigned name_precision=7;
  parse("NAME_PRECISION",name_precision);
  oss.precision(name_precision);
  log.printf("  components name are k value, with NAME_PRECISION = %d\n",name_precision);
  for (unsigned k=0; k<active_k_.size(); k++)
  {
    oss.str("");
    oss<<"Sk-"<<active_k_[k];
    addComponentWithDerivatives(oss.str());
    componentIsNotPeriodic(oss.str());
    valueSk[k]=getPntrToComponent(oss.str());
  }

//finish the parsing: get the atoms...
  vector<AtomNumber> atoms;
  parseAtomList("ATOMS",atoms);
  NumAtom_=atoms.size();
  if (NumAtom_==0) //default is to use all the atoms
  {
    NumAtom_=plumed.getAtoms().getNatoms();
    atoms.resize(NumAtom_);
    for(unsigned j=0; j<NumAtom_; j++)
      atoms[j].setIndex(j);
  }
  requestAtoms(atoms);//this must stay after the addComponentWithDerivatives otherwise segmentation violation

//- parallelization stuff
  NumParallel_=comm.Get_size();
  unsigned rank=comm.Get_rank();
  bool serial=false;
  parseFlag("SERIAL",serial);
  if (serial)
  {
    log.printf(" -- SERIAL: running without loop parallelization\n");
    NumParallel_=1;
    rank=0;
  }
//initialize the array of atoms pairs
  const unsigned tot_pairs=NumAtom_*(NumAtom_-1)/2;
  AtomPair_.reserve(tot_pairs/NumParallel_+1);
  unsigned pivot=0;
  for (unsigned i=0; i<NumAtom_; i++)
  {
    for (unsigned j=i+1; j<NumAtom_; j++)
    {
      pivot++;
      if ((pivot+rank)%NumParallel_==0)
        AtomPair_.emplace_back(i,j); //creates the new object directly in place
    }
  }
  log.printf("  over a total of N_tot=%d, considering a number of atoms N=%d\n",plumed.getAtoms().getNatoms(),NumAtom_);
  log.printf("    which gives a total number of ordered pairs equal to %d\n",tot_pairs);
  if(NumParallel_>1)
    log.printf("    redistributed over %d processors, each dealing with maximum %d pairs\n",NumParallel_,tot_pairs/NumParallel_+1);

//parsing finished
  checkRead();
}

void StructureFactor_iso::calculate()
{
//calculate the structure factor components
//the two modes are separated for better performances (much less 'if' to be checked)
  std::vector<double> Sk(active_k_.size(),0);
  std::vector<double> d_Sk;
  if (grid_mode_)//grid_mode is useful for getting the full structure factor
  {
    if (!doNotCalculateDerivatives())
      d_Sk.resize(3*NumAtom_*active_k_.size(),0);
    for (unsigned n=0; n<AtomPair_.size(); n++)
    {
      const Vector vR_ij=getPosition(AtomPair_[n].i)-getPosition(AtomPair_[n].j);
      const double R_ij=vR_ij.modulo();
      const double R_ij3=pow(R_ij,3);
      const double base_cos=cos(k_const_*R_ij);
      const double base_sin=sin(k_const_*R_ij);
      double prev_cos=cos(k_const_*R_ij*(n_min_-1));//room for improvements?
      double prev_sin=sin(k_const_*R_ij*(n_min_-1));
      for (unsigned k=0; k<active_k_.size(); k++)
      {
        const double KR=active_k_[k]*R_ij;
        const double cosKR=base_cos*prev_cos-base_sin*prev_sin;
        const double sinKR=base_cos*prev_sin+base_sin*prev_cos;
        Sk[k]+=sinKR/KR;
        prev_cos=cosKR;
        prev_sin=sinKR;
      //get derivatives
        if (!doNotCalculateDerivatives())
        {
          const Vector vDeriv_ij=vR_ij*((KR*cosKR-sinKR)/(active_k_[k]*R_ij3));
          for (unsigned l=0; l<3; l++)
            d_Sk[3*(k*NumAtom_+AtomPair_[n].i)+l]+=vDeriv_ij[l];
          for (unsigned l=0; l<3; l++)
            d_Sk[3*(k*NumAtom_+AtomPair_[n].j)+l]-=vDeriv_ij[l];
        }
      }
    }
  }
  else//manually choosing the active k is useful when biasing just a few peaks
  {
    d_Sk.resize(3*NumAtom_*active_k_.size(),0);
    for (unsigned n=0; n<AtomPair_.size(); n++)
    {
      const Vector vR_ij=getPosition(AtomPair_[n].i)-getPosition(AtomPair_[n].j);
      const double R_ij=vR_ij.modulo();
      const double R_ij3=pow(R_ij,3);
      for (unsigned k=0; k<active_k_.size(); k++)
      {
        const double KR=active_k_[k]*R_ij;
        const double sinKR=sin(KR);
        const double cosKR=cos(KR);
        Sk[k]+=sinKR/KR;
      //get derivatives
        const Vector vDeriv_ij=vR_ij*((KR*cosKR-sinKR)/(active_k_[k]*R_ij3));
        for (unsigned l=0; l<3; l++)
          d_Sk[3*(k*NumAtom_+AtomPair_[n].i)+l]+=vDeriv_ij[l];
        for (unsigned l=0; l<3; l++)
          d_Sk[3*(k*NumAtom_+AtomPair_[n].j)+l]-=vDeriv_ij[l];
      }
    }
  }
  if (NumParallel_>1)
  {
    comm.Sum(Sk);
    if (!doNotCalculateDerivatives())
      comm.Sum(d_Sk);
  }

//set the components values
  const double size_norm=2./NumAtom_;
  for (unsigned k=0; k<active_k_.size(); k++)
  {
    valueSk[k]->set(1+Sk[k]*size_norm);
    if (!doNotCalculateDerivatives())
    {
      for(unsigned ii=0; ii<3*NumAtom_; ii++)
        valueSk[k]->setDerivative(ii,d_Sk[3*NumAtom_*k+ii]*size_norm);
      setBoxDerivativesNoPbc(valueSk[k]);
    }
  }
}

} //colvar namespace
} //PLMD namespace
