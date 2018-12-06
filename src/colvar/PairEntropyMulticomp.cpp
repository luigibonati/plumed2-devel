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
#include "tools/NeighborListParallel.h"
#include "tools/Communicator.h"
#include "tools/Tools.h"

#include <string>

using namespace std;

namespace PLMD{
namespace colvar{

//+PLUMEDOC COLVAR PAIRENTROPY_MULTICOMP
/*
Calculate the per-atom global pair entropy of a two-component system using the expression:
\f[
s=-2\pi (\rho_A^2/\rho) k_B \int\limits_0^{r_{\mathrm{max}}} \left [ g_{AA}(r) \ln g_{AA}(r) - g_{AA}(r) + 1 \right ] r^2 dr 
- 4\pi (\rho_A\rho_B/\rho) k_B \int\limits_0^{r_{\mathrm{max}}} \left [ g_{AB}(r) \ln g_{AB}(r) - g_{AB}(r) + 1 \right ] r^2 dr 
- 2\pi (\rho_B^2\rho) k_B \int\limits_0^{r_{\mathrm{max}}} \left [ g_{BB}(r) \ln g_{BB}(r) - g_{BB}(r) + 1 \right ] r^2 dr 
\f]
where \f$ g_{\alpha\beta}(r) $\f are the pair distribution functions of the pairs and \f$ r_{\mathrm{max}} $\f is a cutoff in the integration (MAXR).
\f$ \rho_A $\f is the density of A atoms, \f$ \rho_B $\f is the density of B atoms and \f$ \rho $\f is the total density.
For the integration the interval from 0 to  \f$ r_{\mathrm{max}} $\f is partitioned in NHIST equal intervals. 
To make the calculation of \f$ g_{\alpha\beta}(r) $\f differentiable, the following function is used:
\f[
g_{\alpha\beta}(r) = \frac{N_A+N_B}{4 \pi \rho N_{\alpha} N_{\beta} r^2} \sum\limits_{i \in \alpha} \sum\limits_{i \in \beta} \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-(r-r_{ij})^2/(2\sigma^2)} ,
\f]
where \f$ \rho $\f is the density and \f$ sigma $\f is a broadening parameter (SIGMA).  
\par Example)
The following input tells plumed to calculate the pair entropy of a system of 250 atoms of type A and 250 atoms of type B.
\verbatim
PAIRENTROPY_MULTICOMP ...
 LABEL=S2
 GROUPA=1-250
 GROUPB=251-500
 MAXR=0.6
 SIGMA=0.01
 NHIST=100
 NLIST
 NL_CUTOFF=0.65
 NL_STRIDE=10
... PAIRENTROPY_MULTICOMP
\endverbatim
*/
//+ENDPLUMEDOC

class PairEntropyMulticomp : public Colvar {
  bool pbc;
  bool serial;
  bool do_pairs;
  bool doneigh;
  bool do_ignore_aa, do_ignore_bb;
  NeighborListParallel *nlAA, *nlBB, *nlAB, *nlBA;
  bool invalidateListAA, invalidateListAB, invalidateListBA, invalidateListBB;
  bool firsttimeAA, firsttimeAB, firsttimeBA, firsttimeBB;
  vector<AtomNumber> ga_lista,gb_lista,full_lista;
  double maxr, sigma;
  unsigned nhist;
  double rcut2;
  double invSqrt2piSigma, sigmaSqr2, sigmaSqr, invNormKernel;
  double deltar;
  unsigned deltaBin;
  unsigned numberOfAatoms, numberOfBatoms;
  std::vector<double> vectorX, vectorX2;
  // Integration routines
  double integrate(vector<double> integrand, double delta)const;
  Vector integrate(vector<Vector> integrand, double delta)const;
  Tensor integrate(vector<Tensor> integrand, double delta)const;
  // Output gofr and integrand
  void outputGofr(vector<double> gofrAA, vector<double> gofrAB, vector<double> gofrBB);
  void outputIntegrand(vector<double> gofrAA, vector<double> gofrAB, vector<double> gofrBB);
  mutable PLMD::OFile gofrOfile, integrandOfile;
  bool doOutputGofr;
  bool doOutputIntegrand;
  unsigned outputStride;
  // Average g(r)
  bool doAverageGofr;
  vector<double> avgGofrAA;
  vector<double> avgGofrAB;
  vector<double> avgGofrBB;
  unsigned iteration;
  unsigned averageGofrTau;
  // Kernel to calculate g(r)
  double kernel(double distance, double&der)const;
  // Low communication variant
  bool doLowComm;
public:
  explicit PairEntropyMulticomp(const ActionOptions&);
  ~PairEntropyMulticomp();
// active methods:
  virtual void calculate();
  virtual void prepare();
  static void registerKeywords( Keywords& keys );
};

PLUMED_REGISTER_ACTION(PairEntropyMulticomp,"PAIRENTROPY_MULTICOMP")

void PairEntropyMulticomp::registerKeywords( Keywords& keys ){
  Colvar::registerKeywords(keys);
  keys.addFlag("SERIAL",false,"Perform the calculation in serial - for debug purpose");
  keys.addFlag("NLIST",false,"Use a neighbour list to speed up the calculation");
  keys.addFlag("INDIVIDUAL_PAIRS",false,"Obtain pair entropy of AA, AB, and BB pairs");
  keys.addFlag("OUTPUT_GOFR",false,"Output g(r) of AA, AB, and BB pairs");
  keys.addFlag("OUTPUT_INTEGRAND",false,"Output integrand of AA, AB, and BB pairs");
  keys.addFlag("IGNORE_AA",false,"Ignore the calculation of the AA interaction");
  keys.addFlag("IGNORE_BB",false,"Ignore the calculation of the BB interaction");
  keys.addFlag("LOW_COMM",false,"Use the low communication variant of the algorithm.");
  keys.add("optional","OUTPUT_STRIDE","The frequency with which the output is written to files");
  keys.add("optional","NL_CUTOFF","The cutoff for the neighbour list");
  keys.add("optional","NL_STRIDE","The frequency with which we are updating the atoms in the neighbour list. If non specified or negative, it checks every step and rebuilds as needed.");
  keys.add("atoms","GROUPA","First list of atoms");
  keys.add("atoms","GROUPB","Second list of atoms");
  keys.add("compulsory","MAXR","1","Maximum distance for the radial distribution function ");
  keys.add("optional","NHIST","Number of bins in the rdf ");
  keys.add("compulsory","SIGMA","0.1","Width of gaussians ");
  keys.addFlag("AVERAGE_GOFR",false,"Average g(r) over time");
  keys.add("optional","AVERAGE_GOFR_TAU","Characteristic length of a window in which to average the g(r). It is in units of iterations and should be an integer. Zero corresponds to an normal average (infinite window).");
  keys.addOutputComponent("pairAA","INDIVIDUAL_PAIRS","Pair AA contribution to the multicomponent pair entropy");
  keys.addOutputComponent("pairAB","INDIVIDUAL_PAIRS","Pair AB contribution to the multicomponent pair entropy");
  keys.addOutputComponent("pairBB","INDIVIDUAL_PAIRS","Pair BB contribution to the multicomponent pair entropy");
  keys.addOutputComponent("full","INDIVIDUAL_PAIRS","Total multicomponent pair entropy");

}

PairEntropyMulticomp::PairEntropyMulticomp(const ActionOptions&ao):
PLUMED_COLVAR_INIT(ao),
pbc(true),
serial(false),
do_pairs(false),
invalidateListAA(true),
invalidateListAB(true),
invalidateListBA(true),
invalidateListBB(true),
firsttimeAA(true),
firsttimeAB(true),
firsttimeBA(true),
firsttimeBB(true)
{

  parseFlag("SERIAL",serial);

  parseAtomList("GROUPA",ga_lista);
  parseAtomList("GROUPB",gb_lista);

  bool nopbc=!pbc;
  parseFlag("NOPBC",nopbc);
  pbc=!nopbc;

// neighbor list stuff
  doneigh=false;
  bool nl_full_list=false;
  bool nl_full_list_cross=false;
  bool do_pair=false;
  double nl_cut=0.0;
  int nl_st=-1;
  double nl_skin;
  parseFlag("NLIST",doneigh);
  if(doneigh){
   parse("NL_CUTOFF",nl_cut);
   if(nl_cut<=0.0) error("NL_CUTOFF should be explicitly specified and positive");
   parse("NL_STRIDE",nl_st);
  }

// low communication stuff
  doLowComm=false;
  parseFlag("LOW_COMM",doLowComm);
  if(doLowComm) {
    log.printf("  using the low communication variant of the algorithm\n");
    nl_full_list=true;
  }

  /*
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
  */
  if(pbc) log.printf("  using periodic boundary conditions\n");
  else    log.printf("  without periodic boundary conditions\n");

  parse("MAXR",maxr);
  log.printf("  Integration in the interval from 0. to %f . \n", maxr );
  parse("SIGMA",sigma);
  log.printf("  The pair distribution functions is calculated with a Gaussian kernel with deviation %f . \n", sigma);
  double rcut = maxr + 3*sigma;
  rcut2 = (maxr + 3*sigma)*(maxr + 3*sigma);  // 3*sigma is hard coded
  if(doneigh){
    if(nl_cut<rcut) error("NL_CUTOFF should be larger than MAXR + 3*SIGMA");
    nl_skin=nl_cut-maxr;
  }
  nhist=ceil(maxr/(sigma/2.)) + 1; // Default value
  parse("NHIST",nhist);
  log.printf("  The interval is partitioned in %d equal parts and the integration is perfromed with the trapezoid rule. \n", nhist );
 
  // Construct full list 
  full_lista.reserve ( ga_lista.size() + gb_lista.size() );
  full_lista.insert (  full_lista.end() , ga_lista.begin(),  ga_lista.end() );
  full_lista.insert (  full_lista.end() , gb_lista.begin(),  gb_lista.end() );
  numberOfAatoms=ga_lista.size();
  numberOfBatoms=gb_lista.size();

  if(!(gb_lista.size()>0)){
    error("The group of atoms GROUPB has not been specified");
  }

  requestAtoms(full_lista);
 
  doOutputGofr=false;
  parseFlag("OUTPUT_GOFR",doOutputGofr);
  if (doOutputGofr) { 
     log.printf("  The g(r) will be written to a file \n.");
     gofrOfile.link(*this);
     gofrOfile.open("gofr.txt");
  }
  doOutputIntegrand=false;
  parseFlag("OUTPUT_INTEGRAND",doOutputIntegrand);
  if (doOutputIntegrand) {
     log.printf("  The integrand will be written to a file \n.");
     integrandOfile.link(*this);
     integrandOfile.open("integrand.txt");
  }
  outputStride=1;
  parse("OUTPUT_STRIDE",outputStride);
  if (outputStride!=1 && !doOutputGofr && !doOutputIntegrand) error("Cannot specify OUTPUT_STRIDE if OUTPUT_GOFR or OUTPUT_INTEGRAND not used");
  if (outputStride<1) error("The output stride specified with OUTPUT_STRIDE must be greater than or equal to one.");
  if (outputStride>1) log.printf("  The output stride to write g(r) or the integrand is %d \n", outputStride);


  doAverageGofr=false;
  parseFlag("AVERAGE_GOFR",doAverageGofr);
  if (doAverageGofr) {
     iteration = 1;
     avgGofrAA.resize(nhist);
     avgGofrAB.resize(nhist);
     avgGofrBB.resize(nhist);
  }
  averageGofrTau=0;
  parse("AVERAGE_GOFR_TAU",averageGofrTau);
  if (averageGofrTau!=0 && !doAverageGofr) error("AVERAGE_GOFR_TAU specified but AVERAGE_GOFR not given. Specify AVERAGE_GOFR or remove AVERAGE_GOFR_TAU");
  if (doAverageGofr && averageGofrTau==0) log.printf("The g(r) will be averaged over all frames \n");
  if (doAverageGofr && averageGofrTau!=0) log.printf("The g(r) will be averaged with a window of %d steps \n", averageGofrTau);


  parseFlag("INDIVIDUAL_PAIRS",do_pairs);
  if (do_pairs) log.printf("  The AA, AB, and BB contributions will be computed separately \n");

  do_ignore_aa=false;
  parseFlag("IGNORE_AA",do_ignore_aa);
  if (do_ignore_aa) log.printf("  The AA term will be ignored \n");
  do_ignore_bb=false;
  parseFlag("IGNORE_BB",do_ignore_bb);
  if (do_ignore_bb) log.printf("  The BB term will be ignored \n");

  checkRead();

  if (doneigh) {
     //nl= new NeighborListParallel(full_lista,pbc,getPbc(),comm,log,nl_cut,nl_st,nl_skin);
     nlAA = new NeighborListParallel(ga_lista,pbc,getPbc(),comm,log,nl_cut,nl_full_list,nl_st,nl_skin);
     nlAB = new NeighborListParallel(ga_lista,gb_lista,do_pair,pbc,getPbc(),comm,log,nl_cut,nl_full_list_cross,nl_st,nl_skin);
     if (doLowComm) nlBA = new NeighborListParallel(gb_lista,ga_lista,do_pair,pbc,getPbc(),comm,log,nl_cut,nl_full_list_cross,nl_st,nl_skin);
     nlBB = new NeighborListParallel(gb_lista,pbc,getPbc(),comm,log,nl_cut,nl_full_list,nl_st,nl_skin);
     log.printf("  using neighbor lists with\n");
     log.printf("  cutoff %f, and skin %f\n",nl_cut,nl_skin);
     if(nl_st>=0){
       log.printf("  update every %d steps\n",nl_st);
     } else {
       log.printf("  checking every step for dangerous builds and rebuilding as needed\n");
     }
  }


  // Define heavily used expressions
  double sqrt2piSigma = std::sqrt(2*pi)*sigma;
  invSqrt2piSigma = 1./sqrt2piSigma;
  sigmaSqr2 = 2.*sigma*sigma;
  sigmaSqr = sigma*sigma;
  deltar=maxr/(nhist-1.);
  deltaBin = std::floor(3*sigma/deltar); // 3*sigma is hard coded

  vectorX.resize(nhist);
  vectorX2.resize(nhist);
  for(unsigned i=0;i<nhist;++i){
    vectorX[i]=deltar*i;
    vectorX2[i]=vectorX[i]*vectorX[i];
  }

  // Define output components
  if (do_pairs) {
    addComponentWithDerivatives("pairAA"); componentIsNotPeriodic("pairAA");
    addComponentWithDerivatives("pairAB"); componentIsNotPeriodic("pairAB");
    addComponentWithDerivatives("pairBB"); componentIsNotPeriodic("pairBB");
    addComponentWithDerivatives("full"); componentIsNotPeriodic("full");
  } else {
    addValueWithDerivatives(); setNotPeriodic();
  }
}

PairEntropyMulticomp::~PairEntropyMulticomp(){
  if (doneigh) {
     if (!do_ignore_aa) nlAA->printStats();
     nlAB->printStats();
     if (doLowComm) nlBA->printStats();
     if (!do_ignore_bb) nlBB->printStats();
     delete nlAA;
     delete nlAB;
     if (doLowComm) delete nlBA;
     delete nlBB;
  }
}

void PairEntropyMulticomp::prepare(){
  if(!do_ignore_aa && doneigh && nlAA->getStride()>0){
    if(firsttimeAA) {
      invalidateListAA=true;
      firsttimeAA=false;
    } else if ( (nlAA->getStride()>=0) &&  (getStep()%nlAA->getStride()==0) ){
      invalidateListAA=true;
    } else {
      invalidateListAA=false;
    }
  }
  if(doneigh && nlAB->getStride()>0){
    if(firsttimeAB) {
      invalidateListAB=true;
      firsttimeAB=false;
    } else if ( (nlAB->getStride()>=0) &&  (getStep()%nlAB->getStride()==0) ){
      invalidateListAB=true;
    } else {
      invalidateListAB=false;
    }
  }
  if (doLowComm) {
    if(doneigh && nlBA->getStride()>0){
      if(firsttimeBA) {
        invalidateListBA=true;
        firsttimeBA=false;
      } else if ( (nlBA->getStride()>=0) &&  (getStep()%nlBA->getStride()==0) ){
        invalidateListBA=true;
      } else {
        invalidateListBA=false;
      }
    }
  }
  if(!do_ignore_bb && doneigh && nlBB->getStride()>0){
    if(firsttimeBB) {
      invalidateListBB=true;
      firsttimeBB=false;
    } else if ( (nlBB->getStride()>=0) &&  (getStep()%nlBB->getStride()==0) ){
      invalidateListBB=true;
    /*
    } else if ( (nlBB->getStride()<0) && !(nlBB->isListStillGood(getPositions())) ){
      invalidateListBB=true;
    */
    } else {
      invalidateListBB=false;
    }
  }
}

// calculator
void PairEntropyMulticomp::calculate()
{
  // Define output quantities
  //double pairEntropy;
  vector<Vector> derivAA(getNumberOfAtoms());
  vector<Vector> derivAB(getNumberOfAtoms());
  vector<Vector> derivBB(getNumberOfAtoms());
  // Define intermediate quantities
  vector<double> gofrAA(nhist);
  vector<double> gofrAB(nhist);
  vector<double> gofrBB(nhist);
  Matrix<Vector> gofrPrimeAA(nhist,getNumberOfAtoms());
  Matrix<Vector> gofrPrimeAB(nhist,getNumberOfAtoms());
  Matrix<Vector> gofrPrimeBB(nhist,getNumberOfAtoms());
  vector<Tensor> gofrVirialAA(nhist);
  vector<Tensor> gofrVirialAB(nhist);
  vector<Tensor> gofrVirialBB(nhist);
  // Calculate volume and density
  double volume=getBox().determinant();
  double density=getNumberOfAtoms()/volume; // This is (NA+NB)/V
  double densityA=numberOfAatoms/volume; // This is NA/V
  double densityB=numberOfBatoms/volume; // This is NB/V
  // Normalization of g(r)s
  double normConstantBaseAA = 2*pi*density*numberOfAatoms*(numberOfAatoms-1) / getNumberOfAtoms();
  double normConstantBaseAB = 4*pi*density*numberOfAatoms*numberOfBatoms / getNumberOfAtoms();
  double normConstantBaseBB = 2*pi*density*numberOfBatoms*(numberOfBatoms-1) / getNumberOfAtoms();
  double invNormConstantBaseAA = invSqrt2piSigma/normConstantBaseAA;
  double invNormConstantBaseAB = invSqrt2piSigma/normConstantBaseAB;
  double invNormConstantBaseBB = invSqrt2piSigma/normConstantBaseBB;
  // Setup parallelization
  unsigned stride=comm.Get_size();
  unsigned rank=comm.Get_rank();
  if(serial){
    stride=1;
    rank=0;
  }else{
    stride=comm.Get_size();
    rank=comm.Get_rank();
  }
  if (doneigh && !doLowComm) {
     if(!do_ignore_aa && invalidateListAA){
        vector<Vector> a_positions(getPositions().begin(),getPositions().begin() + ga_lista.size());
        nlAA->update(a_positions);
     }
     if(invalidateListAB){
        nlAB->update(getPositions());
     }
     if(!do_ignore_bb && invalidateListBB){
        vector<Vector> b_positions(getPositions().begin() + ga_lista.size(), getPositions().end() );
        nlBB->update(b_positions);
     }
     if (!do_ignore_aa) {
       // Loop over A atoms
       for(unsigned int i=0;i<nlAA->getNumberOfLocalAtoms();i++) {
          std::vector<unsigned> neighbors;
          unsigned index=nlAA->getIndexOfLocalAtom(i);
          neighbors=nlAA->getNeighbors(index);
          // Loop over A type neighbors
          for(unsigned int j=0;j<neighbors.size();j++) {  
             double dfunc, d2;
             Vector distance;
             Vector distance_versor;
             unsigned i0=index;
             unsigned i1=neighbors[j];
             if(getAbsoluteIndex(i0)==getAbsoluteIndex(i1)) continue;
             if(pbc){
              distance=pbcDistance(getPosition(i0),getPosition(i1));
             } else {
              distance=delta(getPosition(i0),getPosition(i1));
             }
             if ( (d2=distance[0]*distance[0])<rcut2 && (d2+=distance[1]*distance[1])<rcut2 && (d2+=distance[2]*distance[2])<rcut2) {
               //if (index==200) std::cout  << "NL: " << neighbors[j] << "\n";
               double distanceModulo=std::sqrt(d2);
               Vector distance_versor = distance / distanceModulo;
               unsigned bin=std::floor(distanceModulo/deltar);
               int minBin, maxBin; // These cannot be unsigned
               // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual distance
               minBin=bin - deltaBin;
               if (minBin < 0) minBin=0;
               if (minBin > (nhist-1)) minBin=nhist-1;
               maxBin=bin +  deltaBin;
               if (maxBin > (nhist-1)) maxBin=nhist-1;
               for(int k=minBin;k<maxBin+1;k+=1) {
                 invNormKernel=invNormConstantBaseAA/vectorX2[k];
                 gofrAA[k] += kernel(vectorX[k]-distanceModulo, dfunc);
                 if (!doNotCalculateDerivatives()) {
                    Vector value = dfunc * distance_versor;
                    gofrPrimeAA[k][i0] += value;
                    gofrPrimeAA[k][i1] -= value;
                    Tensor vv(value, distance);
                    gofrVirialAA[k] += vv;
                 }      
               }
             }
          }
       }
    }
    if (!do_ignore_bb) {
       // Loop over B atoms
       for(unsigned int i=0;i<nlBB->getNumberOfLocalAtoms();i++) {
          std::vector<unsigned> neighbors;
          unsigned index=nlBB->getIndexOfLocalAtom(i);
          neighbors=nlBB->getNeighbors(index);
          // Loop over B type neighbors
          for(unsigned int j=0;j<neighbors.size();j++) {  
             double dfunc, d2;
             Vector distance;
             Vector distance_versor;
             unsigned i0=index+ga_lista.size();
             unsigned i1=neighbors[j]+ga_lista.size();
             if(getAbsoluteIndex(i0)==getAbsoluteIndex(i1)) continue;
             if(pbc){
              distance=pbcDistance(getPosition(i0),getPosition(i1));
             } else {
              distance=delta(getPosition(i0),getPosition(i1));
             }
             if ( (d2=distance[0]*distance[0])<rcut2 && (d2+=distance[1]*distance[1])<rcut2 && (d2+=distance[2]*distance[2])<rcut2) {
               double distanceModulo=std::sqrt(d2);
               Vector distance_versor = distance / distanceModulo;
               unsigned bin=std::floor(distanceModulo/deltar);
               int minBin, maxBin; // These cannot be unsigned
               // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual distance
               minBin=bin - deltaBin;
               if (minBin < 0) minBin=0;
               if (minBin > (nhist-1)) minBin=nhist-1;
               maxBin=bin +  deltaBin;
               if (maxBin > (nhist-1)) maxBin=nhist-1;
               for(int k=minBin;k<maxBin+1;k+=1) {
                 invNormKernel=invNormConstantBaseBB/vectorX2[k];
                 gofrBB[k] += kernel(vectorX[k]-distanceModulo, dfunc);
                 if (!doNotCalculateDerivatives()) {
                    Vector value = dfunc * distance_versor;
                    gofrPrimeBB[k][i0] += value;
                    gofrPrimeBB[k][i1] -= value;
                    Tensor vv(value, distance);
                    gofrVirialBB[k] += vv;
                 }      
               }
             }
          }
       }
    }
    // Loop over A atoms
    for(unsigned int i=0;i<nlAB->getNumberOfLocalAtoms();i++) {
       std::vector<unsigned> neighbors;
       unsigned index=nlAB->getIndexOfLocalAtom(i);
       neighbors=nlAB->getNeighbors(index);
       // Loop over B type neighbors
       for(unsigned int j=0;j<neighbors.size();j++) {  
          double dfunc, d2;
          Vector distance;
          Vector distance_versor;
          unsigned i0=index;
          unsigned i1=neighbors[j];
          if(getAbsoluteIndex(i0)==getAbsoluteIndex(i1)) continue;
          if(pbc){
           distance=pbcDistance(getPosition(i0),getPosition(i1));
          } else {
           distance=delta(getPosition(i0),getPosition(i1));
          }
          if ( (d2=distance[0]*distance[0])<rcut2 && (d2+=distance[1]*distance[1])<rcut2 && (d2+=distance[2]*distance[2])<rcut2) {
            double distanceModulo=std::sqrt(d2);
            Vector distance_versor = distance / distanceModulo;
            unsigned bin=std::floor(distanceModulo/deltar);
            int minBin, maxBin; // These cannot be unsigned
            // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual distance
            minBin=bin - deltaBin;
            if (minBin < 0) minBin=0;
            if (minBin > (nhist-1)) minBin=nhist-1;
            maxBin=bin +  deltaBin;
            if (maxBin > (nhist-1)) maxBin=nhist-1;
            for(int k=minBin;k<maxBin+1;k+=1) {
              invNormKernel=invNormConstantBaseAB/vectorX2[k];
              gofrAB[k] += kernel(vectorX[k]-distanceModulo, dfunc);
              if (!doNotCalculateDerivatives()) {
                 Vector value = dfunc * distance_versor;
                 gofrPrimeAB[k][i0] += value;
                 gofrPrimeAB[k][i1] -= value;
                 Tensor vv(value, distance);
                 gofrVirialAB[k] += vv;
              }      
            }
          }
       }
    }
  } else if (!doneigh && !doLowComm) {
     // Loop over pairs
     for(unsigned int i=rank;i<(getNumberOfAtoms()-1);i+=stride) {
        for(unsigned int j=i+1;j<getNumberOfAtoms();j+=1) {
           double dfunc, d2;
           Vector distance;
           Vector distance_versor;
           if(getAbsoluteIndex(i)==getAbsoluteIndex(j)) continue;
           if(pbc){
            distance=pbcDistance(getPosition(i),getPosition(j));
           } else {
            distance=delta(getPosition(i),getPosition(j));
           }
           if ( (d2=distance[0]*distance[0])<rcut2 && (d2+=distance[1]*distance[1])<rcut2 && (d2+=distance[2]*distance[2])<rcut2) {
             double distanceModulo=std::sqrt(d2);
             Vector distance_versor = distance / distanceModulo;
             unsigned bin=std::floor(distanceModulo/deltar);
             int minBin, maxBin; // These cannot be unsigned
             // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual distance
             minBin=bin - deltaBin;
             if (minBin < 0) minBin=0;
             if (minBin > (nhist-1)) minBin=nhist-1;
             maxBin=bin +  deltaBin;
             if (maxBin > (nhist-1)) maxBin=nhist-1;
             //if (i==200 && j<numberOfAatoms) std::cout << "NO: " << j << "\n";
             for(int k=minBin;k<maxBin+1;k+=1) {
               // To which gofr does this pair of atoms contribute?
               if (i<numberOfAatoms && j<numberOfAatoms) {
                 if (!do_ignore_aa) {
                    // AA case
                    invNormKernel=invNormConstantBaseAA/vectorX2[k];
                    gofrAA[k] += kernel(vectorX[k]-distanceModulo, dfunc);
                    if (!doNotCalculateDerivatives()) {
                       Vector value = dfunc * distance_versor;
                       gofrPrimeAA[k][i] += value;
                       gofrPrimeAA[k][j] -= value;
                       Tensor vv(value, distance);
                       gofrVirialAA[k] += vv;
                    }
                 }
               } else if (i>=numberOfAatoms && j>=numberOfAatoms) {
                    if (!do_ignore_bb) {
                       // BB case
                       invNormKernel=invNormConstantBaseBB/vectorX2[k];
                       gofrBB[k] += kernel(vectorX[k]-distanceModulo, dfunc);
                       if (!doNotCalculateDerivatives()) {
                          Vector value = dfunc * distance_versor;
                          gofrPrimeBB[k][i] += value;
                          gofrPrimeBB[k][j] -= value;
                          Tensor vv(value, distance);
                          gofrVirialBB[k] += vv;
                       }
                    }
               } else {
                    // AB or BA case
                    invNormKernel=invNormConstantBaseAB/vectorX2[k];
                    gofrAB[k] += kernel(vectorX[k]-distanceModulo, dfunc);
                    if (!doNotCalculateDerivatives()) {
                       Vector value = dfunc * distance_versor;
                       gofrPrimeAB[k][i] += value;
                       gofrPrimeAB[k][j] -= value;
                       Tensor vv(value, distance);
                       gofrVirialAB[k] += vv;
                    }
               }
             }
           }
        }
     }
  } else if (doneigh && doLowComm) {
     if(!do_ignore_aa && invalidateListAA){
        vector<Vector> a_positions(getPositions().begin(),getPositions().begin() + ga_lista.size());
        nlAA->update(a_positions);
     }
     if(invalidateListAB){
        nlAB->update(getPositions());
     }
     if(invalidateListBA){
        vector<Vector> a_positions(getPositions().begin(),getPositions().begin() + ga_lista.size());
        vector<Vector> b_positions(getPositions().begin() + ga_lista.size(), getPositions().end() );
        vector<Vector> positions;
        positions.reserve( b_positions.size() + a_positions.size() );
        positions.insert (  positions.end() , b_positions.begin(),  b_positions.end() );
        positions.insert (  positions.end() , a_positions.begin(),  a_positions.end() );
        nlBA->update(positions);
     }
     if(!do_ignore_bb && invalidateListBB){
        vector<Vector> b_positions(getPositions().begin() + ga_lista.size(), getPositions().end() );
        nlBB->update(b_positions);
     }
     if (!do_ignore_aa) {
       // Loop over A atoms
       for(unsigned int i=0;i<nlAA->getNumberOfLocalAtoms();i++) {
          std::vector<unsigned> neighbors;
          unsigned index=nlAA->getIndexOfLocalAtom(i);
          neighbors=nlAA->getNeighbors(index);
          // Loop over A type neighbors
          for(unsigned int j=0;j<neighbors.size();j++) {  
             double dfunc, d2;
             Vector distance;
             Vector distance_versor;
             unsigned i0=index;
             unsigned i1=neighbors[j];
             if(getAbsoluteIndex(i0)==getAbsoluteIndex(i1)) continue;
             if(pbc){
              distance=pbcDistance(getPosition(i0),getPosition(i1));
             } else {
              distance=delta(getPosition(i0),getPosition(i1));
             }
             if ( (d2=distance[0]*distance[0])<rcut2 && (d2+=distance[1]*distance[1])<rcut2 && (d2+=distance[2]*distance[2])<rcut2) {
               //if (index==200) std::cout  << "NL: " << neighbors[j] << "\n";
               double distanceModulo=std::sqrt(d2);
               Vector distance_versor = distance / distanceModulo;
               unsigned bin=std::floor(distanceModulo/deltar);
               int minBin, maxBin; // These cannot be unsigned
               // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual distance
               minBin=bin - deltaBin;
               if (minBin < 0) minBin=0;
               if (minBin > (nhist-1)) minBin=nhist-1;
               maxBin=bin +  deltaBin;
               if (maxBin > (nhist-1)) maxBin=nhist-1;
               for(int k=minBin;k<maxBin+1;k+=1) {
                 invNormKernel=invNormConstantBaseAA/vectorX2[k];
                 gofrAA[k] += kernel(vectorX[k]-distanceModulo, dfunc)/2.;
                 if (!doNotCalculateDerivatives()) {
                    Vector value = dfunc * distance_versor;
                    gofrPrimeAA[k][i0] += value;
                    //gofrPrimeAA[k][i1] -= value;
                    Tensor vv(value/2., distance);
                    gofrVirialAA[k] += vv;
                 }      
               }
             }
          }
       }
    }
    if (!do_ignore_bb) {
       // Loop over B atoms
       for(unsigned int i=0;i<nlBB->getNumberOfLocalAtoms();i++) {
          std::vector<unsigned> neighbors;
          unsigned index=nlBB->getIndexOfLocalAtom(i);
          neighbors=nlBB->getNeighbors(index);
          // Loop over B type neighbors
          for(unsigned int j=0;j<neighbors.size();j++) {  
             double dfunc, d2;
             Vector distance;
             Vector distance_versor;
             unsigned i0=index+ga_lista.size();
             unsigned i1=neighbors[j]+ga_lista.size();
             if(getAbsoluteIndex(i0)==getAbsoluteIndex(i1)) continue;
             if(pbc){
              distance=pbcDistance(getPosition(i0),getPosition(i1));
             } else {
              distance=delta(getPosition(i0),getPosition(i1));
             }
             if ( (d2=distance[0]*distance[0])<rcut2 && (d2+=distance[1]*distance[1])<rcut2 && (d2+=distance[2]*distance[2])<rcut2) {
               double distanceModulo=std::sqrt(d2);
               Vector distance_versor = distance / distanceModulo;
               unsigned bin=std::floor(distanceModulo/deltar);
               int minBin, maxBin; // These cannot be unsigned
               // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual distance
               minBin=bin - deltaBin;
               if (minBin < 0) minBin=0;
               if (minBin > (nhist-1)) minBin=nhist-1;
               maxBin=bin +  deltaBin;
               if (maxBin > (nhist-1)) maxBin=nhist-1;
               for(int k=minBin;k<maxBin+1;k+=1) {
                 invNormKernel=invNormConstantBaseBB/vectorX2[k];
                 gofrBB[k] += kernel(vectorX[k]-distanceModulo, dfunc)/2.;
                 if (!doNotCalculateDerivatives()) {
                    Vector value = dfunc * distance_versor;
                    gofrPrimeBB[k][i0] += value;
                    //gofrPrimeBB[k][i1] -= value;
                    Tensor vv(value/2., distance);
                    gofrVirialBB[k] += vv;
                 }      
               }
             }
          }
       }
    }
    // Loop over A atoms
    for(unsigned int i=0;i<nlAB->getNumberOfLocalAtoms();i++) {
       std::vector<unsigned> neighbors;
       unsigned index=nlAB->getIndexOfLocalAtom(i);
       neighbors=nlAB->getNeighbors(index);
       // Loop over B type neighbors
       for(unsigned int j=0;j<neighbors.size();j++) {  
          double dfunc, d2;
          Vector distance;
          Vector distance_versor;
          unsigned i0=index;
          unsigned i1=neighbors[j];
          if(getAbsoluteIndex(i0)==getAbsoluteIndex(i1)) continue;
          if(pbc){
           distance=pbcDistance(getPosition(i0),getPosition(i1));
          } else {
           distance=delta(getPosition(i0),getPosition(i1));
          }
          if ( (d2=distance[0]*distance[0])<rcut2 && (d2+=distance[1]*distance[1])<rcut2 && (d2+=distance[2]*distance[2])<rcut2) {
            double distanceModulo=std::sqrt(d2);
            Vector distance_versor = distance / distanceModulo;
            unsigned bin=std::floor(distanceModulo/deltar);
            int minBin, maxBin; // These cannot be unsigned
            // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual distance
            minBin=bin - deltaBin;
            if (minBin < 0) minBin=0;
            if (minBin > (nhist-1)) minBin=nhist-1;
            maxBin=bin +  deltaBin;
            if (maxBin > (nhist-1)) maxBin=nhist-1;
            for(int k=minBin;k<maxBin+1;k+=1) {
              invNormKernel=invNormConstantBaseAB/vectorX2[k];
              gofrAB[k] += kernel(vectorX[k]-distanceModulo, dfunc)/2.;
              if (!doNotCalculateDerivatives()) {
                 Vector value = dfunc * distance_versor;
                 gofrPrimeAB[k][i0] += value;
                 //gofrPrimeAB[k][i1] -= value;
                 Tensor vv(value/2., distance);
                 gofrVirialAB[k] += vv;
              }      
            }
          }
       }
    }
    // Loop over B atoms
    for(unsigned int i=0;i<nlBA->getNumberOfLocalAtoms();i++) {
       std::vector<unsigned> neighbors;
       unsigned index=nlBA->getIndexOfLocalAtom(i);
       neighbors=nlBA->getNeighbors(index);
       // Loop over A type neighbors
       for(unsigned int j=0;j<neighbors.size();j++) {  
          double dfunc, d2;
          Vector distance;
          Vector distance_versor;
          unsigned i0=index+ga_lista.size();
          unsigned i1=neighbors[j]-gb_lista.size();
          if(getAbsoluteIndex(i0)==getAbsoluteIndex(i1)) continue;
          if(pbc){
           distance=pbcDistance(getPosition(i0),getPosition(i1));
          } else {
           distance=delta(getPosition(i0),getPosition(i1));
          }
          if ( (d2=distance[0]*distance[0])<rcut2 && (d2+=distance[1]*distance[1])<rcut2 && (d2+=distance[2]*distance[2])<rcut2) {
            double distanceModulo=std::sqrt(d2);
            Vector distance_versor = distance / distanceModulo;
            unsigned bin=std::floor(distanceModulo/deltar);
            int minBin, maxBin; // These cannot be unsigned
            // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual distance
            minBin=bin - deltaBin;
            if (minBin < 0) minBin=0;
            if (minBin > (nhist-1)) minBin=nhist-1;
            maxBin=bin +  deltaBin;
            if (maxBin > (nhist-1)) maxBin=nhist-1;
            for(int k=minBin;k<maxBin+1;k+=1) {
              invNormKernel=invNormConstantBaseAB/vectorX2[k];
              gofrAB[k] += kernel(vectorX[k]-distanceModulo, dfunc)/2.;
              if (!doNotCalculateDerivatives()) {
                 Vector value = dfunc * distance_versor;
                 gofrPrimeAB[k][i0] += value;
                 //gofrPrimeAB[k][i1] -= value;
                 Tensor vv(value/2., distance);
                 gofrVirialAB[k] += vv;
              }      
            }
          }
       }
    }
  } else if (!doneigh && doLowComm) {
     // Loop over pairs
     for(unsigned int i=rank;i<getNumberOfAtoms();i+=stride) {
        for(unsigned int j=0;j<getNumberOfAtoms();j+=1) {
           double dfunc, d2;
           Vector distance;
           Vector distance_versor;
           if(getAbsoluteIndex(i)==getAbsoluteIndex(j)) continue;
           if(pbc){
            distance=pbcDistance(getPosition(i),getPosition(j));
           } else {
            distance=delta(getPosition(i),getPosition(j));
           }
           if ( (d2=distance[0]*distance[0])<rcut2 && (d2+=distance[1]*distance[1])<rcut2 && (d2+=distance[2]*distance[2])<rcut2) {
             double distanceModulo=std::sqrt(d2);
             Vector distance_versor = distance / distanceModulo;
             unsigned bin=std::floor(distanceModulo/deltar);
             int minBin, maxBin; // These cannot be unsigned
             // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual distance
             minBin=bin - deltaBin;
             if (minBin < 0) minBin=0;
             if (minBin > (nhist-1)) minBin=nhist-1;
             maxBin=bin +  deltaBin;
             if (maxBin > (nhist-1)) maxBin=nhist-1;
             //if (i==200 && j<numberOfAatoms) std::cout << "NO: " << j << "\n";
             for(int k=minBin;k<maxBin+1;k+=1) {
               // To which gofr does this pair of atoms contribute?
               if (i<numberOfAatoms && j<numberOfAatoms) {
                 if (!do_ignore_aa) {
                    // AA case
                    invNormKernel=invNormConstantBaseAA/vectorX2[k];
                    gofrAA[k] += kernel(vectorX[k]-distanceModulo, dfunc)/2.;
                    if (!doNotCalculateDerivatives()) {
                       Vector value = dfunc * distance_versor;
                       gofrPrimeAA[k][i] += value;
                       //gofrPrimeAA[k][j] -= value;
                       Tensor vv(value/2., distance);
                       gofrVirialAA[k] += vv;
                    }
                 }
               } else if (i>=numberOfAatoms && j>=numberOfAatoms) {
                    if (!do_ignore_bb) {
                       // BB case
                       invNormKernel=invNormConstantBaseBB/vectorX2[k];
                       gofrBB[k] += kernel(vectorX[k]-distanceModulo, dfunc)/2.;
                       if (!doNotCalculateDerivatives()) {
                          Vector value = dfunc * distance_versor;
                          gofrPrimeBB[k][i] += value;
                          //gofrPrimeBB[k][j] -= value;
                          Tensor vv(value/2., distance);
                          gofrVirialBB[k] += vv;
                       }
                    }
               } else {
                    // AB or BA case
                    invNormKernel=invNormConstantBaseAB/vectorX2[k];
                    gofrAB[k] += kernel(vectorX[k]-distanceModulo, dfunc)/2.;
                    if (!doNotCalculateDerivatives()) {
                       Vector value = dfunc * distance_versor;
                       gofrPrimeAB[k][i] += value;
                       //gofrPrimeAB[k][j] -= value;
                       Tensor vv(value/2., distance);
                       gofrVirialAB[k] += vv;
                    }
               }
             }
           }
        }
     }
  }
  if(!serial){
    if (!do_ignore_aa) comm.Sum(gofrAA);
    if (!do_ignore_bb) comm.Sum(gofrBB);
    comm.Sum(gofrAB);
    if (!doNotCalculateDerivatives()) {
       if (!doLowComm) {
         if (!do_ignore_aa) comm.Sum(gofrPrimeAA);
         if (!do_ignore_bb) comm.Sum(gofrPrimeBB);
         comm.Sum(gofrPrimeAB);
       }
       if (!do_ignore_aa) comm.Sum(gofrVirialAA);
       if (!do_ignore_bb) comm.Sum(gofrVirialBB);
       comm.Sum(gofrVirialAB);
    }
  }
  /*
  for(unsigned j=0;j<nhist;++j){
    double x=deltar*(j+0.5);
    double normConstantAA = normConstantBaseAA*x*x;
    double normConstantAB = normConstantBaseAB*x*x;
    double normConstantBB = normConstantBaseBB*x*x;
    gofrAA[j] /= normConstantAA;
    gofrAB[j] /= normConstantAB;
    if (!do_ignore_bb) gofrBB[j] /= normConstantBB;
    if (!doNotCalculateDerivatives()) {
       gofrVirialAA[j] /= normConstantAA;
       gofrVirialAB[j] /= normConstantAB;
       if (!do_ignore_bb) gofrVirialBB[j] /= normConstantBB;
       for(unsigned k=0;k<getNumberOfAtoms();++k){
         gofrPrimeAA[j][k] /= normConstantAA;
         gofrPrimeAB[j][k] /= normConstantAB;
         if (!do_ignore_bb) gofrPrimeBB[j][k] /= normConstantBB;
       }
    }
  }
  */
  // Average g(r)s
  if (doAverageGofr) {
     if (!doNotCalculateDerivatives()) error("Cannot calculate derivatives or bias using the AVERAGE_GOFR option");
     double factor;
     if (averageGofrTau==0 || iteration < averageGofrTau) {
        iteration += 1;
        factor = 1./( (double) iteration );
     } else factor = 2./((double) averageGofrTau + 1.);
     for(unsigned i=0;i<nhist;++i){
        avgGofrAA[i] += (gofrAA[i]-avgGofrAA[i])*factor;
        gofrAA[i] = avgGofrAA[i];
        avgGofrAB[i] += (gofrAB[i]-avgGofrAB[i])*factor;
        gofrAB[i] = avgGofrAB[i];
        avgGofrBB[i] += (gofrBB[i]-avgGofrBB[i])*factor;
        gofrBB[i] = avgGofrBB[i];
     }
  }
  // Output of gofrs
  if (doOutputGofr && (getStep()%outputStride==0)) outputGofr(gofrAA,gofrAB,gofrBB);
  // Construct integrands
  vector<double> integrandAA(nhist);
  vector<double> integrandAB(nhist);
  vector<double> integrandBB(nhist);
  vector<double> logGofrAAx2(nhist);
  vector<double> logGofrABx2(nhist);
  vector<double> logGofrBBx2(nhist);
  for(unsigned j=0;j<nhist;++j){
    logGofrAAx2[j] = std::log(gofrAA[j])*vectorX2[j];
    logGofrABx2[j] = std::log(gofrAB[j])*vectorX2[j];
    logGofrBBx2[j] = std::log(gofrBB[j])*vectorX2[j];
    if (!do_ignore_aa) {
       if (gofrAA[j]<1.e-10) {
         integrandAA[j] = vectorX2[j];
       } else {
         integrandAA[j] = (gofrAA[j]*logGofrAAx2[j])+(-gofrAA[j]+1)*vectorX2[j];
       }
    }
    if (gofrAB[j]<1.e-10) {
      integrandAB[j] = vectorX2[j];
    } else {
      integrandAB[j] = (gofrAB[j]*logGofrABx2[j])+(-gofrAB[j]+1)*vectorX2[j];
    }
    if (!do_ignore_bb) {
       if (gofrBB[j]<1.e-10) {
         integrandBB[j] = vectorX2[j];
       } else {
         integrandBB[j] = (gofrBB[j]*logGofrBBx2[j])+(-gofrBB[j]+1)*vectorX2[j];
       }
    }
  }
  // Output of integrands
  if (doOutputIntegrand && (getStep()%outputStride==0)) outputIntegrand(integrandAA,integrandAB,integrandBB);
  // Integrate to obtain pair entropy
  double prefactorAA = -2*pi*(densityA*densityA/density);
  double prefactorAB = -4*pi*(densityA*densityB/density);
  double prefactorBB = -2*pi*(densityB*densityB/density);
  double pairAAvalue;
  if (!do_ignore_aa) pairAAvalue =  prefactorAA*integrate(integrandAA,deltar);
  else pairAAvalue = 0.;
  double pairBBvalue;
  if (!do_ignore_bb) pairBBvalue = prefactorBB*integrate(integrandBB,deltar);
  else pairBBvalue = 0.;
  double pairABvalue = prefactorAB*integrate(integrandAB,deltar);
  // Output individual pairs or only full pair entropy
  if (do_pairs) {
    Value* pairAA=getPntrToComponent("pairAA");
    Value* pairAB=getPntrToComponent("pairAB");
    Value* pairBB=getPntrToComponent("pairBB");
    Value* full=getPntrToComponent("full");
    pairAA->set(pairAAvalue);
    pairAB->set(pairABvalue);
    pairBB->set(pairBBvalue);
    full->set(pairAAvalue+pairABvalue+pairBBvalue);
  } else {
    setValue           (pairAAvalue+pairABvalue+pairBBvalue);
  } 
  if (!doNotCalculateDerivatives() ) {
    if (doLowComm && doneigh) {
      if (!do_ignore_aa) {
        for(unsigned int i=0;i<nlAA->getNumberOfLocalAtoms();i++) {
          unsigned index=nlAA->getIndexOfLocalAtom(i);
          vector<Vector> integrandDerivativesAA(nhist);
          for(unsigned k=0;k<nhist;++k){
            if (gofrAA[k]>1.e-10) { integrandDerivativesAA[k] = gofrPrimeAA[k][index]*logGofrAAx2[k]; }
          }
          derivAA[index] =  prefactorAA*integrate(integrandDerivativesAA,deltar);
          //log.printf("%f %f %f \n",gofrPrimeAA[k][index],derivAA[index]);
        }  
      }
      if (!do_ignore_bb) {
        for(unsigned int i=0;i<nlBB->getNumberOfLocalAtoms();i++) {
          unsigned index=nlBB->getIndexOfLocalAtom(i)+ga_lista.size();
          vector<Vector> integrandDerivativesBB(nhist);
          for(unsigned k=0;k<nhist;++k){
            if (gofrBB[k]>1.e-10) { integrandDerivativesBB[k] = gofrPrimeBB[k][index]*logGofrBBx2[k]; }
          }
          derivBB[index] =  prefactorBB*integrate(integrandDerivativesBB,deltar);
        }  
      }
      for(unsigned int i=0;i<nlAB->getNumberOfLocalAtoms();i++) {
        unsigned index=nlAB->getIndexOfLocalAtom(i);
        vector<Vector> integrandDerivativesAB(nhist);
        for(unsigned k=0;k<nhist;++k){
          if (gofrAB[k]>1.e-10) { integrandDerivativesAB[k] = gofrPrimeAB[k][index]*logGofrABx2[k]; }
        }
        derivAB[index] =  prefactorAB*integrate(integrandDerivativesAB,deltar);
      }  
      for(unsigned int i=0;i<nlBA->getNumberOfLocalAtoms();i++) {
        unsigned index=nlBA->getIndexOfLocalAtom(i)+ga_lista.size();
        vector<Vector> integrandDerivativesAB(nhist);
        for(unsigned k=0;k<nhist;++k){
          if (gofrAB[k]>1.e-10) { integrandDerivativesAB[k] = gofrPrimeAB[k][index]*logGofrABx2[k]; }
        }
        derivAB[index] =  prefactorAB*integrate(integrandDerivativesAB,deltar);
      }  
    } else {
      // Construct integrand and integrate derivatives
      for(unsigned int j=rank;j<getNumberOfAtoms();j+=stride) {
      //for(unsigned j=0;j<getNumberOfAtoms();++j) {
        vector<Vector> integrandDerivativesAA(nhist);
        vector<Vector> integrandDerivativesAB(nhist);
        vector<Vector> integrandDerivativesBB(nhist);
        for(unsigned k=0;k<nhist;++k){
          if (!do_ignore_aa && gofrAA[k]>1.e-10) { integrandDerivativesAA[k] = gofrPrimeAA[k][j]*logGofrAAx2[k]; }
          if (gofrAB[k]>1.e-10) { integrandDerivativesAB[k] = gofrPrimeAB[k][j]*logGofrABx2[k]; }
          if (!do_ignore_bb && gofrBB[k]>1.e-10) { integrandDerivativesBB[k] = gofrPrimeBB[k][j]*logGofrBBx2[k]; }
        }
        if (!do_ignore_aa) derivAA[j] =  prefactorAA*integrate(integrandDerivativesAA,deltar);
        derivAB[j] =  prefactorAB*integrate(integrandDerivativesAB,deltar);
        if (!do_ignore_bb) derivBB[j] =  prefactorBB*integrate(integrandDerivativesBB,deltar);
      }
    }
    if(!serial){
       if (!do_ignore_aa) comm.Sum(derivAA);
       comm.Sum(derivAB);
       if (!do_ignore_bb) comm.Sum(derivBB);
    }
    if (do_pairs) {
      Value* pairAA=getPntrToComponent("pairAA");
      Value* pairAB=getPntrToComponent("pairAB");
      Value* pairBB=getPntrToComponent("pairBB");
      Value* full=getPntrToComponent("full");
      for(unsigned j=0;j<getNumberOfAtoms();++j) setAtomsDerivatives (pairAA,j,derivAA[j]);
      for(unsigned j=0;j<getNumberOfAtoms();++j) setAtomsDerivatives (pairAB,j,derivAB[j]);
      for(unsigned j=0;j<getNumberOfAtoms();++j) setAtomsDerivatives (pairBB,j,derivBB[j]);
      for(unsigned j=0;j<getNumberOfAtoms();++j) setAtomsDerivatives (full,j,derivAA[j]+derivAB[j]+derivBB[j]);
    } else {
      for(unsigned j=0;j<getNumberOfAtoms();++j) setAtomsDerivatives (j,derivAA[j]+derivAB[j]+derivBB[j]);
    }
    // Virial of positions
    // Construct virial integrand
    vector<Tensor> integrandVirialAA(nhist);
    vector<Tensor> integrandVirialAB(nhist);
    vector<Tensor> integrandVirialBB(nhist);
    for(unsigned j=0;j<nhist;++j){
      if (!do_ignore_aa && gofrAA[j]>1.e-10) { integrandVirialAA[j] = gofrVirialAA[j]*logGofrAAx2[j];}
      if (gofrAB[j]>1.e-10) { integrandVirialAB[j] = gofrVirialAB[j]*logGofrABx2[j];}
      if (!do_ignore_bb && gofrBB[j]>1.e-10) { integrandVirialBB[j] = gofrVirialBB[j]*logGofrBBx2[j];}
    }
    // Integrate virial
    Tensor virialAA;
    Tensor virialAB = prefactorAB*integrate(integrandVirialAB,deltar);
    Tensor virialBB;
    if (!do_ignore_aa) virialAA = prefactorAA*integrate(integrandVirialAA,deltar);
    if (!do_ignore_bb) virialBB = prefactorBB*integrate(integrandVirialBB,deltar);
    // Virial of volume
    // Construct virial integrand
    vector<double> integrandVirialVolumeAA(nhist);
    vector<double> integrandVirialVolumeAB(nhist);
    vector<double> integrandVirialVolumeBB(nhist);
    for(unsigned j=0;j<nhist;j+=1) {
      if (!do_ignore_aa) integrandVirialVolumeAA[j] = (-gofrAA[j]+1)*vectorX2[j];
      integrandVirialVolumeAB[j] = (-gofrAB[j]+1)*vectorX2[j];
      if (!do_ignore_bb) integrandVirialVolumeBB[j] = (-gofrBB[j]+1)*vectorX2[j];
    }
    // Integrate virial
    if (!do_ignore_aa) virialAA += prefactorAA*integrate(integrandVirialVolumeAA,deltar)*Tensor::identity();
    virialAB += prefactorAB*integrate(integrandVirialVolumeAB,deltar)*Tensor::identity();
    if (!do_ignore_bb) virialBB += prefactorBB*integrate(integrandVirialVolumeBB,deltar)*Tensor::identity();
    // Set virial
    if (do_pairs) {
      Value* pairAA=getPntrToComponent("pairAA");
      Value* pairAB=getPntrToComponent("pairAB");
      Value* pairBB=getPntrToComponent("pairBB");
      Value* full=getPntrToComponent("full");
      setBoxDerivatives  (pairAA,virialAA);
      setBoxDerivatives  (pairAB,virialAB);
      setBoxDerivatives  (pairBB,virialBB);
      setBoxDerivatives  (full,virialAA+virialAB+virialBB);
    } else {
      setBoxDerivatives  (virialAA+virialAB+virialBB);
    }
  }
}

double PairEntropyMulticomp::kernel(double distance,double&der)const{
  // Gaussian function and derivative
  //double result = invSqrt2piSigma*std::exp(-distance*distance/sigmaSqr2) ;
  double result = invNormKernel*std::exp(-distance*distance/sigmaSqr2) ;
  der = -distance*result/sigmaSqr;
  return result;
}

double PairEntropyMulticomp::integrate(vector<double> integrand, double delta)const{
  // Trapezoid rule
  double result = 0.;
  for(unsigned i=1;i<(integrand.size()-1);++i){
    result += integrand[i];
  }
  result += 0.5*integrand[0];
  result += 0.5*integrand[integrand.size()-1];
  result *= delta;
  return result;
}

Vector PairEntropyMulticomp::integrate(vector<Vector> integrand, double delta)const{
  // Trapezoid rule
  Vector result;
  for(unsigned i=1;i<(integrand.size()-1);++i){
      result += integrand[i];
  }
  result += 0.5*integrand[0];
  result += 0.5*integrand[integrand.size()-1];
  result *= delta;
  return result;
}

Tensor PairEntropyMulticomp::integrate(vector<Tensor> integrand, double delta)const{
  // Trapezoid rule
  Tensor result;
  for(unsigned i=1;i<(integrand.size()-1);++i){
      result += integrand[i];
  }
  result += 0.5*integrand[0];
  result += 0.5*integrand[integrand.size()-1];
  result *= delta;
  return result;
}

void PairEntropyMulticomp::outputGofr(vector<double> gofrAA, vector<double> gofrAB, vector<double> gofrBB) {
  for(unsigned i=0;i<gofrAA.size();++i){
     gofrOfile.printField("r",vectorX[i]).printField("gofrAA",gofrAA[i]).printField("gofrAB",gofrAB[i]).printField("gofrBB",gofrBB[i]).printField();
  }
  gofrOfile.printf("\n");
  gofrOfile.printf("\n");
}

void PairEntropyMulticomp::outputIntegrand(vector<double> gofrAA, vector<double> gofrAB, vector<double> gofrBB) {
  for(unsigned i=0;i<gofrAA.size();++i){
     integrandOfile.printField("r",vectorX[i]).printField("integrandAA",gofrAA[i]).printField("integrandAB",gofrAB[i]).printField("integrandBB",gofrBB[i]).printField();
  }
  integrandOfile.printf("\n");
  integrandOfile.printf("\n");
}

}
}
