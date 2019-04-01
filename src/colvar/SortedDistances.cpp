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
#include "tools/SwitchingFunction.h"

#include <string>

#include <exception>

using namespace std;

namespace PLMD{
namespace colvar{

struct distance {
  Vector vec;
//  Vector der;
  //Tensor box;
  double mod;
  unsigned i; //central atom
  unsigned j; //neighbor
  unsigned n; //n-th distance
} ;

typedef struct distance s_distance;

//+PLUMEDOC COLVAR TETRAENTROPY
/*
*/
//+ENDPLUMEDOC

class SortedDistances : public Colvar {
  bool pbc;
  bool serial;
  bool doneigh;
  std::unique_ptr<NeighborListParallel> nl;
  bool invalidateList;
  bool firsttime;
  vector<AtomNumber> atoms_lista;
  unsigned nat;
  unsigned ndist;

  MPI_Datatype distance_type;
  int structlen;
  int blocklengths[5];
  MPI_Datatype types[5];
  MPI_Aint displacements[5];

public:
  explicit SortedDistances(const ActionOptions&);
  ~SortedDistances();
// active methods:
  virtual void calculate();
  virtual void prepare();
  static void registerKeywords( Keywords& keys );

};

PLUMED_REGISTER_ACTION(SortedDistances,"SORTED_DISTANCES")

void SortedDistances::registerKeywords( Keywords& keys ){

  Colvar::registerKeywords(keys);
  keys.add("optional","NDIST","how many distances to consider");
  keys.addFlag("SERIAL",false,"Perform the calculation in serial - for debug purpose");
  keys.addFlag("NLIST",false,"Use a neighbour list to speed up the calculation");
  keys.add("optional","NL_CUTOFF","The cutoff for the neighbour list");
  keys.add("optional","NL_STRIDE","The frequency with which we are updating the atoms in the neighbour list");
  keys.add("atoms","ATOMS","List of atoms");

}

SortedDistances::SortedDistances(const ActionOptions&ao):
PLUMED_COLVAR_INIT(ao),
pbc(true),
serial(false),
invalidateList(true),
firsttime(true)
{
  ndist=0;
  parse("NDIST",ndist);
  if(ndist<=0.0) error("NDIST should be explicitly specified and positive");

  parseFlag("SERIAL",serial);

  parseAtomList("ATOMS",atoms_lista);

  bool nopbc=!pbc;
  parseFlag("NOPBC",nopbc);
  pbc=!nopbc;

// neighbor list stuff
  doneigh=false;
  bool nl_full_list=true;
  double nl_cut=0.0,nl_skin;
  int nl_st=-1;
  parseFlag("NLIST",doneigh);
  if(doneigh){
    parse("NL_CUTOFF",nl_cut);
    if(nl_cut<=0.0) error("NL_CUTOFF should be explicitly specified and positive");
    parse("NL_STRIDE",nl_st);
    nl_skin=0.1*nl_cut;
    //build neighbor list
    nl.reset(new NeighborListParallel(atoms_lista,pbc,getPbc(),comm,log,nl_cut,nl_full_list,nl_st,nl_skin) );
    requestAtoms(nl->getFullAtomList());
    log.printf("  using neighbor lists with\n");
    log.printf("  cutoff %f, and skin %f\n",nl_cut,nl_skin);
    if(nl_st>=0){
      log.printf("  update every %d steps\n",nl_st);
    } else {
      log.printf("  checking every step for dangerous builds and rebuilding as needed\n");
    }
    if (nl_full_list) {
      log.printf("  using a full neighbor list\n");
    } else {
      log.printf("  using a half neighbor list\n");
    }
  } else {
    requestAtoms(atoms_lista);
  }
  
  checkRead();

  nat=atoms_lista.size();

  for(unsigned i=0; i<nat; i++){
    for(unsigned j=0; j<ndist; j++){
      auto label = std::to_string(i+1)+"_"+ std::to_string(j+1);
      addComponentWithDerivatives(label); componentIsNotPeriodic(label);
    }
  }
  turnOnDerivatives();

  if(doneigh && !serial){ 
    // struct parallelization
    structlen=5;

    blocklengths[0] = 3;
    //blocklengths[1] = 3;
    //blocklengths[2] = 9;
    blocklengths[1] = 1;
    blocklengths[2] = 1;
    blocklengths[3] = 1;
    blocklengths[4] = 1;

    types[0] = MPI_DOUBLE;
    //types[1] = MPI_DOUBLE;
    //types[2] = MPI_DOUBLE;
    types[1] = MPI_DOUBLE;
    types[2] = MPI_INT;
    types[3] = MPI_INT;
    types[4] = MPI_INT;

    displacements[0] = offsetof(s_distance, vec);
    //displacements[1] = offsetof(s_distance, der);
    //displacements[2] = offsetof(s_distance, box);
    displacements[1] = offsetof(s_distance, mod);
    displacements[2] = offsetof(s_distance, i);
    displacements[3] = offsetof(s_distance, j);
    displacements[4] = offsetof(s_distance, n);
   
    MPI_Datatype tmp_type;
    MPI_Aint lb, extent;
    MPI_Type_create_struct(structlen,blocklengths,displacements,types,&tmp_type);
    MPI_Type_get_extent( tmp_type, &lb, &extent );
    MPI_Type_create_resized( tmp_type, lb, extent, &distance_type );  
    MPI_Type_commit( &distance_type );
    {
      MPI_Aint typesize;
      MPI_Type_extent(distance_type,&typesize);
      log.printf("Data structure: %d bytes\n",sizeof(s_distance));   
      log.printf("MPIType extent: %d bytes\n",typesize);   
    }
  }

}

SortedDistances::~SortedDistances(){
  if (doneigh) {
     nl->printStats();
  }
  
  MPI_Type_free(&distance_type);
}

void SortedDistances::prepare(){
  if(doneigh && nl->getStride()>0){
    if(firsttime) {
      invalidateList=true;
      firsttime=false;
    } else if ( (nl->getStride()>0) &&  (getStep()%nl->getStride()==0) ){
      invalidateList=true;
    } else if ( (nl->getStride()<=0) && !(nl->isListStillGood(getPositions())) ){
      invalidateList=true;
    } else {
      invalidateList=false;
    }
  }
}

void SortedDistances::calculate()
{
  //parallelization stuff
  unsigned stride=comm.Get_size();
  unsigned rank=comm.Get_rank();
  if(serial){
    stride=1;
    rank=0;
  }else{
    stride=comm.Get_size();
    rank=comm.Get_rank();
  }

  // Setup neighbor list and parallelization
  if(doneigh && !serial){
    if(invalidateList)
      nl->update(getPositions());

  unsigned nloc=nl->getNumberOfLocalAtoms();
  std::vector<s_distance> d_rank;
  if(!serial) d_rank.resize(nloc*ndist);
  //loop over atoms
  for(unsigned int i=0; i<nloc; i++) {
    //get the neighbors of atom[i] within list
    unsigned index = nl->getIndexOfLocalAtom(i);
    vector<unsigned> neigh = nl->getNeighbors(index);
    unsigned num_neigh = neigh.size();
    if(num_neigh<ndist)
      error("There are not enough neighbors");
    Vector ri = getPosition(index);
    //loop over neighbors of atom[k]
    std::vector<s_distance> d (num_neigh);
    for(unsigned j=0;j<num_neigh;++j){
      //get relative positions between atom j and neighbor i
      d[j].i=index;
      d[j].j=neigh[j];
      if(pbc){ d[j].vec=pbcDistance(ri,getPosition( d[j].j ));
      } else { d[j].vec=delta(ri,getPosition( d[j].j )); }
      d[j].mod=d[j].vec.modulo();
      //if(getAbsoluteIndex(index)==getAbsoluteIndex(d[j].idx)) continue; 
    }
    std::sort(begin(d), end(d),
      [](const s_distance& a, const s_distance& b) { return a.mod < b.mod; });
     
    //loop only over the first ndist distances (remember: the first is the atom i, skip that)
    for(unsigned m=1;m<ndist+1;++m){
      if(!serial){
        d[m].n=m;
        //double invdist=1./d[j].mod;
        //d[j].der=(d[j].vec*invdist);
	//save into rank vector
        d_rank[i*ndist+(m-1)]=d[m];
      }else{
        auto label = std::to_string(i+1)+"_"+ std::to_string(m);
        Value* val=getPntrToComponent(label);
        val->set(d[m].mod);
        double invdist=1./d[m].mod;
        Vector der=(d[m].vec*invdist);
        setAtomsDerivatives (val,d[m].i,-der);
        setAtomsDerivatives (val,d[m].j,der);
        setBoxDerivatives(val,Tensor(d[m].vec,der));
      }
    }
 }
  // gather all the results
  std::vector<s_distance> d_all;
  d_all.resize(nat*ndist);

    MPI_Allgather(/*void* send_data */ d_rank.data(),/*int send_count */ nloc*ndist, /*MPI_Datatype send_datatype */ distance_type,
  		/*void* recv_data */ d_all.data(),/*int recv_count */ nloc*ndist,/*MPI_Datatype recv_datatype */ distance_type,
  		/*MPI_Comm communicator*/ comm.Get_comm() );

  //set values and derivatives
  for(unsigned k=0;k<d_all.size();++k){
    auto label = std::to_string(d_all[k].i+1)+"_"+ std::to_string(d_all[k].n);
    Value* val=getPntrToComponent(label);
    val->set(d_all[k].mod);
    Vector der = d_all[k].vec/d_all[k].mod;
    setAtomsDerivatives (val,d_all[k].j,der);
    setAtomsDerivatives (val,d_all[k].i,-der);
    setBoxDerivatives(val,Tensor(d_all[k].vec,der));
  }
 } else { //without neighbor list
  unsigned nloc=atoms_lista.size();
  //loop over atoms
  for(unsigned int i=0; i<nloc; i++) {
    unsigned index = atoms_lista[i].index();
    Vector ri = getPosition(index);
    std::vector<s_distance> d (nloc);
    for(unsigned j=0;j<nloc;++j){
      //get relative positions between atom j and neighbor i
      d[j].i=index;
      d[j].j=atoms_lista[j].index();
      if(pbc){ d[j].vec=pbcDistance(ri,getPosition( d[j].j ));
      } else { d[j].vec=delta(ri,getPosition( d[j].j )); }
      d[j].mod=d[j].vec.modulo();
      //if(getAbsoluteIndex(index)==getAbsoluteIndex(d[j].idx)) continue; 
    }
    std::sort(begin(d), end(d),
      [](const s_distance& a, const s_distance& b) { return a.mod < b.mod; });
     
    //loop only over the first ndist distances (remember: the first is the atom i, skip that)
    for(unsigned j=1;j<ndist+1;++j){
        auto label = std::to_string(i+1)+"_"+ std::to_string(j);
        Value* val=getPntrToComponent(label);
        val->set(d[j].mod);
        double invdist=1./d[j].mod;
        Vector der=(d[j].vec*invdist);
        setAtomsDerivatives (val,d[j].i,-der);
        setAtomsDerivatives (val,d[j].j,der);
        setBoxDerivatives(val,Tensor(d[j].vec,der));
    }
 }

 }

} //end calculate

}
}
