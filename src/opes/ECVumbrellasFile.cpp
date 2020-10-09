/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Copyright (c) 2020 of Michele Invernizzi.

The opes module is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The opes module is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "ExpansionCVs.h"
#include "core/ActionRegister.h"
#include "tools/File.h"

namespace PLMD {
namespace opes {

//+PLUMEDOC EXPANSION_CV ECV_UMBRELLAS_FILE
/*
Place Gaussian umbrellas on a line.
The umbrellas can be multidimensional, but you should rescale the dimensions so that a single SIGMA can be used.
Can be used with any Colvar as ARG.

\par Examples

us: ECV_UMBRELLAS_FILE ARG=cv MIN_CV=-1 MAX_CV=1 SIMGA=0.1

*/
//+ENDPLUMEDOC

class ECVumbrellasFile :
  public ExpansionCVs
{
private:
  unsigned P0_contribution_;
  std::vector<double> deltaFguess_;
  std::vector< std::vector<double> > centers_;
  std::vector< std::vector<double> > sigmas_;
  std::vector< std::vector<double> > ECVs_;
  std::vector< std::vector<double> > derECVs_;
  void initECVs();

public:
  explicit ECVumbrellasFile(const ActionOptions&);
  static void registerKeywords(Keywords& keys);
  void calculateECVs(const double *) override;
  const double * getPntrToECVs(unsigned) override;
  const double * getPntrToDerECVs(unsigned) override;
  std::vector< std::vector<unsigned> > getIndex_k() const override;
  std::vector<std::string> getLambdas() const override;
  void initECVs_observ(const std::vector<double>&,const unsigned,const unsigned) override;
  void initECVs_restart(const std::vector<std::string>&) override;
};

PLUMED_REGISTER_ACTION(ECVumbrellasFile,"ECV_UMBRELLAS_FILE")

void ECVumbrellasFile::registerKeywords(Keywords& keys) {
  ExpansionCVs::registerKeywords(keys);
  keys.use("ARG");
  keys.add("compulsory","FILE","the name of the file containing the umbrellas");
  keys.addFlag("ADD_P0",false,"add the unbiased Boltzmann distribution to the target distribution, to make sure to sample it");
  keys.addFlag("READ_HEIGHT",false,"read from FILE also the height of the umbrellas and use it for an initial guess DeltaF_i=-kbt*log(h_i)");
}

ECVumbrellasFile::ECVumbrellasFile(const ActionOptions&ao):
  Action(ao),
  ExpansionCVs(ao)
{
//get number of CVs
  const unsigned ncv=getNumberOfArguments();
  centers_.resize(ncv);
  sigmas_.resize(ncv);

//set P0_contribution_
  bool add_P0=false;
  parseFlag("ADD_P0",add_P0);
  if(add_P0)
    P0_contribution_=1;
  else
    P0_contribution_=0;

//set umbrellas
  bool read_height;
  parseFlag("READ_HEIGHT",read_height);
  std::string umbrellasFileName;
  parse("FILE",umbrellasFileName);
  IFile ifile;
  ifile.link(*this);
  if(ifile.FileExist(umbrellasFileName))
  {
    log.printf("  reading from FILE '%s'\n",umbrellasFileName.c_str());
    ifile.open(umbrellasFileName);
    ifile.allowIgnoredFields();
    double time;//first field is ignored
    while(ifile.scanField("time",time))
    {
      for(unsigned j=0; j<ncv; j++)
      {
        double centers_j;
        ifile.scanField(getPntrToArgument(j)->getName(),centers_j);
        centers_[j].push_back(centers_j);//this might be slow
      }
      for(unsigned j=0; j<ncv; j++)
      {
        double sigmas_j;
        ifile.scanField("sigma_"+getPntrToArgument(j)->getName(),sigmas_j);
        sigmas_[j].push_back(sigmas_j);
      }
      if(read_height)
      {
        double height;
        ifile.scanField("height",height);
        deltaFguess_.push_back(-kbt_*std::log(height));
      }
      ifile.scanField();
    }
  }
  else
    plumed_merror("Umbrellas FILE '"+umbrellasFileName+"' not found");

  checkRead();

//set ECVs stuff
  totNumECVs_=centers_[0].size()+P0_contribution_;
  ECVs_.resize(ncv,std::vector<double>(totNumECVs_));
  derECVs_.resize(ncv,std::vector<double>(totNumECVs_));

//printing some info
  log.printf("  total number of umbrellas = %lu\n",centers_[0].size());
  if(P0_contribution_==1)
    log.printf(" -- ADD_P0: the target includes also the unbiased probability itself\n");
}

void ECVumbrellasFile::calculateECVs(const double * cv) {
  for(unsigned j=0; j<getNumberOfArguments(); j++)
  {
    for(unsigned k=P0_contribution_; k<totNumECVs_; k++) //if ADD_P0, the first ECVs=0
    {
      const unsigned kk=k-P0_contribution_;
      const double dist_jk=difference(j,centers_[j][kk],cv[j])/sigmas_[j][kk]; //PBC might be present
      ECVs_[j][k]=0.5*std::pow(dist_jk,2);
      derECVs_[j][k]=dist_jk/sigmas_[j][kk];
    }
  }
}

const double * ECVumbrellasFile::getPntrToECVs(unsigned j)
{
  plumed_massert(isReady_,"cannot access ECVs before initialization");
  plumed_massert(j<getNumberOfArguments(),getName()+" has fewer CVs");
  return &ECVs_[j][0];
}

const double * ECVumbrellasFile::getPntrToDerECVs(unsigned j)
{
  plumed_massert(isReady_,"cannot access ECVs before initialization");
  plumed_massert(j<getNumberOfArguments(),getName()+" has fewer CVs");
  return &derECVs_[j][0];
}

std::vector< std::vector<unsigned> > ECVumbrellasFile::getIndex_k() const
{
  std::vector< std::vector<unsigned> > index_k(totNumECVs_,std::vector<unsigned>(getNumberOfArguments()));
  for(unsigned k=0; k<totNumECVs_; k++)
    for(unsigned j=0; j<getNumberOfArguments(); j++)
      index_k[k][j]=k; //this is trivial, since each center has a unique set of CVs
  return index_k;
}

std::vector<std::string> ECVumbrellasFile::getLambdas() const
{ //FIXME check also sigma?
  std::vector<std::string> lambdas(totNumECVs_);
  if(P0_contribution_==1)
  {
    std::ostringstream subs;
    subs<<"P0";
    for(unsigned j=1; j<getNumberOfArguments(); j++)
      subs<<"_P0";
    lambdas[0]=subs.str();
  }
  for(unsigned k=P0_contribution_; k<totNumECVs_; k++)
  {
    const unsigned kk=k-P0_contribution_;
    std::ostringstream subs;
    subs<<centers_[0][kk];
    for(unsigned j=1; j<getNumberOfArguments(); j++)
      subs<<"_"<<centers_[j][kk];
    lambdas[k]=subs.str();
  }
  return lambdas;
}

void ECVumbrellasFile::initECVs()
{
  plumed_massert(!isReady_,"initialization should not be called twice");
  isReady_=true;
  log.printf("  *%4u windows for %s\n",totNumECVs_,getName().c_str());
}

void ECVumbrellasFile::initECVs_observ(const std::vector<double>& all_obs_cvs,const unsigned ncv,const unsigned j)
{
  initECVs();
  if(deltaFguess_.size()>0)
  {
    for(unsigned j=0; j<getNumberOfArguments(); j++)
      for(unsigned k=P0_contribution_; k<totNumECVs_; k++)
        ECVs_[j][k]=std::min(barrier_,deltaFguess_[k])/kbt_;
    deltaFguess_.clear();
  }
  else
  {
    calculateECVs(&all_obs_cvs[j]);
    for(unsigned j=0; j<getNumberOfArguments(); j++)
      for(unsigned k=P0_contribution_; k<totNumECVs_; k++)
        ECVs_[j][k]=std::min(barrier_/kbt_,ECVs_[j][k]);
  }
}

void ECVumbrellasFile::initECVs_restart(const std::vector<std::string>& lambdas)
{
  std::size_t pos=0;
  for(unsigned j=0; j<getNumberOfArguments()-1; j++)
    pos = lambdas[0].find("_", pos+1); //checking only lambdas[0] is hopefully enough
  plumed_massert(pos<lambdas[0].length(),"this should not happen, fewer '_' than expected in "+getName());
  pos = lambdas[0].find("_", pos+1);
  plumed_massert(pos>lambdas[0].length(),"this should not happen, more '_' than expected in "+getName());

  std::vector<std::string> myLambdas=getLambdas();
  plumed_massert(myLambdas.size()==lambdas.size(),"RESTART - mismatch in number of "+getName());
  plumed_massert(std::equal(myLambdas.begin(),myLambdas.end(),lambdas.begin()),"RESTART - mismatch in lambda values of "+getName());

  initECVs();
}

}
}
