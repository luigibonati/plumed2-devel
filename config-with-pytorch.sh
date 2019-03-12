#where to install LIBTORCH
LIBTORCH=/cluster/home/bonatil/Software/libtorch

if [ ! -d "$LIBTORCH" ]; then
  wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
  unzip libtorch-shared-with-deps-latest.zip
  mv libtorch $LIBTORCH
  rm libtorch-shared-with-deps-latest.zip
fi

./configure     --enable-modules=+ves:+crystallization \
                --enable-mpi --disable-openmp --enable-rpath CXX=mpic++ CC=mpicc \
                CXXFLAGS="-O3 -D_GLIBCXX_USE_CXX11_ABI=0" \
                CPPFLAGS="-I${LIBTORCH}/include/torch/csrc/api/include/ -I${LIBTORCH}/include/ -I${LIBTORCH}/include/torch" \
                LDFLAGS="-L${LIBTORCH}/lib -ltorch -lcaffe2 -lc10 -Wl,rpath,${LIBTORCH}/lib"

