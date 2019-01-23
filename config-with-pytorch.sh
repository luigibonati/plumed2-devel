#fast and working (RGP CLUSTER)
./configure 	--enable-modules=+ves:+crystallization \
		CXXFLAGS="-O3 -D_GLIBCXX_USE_CXX11_ABI=0" \
		CPPFLAGS="-I/home/bonatl/Software/libtorch/include/torch/csrc/api/include/ -I/home/bonatl/Software/libtorch/include/" \
		LDFLAGS="-L/home/bonatl/Software/libtorch/lib -ltorch -lcaffe2 -lc10" \


# -------------------------------------------------------------------------------------------------------

#general setup

#where to install plumed
PREF=/local/bonatl/Software/plumed2

#assuming that libtorch-shared-with-deps-latest has been downloaded and unzipped in $LIBTORCH
LIBTORCH=/home/bonatl/Software/libtorch

#./configure 	--prefix=${PREF} \
#		--enable-modules=+ves:+crystallization \
#		CXXFLAGS="-O3 -D_GLIBCXX_USE_CXX11_ABI=0" \
#		CPPFLAGS="-I${LIBTORCH}/include/torch/csrc/api/include/ -I${LIBTORCH}/include/" \
#		LDFLAGS="-L${LIBTORCH}/lib -ltorch -lcaffe2 -lc10"


