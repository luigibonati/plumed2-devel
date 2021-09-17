
exe:
	$(CXX) -c $(CPPFLAGS) $(ADDCPPFLAGS) $(CXXFLAGS) *.cpp
	$(LD) *.o -o $@ $(PLUMED_LOAD)

exe-fortran:
	$(FC) -c $(PLUMED_FORTRAN) *.f90
	$(FC) *.o -o exe $(PLUMED_LOAD)

exe-fortran08:
	$(FC) -c $(PLUMED_FORTRAN08) *.f90
	$(FC) *.o -o exe $(PLUMED_LOAD)

test-fortran08:
	$(FC) -c __test_fortran08.f90
	rm -f test_fortran08*
	@echo "SUCCESS=YES"

print-fortran:
	@echo FC=$(FC)
