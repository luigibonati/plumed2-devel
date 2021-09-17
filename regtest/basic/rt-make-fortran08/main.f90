PROGRAM main
  USE PLUMED_MODULE_F08
  IMPLICIT NONE
  TYPE(PLUMED) :: p
  if(.not. plumed_installed()) then
    stop "plumed not installed"
  endif
  CALL TEST3()
  CALL TEST4()
  CALL TEST5()
  call p%cmd("init")
  call p%finalize()
END PROGRAM main
