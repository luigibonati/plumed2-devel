! vim:ft=fortran



module plumed_module_f08
  use iso_c_binding
  use plumed_module
  implicit none

  private

  public :: plumed
  public :: plumed_create
  public :: plumed_installed

  type plumed
    character(kind=c_char,len=32), private :: handle
    logical,                       private :: initialized = .false.
  contains 
    private
    generic, public :: cmd => &
    pl_cmd_integer_0_0, &
    pl_cmd_integer_0_1, &
    pl_cmd_integer_0_2, &
    pl_cmd_integer_0_3, &
    pl_cmd_integer_0_4, &
    pl_cmd_integer_1_0, &
    pl_cmd_integer_1_1, &
    pl_cmd_integer_1_2, &
    pl_cmd_integer_1_3, &
    pl_cmd_integer_1_4, &
    pl_cmd_integer_2_0, &
    pl_cmd_integer_2_1, &
    pl_cmd_integer_2_2, &
    pl_cmd_integer_2_3, &
    pl_cmd_integer_2_4, &
    pl_cmd_real_0_0, &
    pl_cmd_real_0_1, &
    pl_cmd_real_0_2, &
    pl_cmd_real_0_3, &
    pl_cmd_real_0_4, &
    pl_cmd_real_1_0, &
    pl_cmd_real_1_1, &
    pl_cmd_real_1_2, &
    pl_cmd_real_1_3, &
    pl_cmd_real_1_4, &
    pl_cmd_real_2_0, &
    pl_cmd_real_2_1, &
    pl_cmd_real_2_2, &
    pl_cmd_real_2_3, &
    pl_cmd_real_2_4, &
    pl_cmd, &
    pl_cmd_char

    procedure :: pl_cmd_integer_0_0
    procedure :: pl_cmd_integer_0_1
    procedure :: pl_cmd_integer_0_2
    procedure :: pl_cmd_integer_0_3
    procedure :: pl_cmd_integer_0_4
    procedure :: pl_cmd_integer_1_0
    procedure :: pl_cmd_integer_1_1
    procedure :: pl_cmd_integer_1_2
    procedure :: pl_cmd_integer_1_3
    procedure :: pl_cmd_integer_1_4
    procedure :: pl_cmd_integer_2_0
    procedure :: pl_cmd_integer_2_1
    procedure :: pl_cmd_integer_2_2
    procedure :: pl_cmd_integer_2_3
    procedure :: pl_cmd_integer_2_4
    procedure :: pl_cmd_real_0_0
    procedure :: pl_cmd_real_0_1
    procedure :: pl_cmd_real_0_2
    procedure :: pl_cmd_real_0_3
    procedure :: pl_cmd_real_0_4
    procedure :: pl_cmd_real_1_0
    procedure :: pl_cmd_real_1_1
    procedure :: pl_cmd_real_1_2
    procedure :: pl_cmd_real_1_3
    procedure :: pl_cmd_real_1_4
    procedure :: pl_cmd_real_2_0
    procedure :: pl_cmd_real_2_1
    procedure :: pl_cmd_real_2_2
    procedure :: pl_cmd_real_2_3
    procedure :: pl_cmd_real_2_4
    procedure :: pl_cmd
    procedure :: pl_cmd_char

    procedure, public :: finalize => pl_finalize
    procedure, public :: incref => pl_incref
    procedure, public :: decref => pl_decref
    generic,   public :: assignment(=) => pl_assign
    final     :: pl_destructor
    procedure, public :: valid => pl_valid
    procedure, public :: use_count => pl_use_count
    procedure :: pl_assign
  end type plumed

  contains

     function plumed_installed() result(res)
       logical             :: res
       integer(kind=c_int) :: i
       call plumed_f_installed(i)
       res=i>0
     end function plumed_installed

     impure elemental subroutine plumed_create(this,kernel)
       type(plumed),    intent(out)          :: this
       character(len=*), intent(in), optional :: kernel
       if(present(kernel)) then
         call plumed_f_create_dlopen(kernel // c_null_char,this%handle)
       else
         call plumed_f_create(this%handle)
       endif
       this%initialized=.true.
     end subroutine plumed_create

     impure elemental subroutine pl_finalize(this)
       class(plumed), intent(inout) :: this
       if(this%initialized) then
         call plumed_f_finalize(this%handle)
         this%initialized=.false.
       endif
     end subroutine pl_finalize

     impure elemental subroutine pl_incref(this)
       class(plumed), intent(inout) :: this
       character(kind=c_char,len=32) :: that
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_create_reference(this%handle,that)
     end subroutine pl_incref

     impure elemental subroutine pl_decref(this,to)
       class(plumed),     intent(inout) :: this
       integer, optional, intent(in)    :: to
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       if(present(to)) then
         do while(this%use_count()>to)
           call plumed_f_finalize(this%handle)
         end do
       else
         call plumed_f_finalize(this%handle)
       endif
     end subroutine pl_decref

     ! "impure elemental" needed for the destructor to work on arrays
     impure elemental subroutine pl_destructor(this)
       type(plumed), intent(inout) :: this
       call this%finalize()
     end subroutine pl_destructor

     impure elemental function pl_valid(this) result(valid)
       class(plumed), intent(inout) :: this
       logical :: valid
       integer(c_int) :: i
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_valid(this%handle,i)
       valid=i>0
     end function pl_valid

     impure elemental function pl_use_count(this) result(use_count)
       class(plumed), intent(inout) :: this
       integer(c_int) :: use_count
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_use_count(this%handle,use_count)
     end function pl_use_count

     impure elemental subroutine pl_assign(this,that)
       class(plumed),intent(out) :: this
       class(plumed),intent(in)  :: that
       if(that%initialized) then
         call plumed_f_create_reference(that%handle,this%handle)
         this%initialized=.true.
       endif
     end subroutine pl_assign

     impure elemental subroutine pl_cmd(this,key)
       class(plumed),                 intent(inout) :: this ! inout to allow for initialization
       character(kind=c_char,len=*),  intent(in)    :: key
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,0) ! FIX: replace this to send NULL
     end subroutine pl_cmd

     subroutine pl_cmd_char(this,key,val)
       class(plumed),                 intent(inout) :: this ! inout to allow for initialization
       character(kind=c_char,len=*),  intent(in)    :: key
       character(kind=c_char,len=*), asynchronous   :: val
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val // c_null_char)
     end subroutine pl_cmd_char

    subroutine pl_cmd_integer_0_0(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      integer(KIND=c_int), asynchronous              :: val
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_integer_0_0
    subroutine pl_cmd_integer_0_1(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      integer(KIND=c_int), asynchronous              :: val(:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_integer_0_1
    subroutine pl_cmd_integer_0_2(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      integer(KIND=c_int), asynchronous              :: val(:,:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_integer_0_2
    subroutine pl_cmd_integer_0_3(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      integer(KIND=c_int), asynchronous              :: val(:,:,:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_integer_0_3
    subroutine pl_cmd_integer_0_4(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      integer(KIND=c_int), asynchronous              :: val(:,:,:,:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_integer_0_4
    subroutine pl_cmd_integer_1_0(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      integer(KIND=c_short), asynchronous              :: val
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_integer_1_0
    subroutine pl_cmd_integer_1_1(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      integer(KIND=c_short), asynchronous              :: val(:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_integer_1_1
    subroutine pl_cmd_integer_1_2(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      integer(KIND=c_short), asynchronous              :: val(:,:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_integer_1_2
    subroutine pl_cmd_integer_1_3(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      integer(KIND=c_short), asynchronous              :: val(:,:,:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_integer_1_3
    subroutine pl_cmd_integer_1_4(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      integer(KIND=c_short), asynchronous              :: val(:,:,:,:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_integer_1_4
    subroutine pl_cmd_integer_2_0(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      integer(KIND=c_long), asynchronous              :: val
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_integer_2_0
    subroutine pl_cmd_integer_2_1(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      integer(KIND=c_long), asynchronous              :: val(:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_integer_2_1
    subroutine pl_cmd_integer_2_2(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      integer(KIND=c_long), asynchronous              :: val(:,:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_integer_2_2
    subroutine pl_cmd_integer_2_3(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      integer(KIND=c_long), asynchronous              :: val(:,:,:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_integer_2_3
    subroutine pl_cmd_integer_2_4(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      integer(KIND=c_long), asynchronous              :: val(:,:,:,:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_integer_2_4
    subroutine pl_cmd_real_0_0(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      real(KIND=c_float), asynchronous              :: val
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_real_0_0
    subroutine pl_cmd_real_0_1(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      real(KIND=c_float), asynchronous              :: val(:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_real_0_1
    subroutine pl_cmd_real_0_2(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      real(KIND=c_float), asynchronous              :: val(:,:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_real_0_2
    subroutine pl_cmd_real_0_3(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      real(KIND=c_float), asynchronous              :: val(:,:,:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_real_0_3
    subroutine pl_cmd_real_0_4(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      real(KIND=c_float), asynchronous              :: val(:,:,:,:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_real_0_4
    subroutine pl_cmd_real_1_0(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      real(KIND=c_double), asynchronous              :: val
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_real_1_0
    subroutine pl_cmd_real_1_1(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      real(KIND=c_double), asynchronous              :: val(:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_real_1_1
    subroutine pl_cmd_real_1_2(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      real(KIND=c_double), asynchronous              :: val(:,:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_real_1_2
    subroutine pl_cmd_real_1_3(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      real(KIND=c_double), asynchronous              :: val(:,:,:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_real_1_3
    subroutine pl_cmd_real_1_4(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      real(KIND=c_double), asynchronous              :: val(:,:,:,:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_real_1_4
    subroutine pl_cmd_real_2_0(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      real(KIND=c_long_double), asynchronous              :: val
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_real_2_0
    subroutine pl_cmd_real_2_1(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      real(KIND=c_long_double), asynchronous              :: val(:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_real_2_1
    subroutine pl_cmd_real_2_2(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      real(KIND=c_long_double), asynchronous              :: val(:,:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_real_2_2
    subroutine pl_cmd_real_2_3(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      real(KIND=c_long_double), asynchronous              :: val(:,:,:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_real_2_3
    subroutine pl_cmd_real_2_4(this,key,val)
      class(plumed),                 intent(inout) :: this ! inout to allow for initialization
      character(kind=c_char,len=*),  intent(in)    :: key
      real(KIND=c_long_double), asynchronous              :: val(:,:,:,:)
       if(.not.this%initialized) then
         call plumed_create(this)
       endif
       call plumed_f_cmd(this%handle,key // c_null_char,val)
    end subroutine pl_cmd_real_2_4

end module plumed_module_f08

