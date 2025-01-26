#include "fdebug.h"

#define _INFO_(s)
#define _INFO2_(s1,s2)

#ifndef _INFO_
# define _INFO_(s) write(*,*)      "", s
# define _INFO2_(s1,s2) write(*,*) "", s1, s2
#endif

