!#begindoc
 
#ifndef WITH_UNION

#define scdematrix  anymatrix

#endif

MODULE m_lumpdrop

CONTAINS
 
SUBROUTINE lumpdrop (nupp, newper, a, cspace, rspace)

USE m_dump
USE m_prcpars
USE m_build

INTEGER						, INTENT(IN)            :: nupp
TYPE (scdematrix)				, POINTER		:: a
INTEGER	, DIMENSION(1:a%n)			, INTENT(IN)            :: newper
DOUBLE PRECISION, DIMENSION(1:a%n)		, INTENT(IN OUT)        :: cspace
DOUBLE PRECISION, DIMENSION(1:a%n)		, INTENT(IN OUT)        :: rspace

!     Lump or Drop the "small" off-diagonal elements.

!     Computes the matrix  B  from the matrix  A, by lumping or
!     dropping the "small" off-diagonal elements in the new left- or
!     new upper part of  A  on the diagonal, such that:
!                             ( B11 | B12 )
!        B  =  P' A P - R  =  ( ----+---- )
!                             ( B21 | B22 )
!     and
!        B11 is a (block-)diagonal matrix of order 'Nupp'.

!     P  is a permutation matrix characterized by the permutation
!     vector 'NewPer'
!     R  is a rest matrix.

!     The result matrix ( P B P' ) is stored in the representation of  A.

!     Arguments:
!     ==========
!     A%N  		i   Number of rows/columns in the sub-matrix A.
!     Nupp     		i   Number of rows in upper partition, B11, of matrix B.
!     NewPer   		i   Permutation of 1:N, with
!                	    NewPer(i): row/column number in matrix A corresponding
!             		    with row/column i in partitioned matrix B.
        !                   The row numbers  NewPer(1:Nupp)  of A will belong to
!                           the upper part of B.
!     a%offd%beg	io  In:  a%offd%beg(i): index in 'a%offd%jco' and 'a%offd%co' of the first
!                           off-diagonal element in row i of matrix A.
!                  	    Out: a%offd%beg(i): index in 'a%offd%jco' and 'a%offd%co' of the first
!                           off-diagonal element in row i of matrix A after
!                           the "small" elements have been lumped or dropped.
!     a%offd%jco     	io  In:  a%offd%jco(nz): column number of off-diagonal element
!                           a%offd%co(nz).
!                  	    Out: a%offd%jco(nz): column number of off-diagonal element
!                           a%offd%co(nz) after the "small" elements have been
!                           lumped or dropped.
!     a%offd%co      	io  In:  a%offd%co(nz): value of off-diagonal element.
!                           Out: a%offd%co(nz): value of off-diagonal element after the
!                           "small" elements have been lumped or dropped.
!     a%dia%com     	io  In:  a%offd%co(nz): value of diagonal element.
!                           Out: a%offd%co(nz): value of diagonal element after the
!                           "small" off-diagonal elements have been lumped
!                           or dropped on this diagonal element
!     CSpace   		io  CSpace(1:A%N) Column lump space in A
!     RSpace   		io  RSpace(1:A%N) Row lump space in A


!#enddoc

!      Er wordt zodanig gelumpt dat eventueel aanwezige symmetrie
!      zo goed mogelijk wordt bewaard.
!      Daarom betekent lumpen in  B12  ook meteen lumpen in B21.
!      Kleine elementen in  B12^T  worden gelumpt, zodanig dat het
!      totaal van absolute waarden in zeg rij $i$ begrensd wordt
!      door RSpace(i).

!      Ook worden kleine elementen in  B21  gelumpt,
!      zodanig dat het totaal van absolute waarden in zeg rij $i$
!      begrensd wordt door ElmFctr*RSpace(i).
!      Dit totaal wordt vervolgens
!      afgetrokken van RSpace(i).
!      In  B12  wordt op soortgelijke manier gelumpt, op zo'n
!      manier dat bij een symmetrische matrix de symmetrie zo goed
!      mogelijk wordt bewaard.
!      Omdat het weggooien afhankelijk is van de volgorde waarin
!      niet-nul elementen worden doorlopen, kunnen we de symmetrie
!      echter niet garanderen.

!     Array arguments used as local variables:
!     ========================================
!     invper       Inverse of  NewPer:  invper(NewPer(i)) = i
!     rowsiz       rowsiz(i) = number of nonzeros in row i
!     sumic        sumic(c): sum in column  c  of lumped elements
!                  in B11 + B12 + B21
!     sumir        sumir(r): sum in row  r  of lumped elements
!                  in B11 + B12 + B21

!     Local Variables:
!     ================

INTEGER						:: ier
INTEGER 					:: firnz, lasnz, nz, nnzrow, annz
INTEGER 					:: newcol, newrow, oldcol, oldrow, baserow,offsetrow, offsetcol
DOUBLE PRECISION 				:: ratio, collum, rowlum, absval
INTEGER, ALLOCATABLE, DIMENSION(:)		:: invper, rowsiz
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:)	:: sumic, sumir

#ifdef DEBUG
DOUBLE PRECISION :: maxspc, minspc
DOUBLE PRECISION :: maxspr, minspr


CHARACTER (LEN=*), PARAMETER :: rounam = 'lumpdrop'

!     TRACE INFORMATION
PRINT '(A, X, A)' , 'Entry:', rounam
#endif

ALLOCATE( invper(1:a%n), STAT=ier )
IF (ier /= 0) CALL dump(__FILE__,__LINE__,'Allocation error')
ALLOCATE( rowsiz(1:a%n), STAT=ier )
IF (ier /= 0) CALL dump(__FILE__,__LINE__,'Allocation error')
ALLOCATE( sumic(1:a%n), STAT=ier )
IF (ier /= 0) CALL dump(__FILE__,__LINE__,'Allocation error')
ALLOCATE( sumir(1:a%n), STAT=ier )
IF (ier /= 0) CALL dump(__FILE__,__LINE__,'Allocation error')

ratio = FLOAT(nupp)/FLOAT(a%n)

!     Construct inverse permutation:
FORALL (newcol=1:a%n) invper(newper(newcol)) = newcol

!     Initialize partial row/column sums of dropped nonzeros:
sumic = 0.0D0
sumir = 0.0D0

!     NEW UPPER PARTITION:

DO  newrow = 1, nupp
  oldrow = newper(newrow)
  offsetrow = MOD(oldrow-1,a%dia%blksiz)+1
  baserow   = oldrow - offsetrow

!        Consider each non-zero off-diagonal element in row to
!        see whether the element can be lumped onto diagonal.
  
  firnz = a%offd%beg(oldrow)
  lasnz = a%offd%beg(oldrow+1)-1
  
  rowlum = rspace(oldrow)
  nnzrow = firnz
  DO nz = firnz, lasnz
    oldcol = a%offd%jco(nz)
    offsetcol = MOD(oldcol-1,a%dia%blksiz)+1
    newcol = invper(oldcol)
    
    absval = DABS(a%offd%co(nz))
    IF (newcol <= nupp) THEN
!              Off-diagonal element in block B11:
!              Lump element modified with Gustafsson-Factor:
      a%dia%com(offsetrow,baserow+offsetcol) = a%dia%com(offsetrow,baserow+offsetcol) + a%offd%co(nz) * gusfctr
      
!              Modify partial sums of lumped non-zeros of B11:
!               sumic(OldCol) = sumic(OldCol) + AbsVal
!               sumir(OldRow) = sumir(OldRow) + AbsVal
    ELSE
!              Off-diagonal element in block B12:
      
      collum = ratio*cspace(oldcol)
      IF ( (absval >= elmfctr*collum) .OR. (absval >= elmfctr*rowlum) .OR.  &
          (sumic(oldcol)+absval >= collum) .OR. (sumir(oldrow)+absval >= rowlum) ) THEN
!                 Element cannot be lumped, store back in matrix A:
      a%offd%jco(nnzrow) = oldcol
      a%offd%co(nnzrow)  = a%offd%co(nz)
      nnzrow       = nnzrow + 1
    ELSE
!                 Lump element modified with Gustafsson-Factor:
      a%dia%com(offsetrow,baserow+offsetcol) = a%dia%com(offsetrow,baserow+offsetcol) + a%offd%co(nz) * gusfctr
      
!                 Modify partial sums of lumped non-zeros of B12:
      sumic(oldcol) = sumic(oldcol) + absval
      sumir(oldrow) = sumir(oldrow) + absval
    END IF
  END IF
END DO

rowsiz(oldrow) = nnzrow - firnz
END DO


!     NEW LOWER PARTITION:

DO  newrow = nupp + 1, a%n
  oldrow = newper(newrow)
  offsetrow = MOD(oldrow-1,a%dia%blksiz)+1
  baserow   = oldrow - offsetrow
  
  firnz = a%offd%beg(oldrow)
  lasnz = a%offd%beg(oldrow+1)-1
  
  rowlum = ratio*rspace(oldrow)
  nnzrow = firnz
  DO nz = firnz, lasnz
    oldcol = a%offd%jco(nz)
    offsetcol = MOD(oldcol-1,a%dia%blksiz)+1
    newcol = invper(oldcol)
    
    IF (newcol > nupp) THEN
!              Store off-diagonal element:
      a%offd%jco(nnzrow) = oldcol
      a%offd%co(nnzrow)  = a%offd%co(nz)
      nnzrow       = nnzrow + 1
    ELSE
!              Off-diagonal element in block B21:
      
      absval = DABS(a%offd%co(nz))
      collum = cspace(oldcol)
      IF ( (absval >= elmfctr*rowlum) .OR. (absval >= elmfctr*collum) .OR.  &
          (sumir(oldrow)+absval >= rowlum) .OR. (sumic(oldcol)+absval >= collum) ) THEN
!                 Element cannot be lumped, store back in matrix  A:
      a%offd%jco(nnzrow) = oldcol
      a%offd%co(nnzrow)  = a%offd%co(nz)
      nnzrow       = nnzrow + 1
    ELSE
!                 Lump element modified with Gustafsson-Factor:
      a%dia%com(offsetrow,baserow+offsetcol) = a%dia%com(offsetrow,baserow+offsetcol) + a%offd%co(nz) * gusfctr
      
!                 Modify partial sums of lumped non-zeros of B21:
      sumic(oldcol) = sumic(oldcol) + absval
      sumir(oldrow) = sumir(oldrow) + absval
    END IF
  END IF
END DO

rowsiz(oldrow) = nnzrow - firnz
END DO


!     Remove the Lumped or Dropped elements in the representation of A:

annz = a%offd%beg(1) - 1
DO oldrow = 1, a%n
  firnz        = a%offd%beg(oldrow)
  a%offd%beg(oldrow) = annz + 1
  a%offd%jco(annz+1:annz+rowsiz(oldrow)) = a%offd%jco(firnz-1+1:firnz-1+rowsiz(oldrow))
  a%offd%co (annz+1:annz+rowsiz(oldrow)) = a%offd%co (firnz-1+1:firnz-1+rowsiz(oldrow))
  annz = annz + rowsiz(oldrow)
END DO
a%offd%beg(a%n+1) = annz + 1

#ifdef DEBUG
maxspc = cspace(1)-sumic(1)
minspc = cspace(1)-sumic(1)
maxspr = rspace(1)-sumir(1)
minspr = rspace(1)-sumir(1)
#endif

#ifdef DEBUG
DO   oldrow = 1, a%n
! CSpace(OldRow) = CSpace(OldRow)-sumic(OldRow)
! RSpace(OldRow) = RSpace(OldRow)-sumir(OldRow)
  maxspc = MAX(maxspc, cspace(oldrow))
  minspc = MIN(minspc, cspace(oldrow))
  maxspr = MAX(maxspr, rspace(oldrow))
  minspr = MIN(minspr, rspace(oldrow))
END DO
#endif

DEALLOCATE( invper, STAT=ier )
IF (ier /= 0) CALL dump(__FILE__,__LINE__,'Deallocation error')
DEALLOCATE( rowsiz, STAT=ier )
IF (ier /= 0) CALL dump(__FILE__,__LINE__,'Deallocation error')
DEALLOCATE( sumic, STAT=ier )
IF (ier /= 0) CALL dump(__FILE__,__LINE__,'Deallocation error')
DEALLOCATE( sumir, STAT=ier )
IF (ier /= 0) CALL dump(__FILE__,__LINE__,'Deallocation error')

#ifdef DEBUG
PRINT '(1P, 2(2X, A, E12.6), /, 2(2X, A, E12.6))',  &
    'Max. Row space = ', maxspr, 'Min. Row space = ', minspr,  &
    'Max. Col space = ', maxspc, 'Min. Col space = ', minspc
#endif

!     End of  lumpdrop
END SUBROUTINE lumpdrop

END MODULE