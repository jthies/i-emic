/**********************************************************************
 * Copyright by Jonas Thies, Univ. of Groningen 2006/7/8.             *
 * Permission to use, copy, modify, redistribute is granted           *
 * as long as this header remains intact.                             *
 * contact: jonas@math.rug.nl                                         *
 **********************************************************************/
#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include "Teuchos_RCP.hpp"

class Epetra_Map;
class Epetra_CrsMatrix;
class Epetra_Import;

#include "ptr_types.H"

namespace TRIOS {

//! utility class for the block-ILU preconditioner

/*! a purely static class that offers auxiliary routines
     for splitting and reordering matrices/vectors and
     related operations.
     */
class MatrixUtils
  {

  public:

    //! number of unknowns in our ocean model
    static const int nun=6;

    //! a general map creating function for a 3D grid
    
    /* creates a map of the form (i0:i1,j0:j1,k0:k1),
       column-major access (i is fastest).           
       Small letters indicate the bounds of the      
       subdomain, capital letters the bounds of the  
       global domain. The indices of subdomains are  
       allowed to overlap                          */
    static map_ptr CreateMap(int i0, int i1, int j0, int j1, int k0, int k1,
                             int I0, int I1, int J0, int J1, int K0, int K1,
                             const Epetra_Comm& comm);
  
    //! extract a map with nun=1 from a map with nun=6. 'var'
    //! is the variable to be extracted, i.e. UU,VV etc.
    static map_ptr CreateSubMap(const Epetra_Map& map, int var);

    //! extract a map with nun=2 from a map with nun=6. 'var'
    //! are the variables to be extracted, i.e. {UU,VV}, {TT,SS} etc.
    static map_ptr CreateSubMap(const Epetra_Map& map, const int var[2]);
    
    //! extract a map with nun=nvars from a map with nun=6. 'var'
    //! is the array of variables to be extracted.
    static map_ptr CreateSubMap
              (const Epetra_Map& map, const int *var, int nvars);

    //! given a map and an array indicating wether each node of the map is to be
    //! discarded (true) or not (false), this function creates a new map with the
    //! discarded entries removed.
    static map_ptr CreateSubMap(const Epetra_Map& map, const bool* discard);
    
    //! Split a map sequentially. The global indices corresponding to 
    // local index range [i1 i2] are returned in a new map
    static map_ptr CreateSubMap(const Epetra_Map& M, int i1, int i2);

    //! extracts those nodes from a column map that are actually present in a
    //! Matrix' Graph
    static map_ptr CompressColMap(const Epetra_CrsMatrix& A);

    //! a general function for gathering sparse matrices. It is fairly slow
    //! as it rebuilds the required "GatherMap" every time.
    static mat_ptr Gather(const Epetra_CrsMatrix& mat, int root);

    //! a general function for gathering vectors. It is fairly slow
    //! as it rebuilds the required "GatherMap" every time.
    static vec_ptr Gather(const Epetra_Vector& vec, int root);

    //! transform a "solve" or "standard" into a replicated "gather" map
    //! The new map will have its indices sorted in ascending order.    
    static map_ptr Gather(const Epetra_BlockMap& map, int root);

    //! transform a "solve" into a replicated "col" map, i.e.     
    //! the resulting map will have all gid's of the original one 
    //! on every process. The new map will have its indices       
    //! sorted in ascending order if you choose reorder=true,     
    //! otherwise the ordering is retained as it is.              
    static map_ptr AllGather(const Epetra_BlockMap& map, bool reorder=true);

    //! a general function for gathering vectors. It is fairly slow
    //! as it rebuilds the required "GatherMap" every time.
    static vec_ptr AllGather(const Epetra_Vector& vec);
    
    //! convert a 'gathered' vector into a distributed vector
    static vec_ptr Scatter(const Epetra_Vector& vec, const Epetra_Map& distmap);

    //! Workaround for the buggy 'ReplaceRowMap' routine in Trilinos 7.0.
    
    /*! It is much slower as it copies the whole matrix and should only 
        be used until they fix the builtin one
    */
    static mat_ptr ReplaceRowMap(mat_ptr A, const Epetra_Map& newmap);

    //! generalized ReplaceColMap that only requires the new col map to be a 
    //! subset of the old one (instead of being 'PointSameAs' as in the Trilinos
    //! function). The map should contain all column indices actually appearing,
    //! such a map can be created by 'CompressColMap'.
    static mat_ptr ReplaceColMap(mat_ptr A, const Epetra_Map& newmap);

  //! create an exact copy of a matrix removing the column map.
  //! This means that row- and column map have to be 'compatible'
  //! in the sense that the ColMap is a subset of the RowMap.
  //! It seems to be required in order to use Ifpack in some cases.
  static mat_ptr RemoveColMap(mat_ptr A);

    //! simultaneously replace row and column map (see comment for previous function)
    //! This is a special purpose function. The newcolmap must be a subset of the current
    //! colmap, i.e. you cannot really change the indexing scheme for the columns.
    static mat_ptr ReplaceBothMaps(mat_ptr A, const Epetra_Map& newmap, 
                                            const Epetra_Map& newcolmap);
    
    //! local upper or lower tri-solve. This can be used to gain insight into the process,
    //! but in principle the Trilinos version should work fine.
    static void TriSolve(const Epetra_CrsMatrix& A,const Epetra_Vector& b, Epetra_Vector& x);
        
    //! replace matrix by identity matrix
    
    /*! All matrix entries are set to 0 and the diagonal is set to 1.
    this is used for testing the block ILU precondioner step-wise and
    is not at all foolproof (it may fail when there is no diagonal   
    in the original matrix, for instance)
    */
    static void Identity(mat_ptr A);
    

    //! dump CrsMatrix to file (the matrix is gathered so it can be easily
    //! read into MATLAB etc., but this is only meant for debugging etc.)
    static void DumpMatrix(const Epetra_CrsMatrix& A, const std::string& filename);
    
    //! dump CrsMatrix in binary file (HDF5 format), which is parallel,   
    //! space efficient and preserves the data distribution. If the code  
    //! is compiled without -DHAVE_XDMF, this will cause an error.        
    //! The groupname is used to locate the stored matrix within the HDF- 
    //! file.                                                             
    //! if new_file==true, a new HDF file is created, deleting any old    
    //! file with the same name. Otherwise, an old file is opened, which  
    //! has to exist or an exception will be thrown.                      
    static void DumpMatrixHDF(const Epetra_CrsMatrix& A, const std::string& filename, 
        const std::string& groupname, bool new_file=true);
    
    //! Print an Epetra_RowMatrix to a stream
    static void PrintRowMatrix(const Epetra_RowMatrix&,std::ostream&);

    
    //! compute triple matrix product A*B*C or the like
    static mat_ptr TripleProduct(bool transA, const Epetra_CrsMatrix& A,
                                 bool transB, const Epetra_CrsMatrix& B,
                                 bool transC, const Epetra_CrsMatrix& C);

    //! compute triple matrix product M=A*B*C or the like if the pattern of M is known
    static void TripleProduct(mat_ptr M, bool transA, const Epetra_CrsMatrix& A,
                                 bool transB, const Epetra_CrsMatrix& B,
                                 bool transC, const Epetra_CrsMatrix& C);



    //! compute matrix-matrix product C=A*B (implemented using EpetraExt)
    static mat_ptr MatrixProduct(bool transA, const Epetra_CrsMatrix& A,
                                 bool transB, const Epetra_CrsMatrix& B);

    //! compute matrix-matrix product C=A*B (implemented using EpetraExt) if the patern of C 
    //! is already known (C is Filled)
    static void MatrixProduct(mat_ptr M, bool transA, const Epetra_CrsMatrix& A,
                                 bool transB, const Epetra_CrsMatrix& B);

    //! reads sparse matrix from THCM's beg/co/jco/info format
    //! if the prefix is, for instance, "A", then the matrix  
    //! is read from A.info, A.beg etc.
    static mat_ptr ReadThcmMatrix(std::string prefix, const Epetra_Comm& comm,
           const Epetra_Map& rowmap, const Epetra_Map& colmap,
           const Epetra_Map* rangemap=NULL,const Epetra_Map* domainmap=NULL);


  private:

    //! read array from file
    static void read_fortran_array(int n, int* array, std::string filename);

    //! read array from file
    static void read_fortran_array(int n, double* array, std::string filename);

  };

}//TRIOS

#endif // MATRIX_UTILS_h

