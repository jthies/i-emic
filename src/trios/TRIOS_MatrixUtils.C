/**********************************************************************
 * Copyright by Jonas Thies, Univ. of Groningen 2006/7/8.             *
 * Permission to use, copy, modify, redistribute is granted           *
 * as long as this header remains intact.                             *
 * contact: jonas@math.rug.nl                                         *
 **********************************************************************/
#include "GlobalDefinitions.H"
#include "Teuchos_RCP.hpp"
#include "Epetra_Map.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Vector.h"
#include "Epetra_Import.h"
#include "ptr_types.H"
#include "TRIOS_MatrixUtils.H"
#include "Epetra_Comm.h"
#include "EpetraExt_MatrixMatrix.h"

#include "Teuchos_FancyOStream.hpp"

// for sorting indices
#include <algorithm>

#include <fstream>

#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#include <mpi.h>
#endif

#include "Teuchos_StandardCatchMacros.hpp"

#ifdef HAVE_HDF5
#include "EpetraExt_HDF5.h"
#endif

namespace TRIOS {

    map_ptr
    MatrixUtils::CreateMap(int i0, int i1, int j0, int j1, int k0, int k1,
                           int I0, int I1, int J0, int J1, int K0, int K1,
                           const Epetra_Comm& comm)
      {
      map_ptr result = null;

      DEBUG("MatrixUtils::CreateMap ");
      DEBUG("["<<i0<<".."<<i1<<"]");
      DEBUG("["<<j0<<".."<<j1<<"]");
      DEBUG("["<<k0<<".."<<k1<<"]");

      int n = i1-i0+1; int N=I1-I0+1;
      int m = j1-j0+1; int M=J1-J0+1;
      int l = k1-k0+1; int L=K1-K0+1;

      DEBVAR(M);
      DEBVAR(N);
      DEBVAR(L);

      int NumMyElements = n*m*l;
      int NumGlobalElements = -1; // note that there may be overlap
      int *MyGlobalElements = new int[NumMyElements];

      int pos = 0;
      for (int k=k0; k<=k1; k++)
        for (int j=j0; j<=j1; j++)
          for (int i=i0; i<=i1; i++)
            {
            MyGlobalElements[pos++] = k*N*M + j*N + MOD((double)i,(double)N);
            }
      result = rcp(new Epetra_Map(NumGlobalElements,
                NumMyElements,MyGlobalElements,0,comm));
      delete [] MyGlobalElements;
      return result;
      }


    // extract a map with nun=1 from a map with nun=6. 'var'
    // is the variable to be extracted, i.e. UU,VV etc.
    map_ptr
    MatrixUtils::CreateSubMap(const Epetra_Map& map, int var)
      {
      return CreateSubMap(map,&var,1);
      }

    //! extract a map with nun=2 from a map with nun=6. 'var'
    //! are the variables to be extracted, i.e. {UU,VV}, {TT,SS} etc.
    map_ptr
    MatrixUtils::CreateSubMap(const Epetra_Map& map, const int var[2])
      {
      return CreateSubMap(map,var,2);
      }

    //! extract a map with nun=nvars from a map with nun=6. 'var'
    //! is the array of variables to be extracted.
    map_ptr 
    MatrixUtils::CreateSubMap(const Epetra_Map& map, const int *var, int nvars)
      {
      int dim = map.NumMyElements(); // number of entries in original map
      int numel = dim/nun; // number of blocks
      int subdim = numel*nvars; // number of entries in new map (<=dim)
      if (numel*nun!=dim)
        {
        ERROR("unexpected number of elements in map!",__FILE__,__LINE__);
        }
#ifdef DEBUGGING
  DEBUG("oooooo   Create submap ...\n");
  for (int i=0;i<nvars;i++) (*outFile) << var[i] << " ";
  (*outFile) << "\n number of points in original map: " << dim << std::endl;
  (*outFile) << " number of elements in original map: " << numel << std::endl;
  (*outFile) << " number of points in new map: "<<subdim<<std::endl;
#endif

      int *MyGlobalElements = new int[subdim];

      // take the entries from the old map that correspond
      // to those in 'vars' and put them in the input array
      // for the new map.
      int k=0;
      for (int i=0; i<numel; i++)
        {
        for (int j=0; j<nvars; j++)
          {
          const int nun=6;
          MyGlobalElements[k] = map.GID(i*nun+(var[j]-1));
          k++;
          }
        }

      map_ptr submap = 
        rcp(new Epetra_Map(-1, subdim, MyGlobalElements, 0, map.Comm()));
      delete [] MyGlobalElements;
      return submap;
      }

    //! given a map and an array indicating wether each node of the map is to be 
    //! discarded (true) or not (false), this function creates a new map with the
    //! discarded entries removed.
    map_ptr MatrixUtils::CreateSubMap
                 (const Epetra_Map& map, const bool* discard)
      {
      int numel = map.NumMyElements(); 
      int *MyGlobalElements = new int[numel]; // 'worst' case: no discarded nodes
      int numel_new = 0;

      for (int k=0;k<numel;k++)
        {
        if (!discard[k])
          {
          MyGlobalElements[numel_new] = map.GID(k); 
          numel_new++;
          }
        }
      map_ptr submap = rcp(new Epetra_Map(-1, numel_new, MyGlobalElements, 
                  map.IndexBase(), map.Comm()));
      delete [] MyGlobalElements;
      return submap;
      }

  // extract indices in a given global range [i1,i2]
  map_ptr MatrixUtils::CreateSubMap(const Epetra_Map& M, int i1, int i2)
    {

    int n = M.MaxAllGID();

#ifdef TESTING
 if (i1<0||i1>n) ERROR("CreateSubMap: lower bound out of range!",__FILE__,__LINE__);
 if (i2<0||i2>n) ERROR("CreateSubMap: upper bound out of range!",__FILE__,__LINE__);
 if (i2<i1) ERROR("CreateSubMap: invalid interval bounds!",__FILE__,__LINE__);
#endif

    int *MyGlobalElements = new int[M.NumMyElements()];
    int p=0;
    int gid;
    for (int i=0;i<M.NumMyElements();i++)
      {
      gid = M.GID(i);
      if (gid>=i1 && gid<=i2) MyGlobalElements[p++]=gid;
      }

    // build the two new maps. Set global num el. to -1 so Epetra recomputes it
    map_ptr M1 = rcp(new Epetra_Map(-1,p,MyGlobalElements,M.IndexBase(),M.Comm()) );
    delete [] MyGlobalElements;
    return M1;
    }


  // compress a matrix' column map so that the resulting map contains 
  // only points actually appearing as column indices of the matrix 
  map_ptr MatrixUtils::CompressColMap(const Epetra_CrsMatrix& A)
    {
    DEBUG("Compress column map of "<<A.Label());

    if (!A.HaveColMap()) ERROR("Matrix has no column map!",__FILE__,__LINE__);

    const Epetra_Map& old_map = A.ColMap();
    int n_old = old_map.NumMyElements();
    bool *is_col_entry = new bool[n_old];

    for (int i=0;i<n_old;i++) is_col_entry[i]=false;

    for (int i=0;i<A.NumMyRows();i++)
      {
      int *ind;
      int len;
      CHECK_ZERO(A.Graph().ExtractMyRowView(i,len,ind));
      for (int j=0;j<len;j++) is_col_entry[ind[j]]=true;
      }

    int n_new = 0;
    int *new_elements = new int[n_old];

    for (int i=0;i<n_old;i++) 
      {
      if (is_col_entry[i]) 
        {
        new_elements[n_new++] = old_map.GID(i);
        }
      }

    map_ptr new_map = rcp(new 
           Epetra_Map(-1,n_new,new_elements,old_map.IndexBase(),old_map.Comm()));

    delete [] new_elements;
    delete [] is_col_entry;

//    DEBVAR(old_map);
//    DEBVAR(*new_map);

    return new_map;
    }


  // create "Gather" map from "Solve" map
  map_ptr MatrixUtils::Gather(const Epetra_BlockMap& map, int root)
    {

    int NumMyElements = map.NumMyElements();
    int NumGlobalElements = map.NumGlobalElements();
    const Epetra_Comm& Comm = map.Comm();

    int *MyGlobalElements = new int[NumMyElements];
    int *AllGlobalElements = NULL;

    for (int i=0; i<NumMyElements;i++)
      {
      MyGlobalElements[i] = map.GID(i);
      }

    if (Comm.MyPID()==root)
      {
      AllGlobalElements = new int[NumGlobalElements];
      }

#ifdef HAVE_MPI

    const Epetra_MpiComm MpiComm = dynamic_cast<const Epetra_MpiComm&>(Comm);
    int *counts, *disps;
    counts = new int[Comm.NumProc()];
    disps = new int[Comm.NumProc()+1];
    MPI_Gather(&NumMyElements,1,MPI_INTEGER,
               counts,1,MPI_INTEGER,root,MpiComm.GetMpiComm());

    if (Comm.MyPID()==root)
      {
      disps[0]=0;
      for (int p=0;p<Comm.NumProc();p++)
        {
        disps[p+1] = disps[p]+counts[p];
        }
      }

    MPI_Gatherv(MyGlobalElements, NumMyElements,MPI_INTEGER, 
                AllGlobalElements, counts,disps, MPI_INTEGER, root, MpiComm.GetMpiComm());
#else
   for (int i=0;i<NumMyElements;i++) AllGlobalElements[i]=MyGlobalElements[i];
#endif

    if (Comm.MyPID()!=root) 
      {
      NumMyElements=0;
      }
    else
      {
      NumMyElements=NumGlobalElements;
      std::sort(AllGlobalElements,AllGlobalElements+NumGlobalElements);
      }

  // build the new (gathered) map
  map_ptr gmap = rcp(new Epetra_Map (NumGlobalElements, NumMyElements, 
                       AllGlobalElements, map.IndexBase(), Comm) );

    if (Comm.MyPID()==root)
      {
      delete [] AllGlobalElements;
      }


    delete [] MyGlobalElements;

    return gmap;

    }


  // create "col" map from "Solve" map
  map_ptr MatrixUtils::AllGather(const Epetra_BlockMap& map, bool reorder)
    {

    int NumMyElements = map.NumMyElements();
    int NumGlobalElements = map.NumGlobalElements();
    const Epetra_Comm& Comm = map.Comm();

    int *MyGlobalElements = new int[NumMyElements];
    int *AllGlobalElements = new int[NumGlobalElements];

    for (int i=0; i<NumMyElements;i++)
      {
      MyGlobalElements[i] = map.GID(i);
      }

#ifdef HAVE_MPI

    const Epetra_MpiComm MpiComm = dynamic_cast<const Epetra_MpiComm&>(Comm);
    int *counts, *disps;
    counts = new int[Comm.NumProc()];
    disps = new int[Comm.NumProc()+1];
    MPI_Allgather(&NumMyElements,1,MPI_INTEGER,
               counts,1,MPI_INTEGER,MpiComm.GetMpiComm());

    disps[0]=0;
    for (int p=0;p<Comm.NumProc();p++)
      {
      disps[p+1] = disps[p]+counts[p];
      }

    MPI_Allgatherv(MyGlobalElements, NumMyElements,MPI_INTEGER, 
                AllGlobalElements, counts,disps, MPI_INTEGER, MpiComm.GetMpiComm());
#else
   for (int i=0;i<NumMyElements;i++) AllGlobalElements[i]=MyGlobalElements[i];
#endif

  NumMyElements=NumGlobalElements;
  NumGlobalElements = -1;

  if (reorder)
    {
    std::sort(AllGlobalElements,AllGlobalElements+NumMyElements);
    }

  // build the new (gathered) map
  map_ptr gmap = rcp(new Epetra_Map (NumGlobalElements, NumMyElements, 
                       AllGlobalElements, map.IndexBase(), Comm) );



    delete [] MyGlobalElements;
    delete [] AllGlobalElements;

    return gmap;

    }//AllGather

    vec_ptr MatrixUtils::Gather(const Epetra_Vector& vec, int root)
      {
      DEBUG("Gather vector "<<vec.Label());
      const Epetra_BlockMap& map_dist = vec.Map();
      map_ptr map = Gather(map_dist,root);

      vec_ptr gvec = rcp(new Epetra_Vector(*map));

      import_ptr import = rcp(new Epetra_Import(*map,map_dist) );

      CHECK_ZERO(gvec->Import(vec,*import,Insert));

      gvec->SetLabel(vec.Label());

      return gvec;

      }

    vec_ptr MatrixUtils::AllGather(const Epetra_Vector& vec)
      {
      DEBUG("AllGather vector "<<vec.Label());
      const Epetra_BlockMap& map_dist = vec.Map();
      map_ptr map = AllGather(map_dist);
      vec_ptr gvec = rcp(new Epetra_Vector(*map));

      import_ptr import = rcp(new Epetra_Import(*map,map_dist) );

      CHECK_ZERO(gvec->Import(vec,*import,Insert));

      gvec->SetLabel(vec.Label());
      DEBUG("done!");
      return gvec;

      }

    mat_ptr MatrixUtils::Gather(const Epetra_CrsMatrix& mat, int root)
      {
      DEBUG("Gather matrix "<<mat.Label());
      const Epetra_Map& rowmap_dist = mat.RowMap();
      // we take the domain map as the colmap is potentially overlapping
      const Epetra_Map& colmap_dist = mat.DomainMap();
      // gather the row map
      map_ptr rowmap = Gather(rowmap_dist,root);
      // gather the col map
      map_ptr colmap = Gather(colmap_dist,root);

      //we only guess the number of row entries, this routine is not performance critical
      // as it should only be used for debugging anyway
      int num_entries = mat.NumGlobalNonzeros()/mat.NumGlobalRows();
      mat_ptr gmat = rcp(new Epetra_CrsMatrix(Copy,*rowmap, *colmap, num_entries) );

      import_ptr import = rcp(new Epetra_Import(*rowmap,rowmap_dist) );

      CHECK_ZERO(gmat->Import(mat,*import,Insert));

      CHECK_ZERO(gmat->FillComplete());
      gmat->SetLabel(mat.Label());

      return gmat;

      }

  // distribute a gathered vector among processors
  vec_ptr MatrixUtils::Scatter(const Epetra_Vector& vec, const Epetra_Map& distmap)
    {
    vec_ptr dist_vec =  rcp(new Epetra_Vector(distmap));
    import_ptr import = rcp(new Epetra_Import(vec.Map(),distmap));
    CHECK_ZERO(dist_vec->Export(vec,*import,Insert));
    return dist_vec;
    }

// workaround for the buggy Trilinos routine with the same name
mat_ptr MatrixUtils::ReplaceRowMap(mat_ptr A,const Epetra_Map& newmap)
   {
   int maxlen = A->MaxNumEntries();
   int len;
   int *ind = new int[maxlen];
   double *val = new double[maxlen];
   int nloc = A->NumMyRows();
   int *row_lengths = new int[nloc];
   for (int i=0;i<nloc;i++) row_lengths[i]=A->NumMyEntries(i);
   mat_ptr tmpmat;
   if (A->HaveColMap())
     {
     tmpmat = rcp(new Epetra_CrsMatrix(Copy,newmap,A->ColMap(), row_lengths) );
     }
   else
     {
     tmpmat = rcp(new Epetra_CrsMatrix(Copy,newmap, row_lengths) );
     }
 
   int rowA,rowNew;
   for (int i=0;i<A->NumMyRows();i++)
      {
      rowA = A->GRID(i);
      rowNew = newmap.GID(i);
      CHECK_ZERO(A->ExtractGlobalRowCopy(rowA,maxlen,len,val,ind));
      CHECK_ZERO(tmpmat->InsertGlobalValues(rowNew, len, val, ind));
      }
   tmpmat->SetLabel(A->Label());
   delete [] ind;
   delete [] val;
   delete [] row_lengths; 
   return tmpmat;
   }

// create an exact copy of a matrix replacing the column map.
// The column maps have to be 'compatible' 
// in the sense that the new ColMap is a subset of the old one.
mat_ptr MatrixUtils::ReplaceColMap(mat_ptr A, const Epetra_Map& newcolmap)
   {
   int maxlen = A->MaxNumEntries();
   int len;
   int *ind = new int[maxlen];
   double *val = new double[maxlen];
   int nloc = A->NumMyRows();
   int *row_lengths = new int[nloc];
   for (int i=0;i<nloc;i++) row_lengths[i]=A->NumMyEntries(i);
   mat_ptr tmpmat;
   tmpmat = rcp(new Epetra_CrsMatrix(Copy,A->RowMap(),
        newcolmap, row_lengths) );
 
   int grid;
   for (int i=0;i<nloc;i++)
      {
      grid = A->GRID(i);
      CHECK_ZERO(A->ExtractGlobalRowCopy(grid,maxlen,len,val,ind));
#ifdef DEBUGGING
//      (*outFile) << "row " << grid << ": ";
//      for (int j=0;j<len;j++) (*outFile) << ind[j] << " ";
//      (*outFile) << std::endl;
#endif
      CHECK_ZERO(tmpmat->InsertGlobalValues(grid, len, val, ind));
      }
   tmpmat->SetLabel(A->Label());
   delete [] ind;
   delete [] val;
   delete [] row_lengths;
   return tmpmat;
   }
 
 
// create an exact copy of a matrix removing the column map.
// This means that row- and column map have to be 'compatible' 
// in the sense that the ColMap is a subset of the RowMap.
// It seems to be required in order to use Ifpack in some cases.
mat_ptr MatrixUtils::RemoveColMap(mat_ptr A)
   {
   int maxlen = A->MaxNumEntries();
   int len;
   int *ind = new int[maxlen];
   double *val = new double[maxlen];
   int nloc = A->NumMyRows();
   int *row_lengths = new int[nloc];
   for (int i=0;i<nloc;i++) row_lengths[i]=A->NumMyEntries(i);
   mat_ptr tmpmat;
   tmpmat = rcp(new Epetra_CrsMatrix(Copy,A->RowMap(),
        row_lengths) );
 
   int grid;
   for (int i=0;i<A->NumMyRows();i++)
      {
      grid = A->GRID(i);
      CHECK_ZERO(A->ExtractGlobalRowCopy(grid,maxlen,len,val,ind));
      CHECK_ZERO(tmpmat->InsertGlobalValues(grid, len, val, ind));
      }
   tmpmat->SetLabel(A->Label());
   delete [] ind;
   delete [] val;
   delete [] row_lengths;
   return tmpmat;
   }
 
 
 
// simultaneously replace row and column map
mat_ptr MatrixUtils::ReplaceBothMaps(mat_ptr A,const Epetra_Map& newmap, 
                   const Epetra_Map& newcolmap)
   {
   DEBUG("Replace Row and Col Map...");
//   DEBVAR(A->RowMap());
//   DEBVAR(A->ColMap());
//   DEBVAR(newmap);
//   DEBVAR(newcolmap);
   int maxlen = A->MaxNumEntries();
   int len;
   int *ind = new int[maxlen];
   double *val = new double[maxlen];
   int nloc = A->NumMyRows();
   int *row_lengths = new int[nloc];
   for (int i=0;i<nloc;i++) row_lengths[i]=A->NumMyEntries(i);
   mat_ptr tmpmat;
   tmpmat = rcp(new Epetra_CrsMatrix(Copy,newmap,newcolmap,row_lengths) );
 
   int rowA,rowNew;
 
   for (int i=0;i<A->NumMyRows();i++)
      {
      rowA = A->GRID(i);
      rowNew = newmap.GID(i);
      CHECK_ZERO(A->ExtractGlobalRowCopy(rowA,maxlen,len,val,ind));
//      for (int j=0;j<len;j++) 
//        {
//        int newind=newcolmap.GID(A->LCID(ind[j]));
//        DEBUG(i<<" ("<<rowA<<"->"<<rowNew<<"), "<<A->LCID(ind[j])<<"("<<ind[j]<<"->"<<newind<<")");
//        ind[j] = newind;
//        }
      CHECK_ZERO(tmpmat->InsertGlobalValues(rowNew, len, val, ind));
      }

   tmpmat->SetLabel(A->Label());
   delete [] ind;
   delete [] val;
   delete [] row_lengths;
   return tmpmat;
   }

//! work-around for 'Solve' bug (not sure it is one, yet)
void MatrixUtils::TriSolve(const Epetra_CrsMatrix& A, const Epetra_Vector& b, Epetra_Vector& x)
  {
#ifdef TESTING
  if (!(A.UpperTriangular()||A.LowerTriangular()))
    ERROR("Matrix doesn't look (block-)triangular enough for TriSolve...",__FILE__,__LINE__);
  if (!A.StorageOptimized())
    ERROR("Matrix has to be StorageOptimized() for TriSolve!",__FILE__,__LINE__);
  if (!b.Map().SameAs(A.RangeMap()))
    ERROR("Rhs vector out of range for TriSolve!",__FILE__,__LINE__);
  if (!x.Map().SameAs(A.DomainMap()))
    ERROR("Sol vector not in domain!",__FILE__,__LINE__);
#endif

//  DEBVAR(b);

  if (A.UpperTriangular())
    {
    DEBUG("Upper Tri Solve with "<<A.Label()<<"...");
    int *begA,*jcoA;
    double *coA;
    CHECK_ZERO(A.ExtractCrsDataPointers(begA,jcoA,coA));
    double sum;
    int diag;
    for (int i=A.NumMyRows()-1;i>=0;i--)
      {
      diag = begA[i];
      sum = 0.0;
      for (int j=diag+1;j<begA[i+1];j++)
        {
//        DEBUG(i<<" "<<jcoA[j]<<" "<<coA[j]);
        sum+=coA[j]*x[jcoA[j]];
        }
//      DEBUG("diag: "<<i<<" "<<jcoA[diag]<<" "<<coA[diag]);
      x[i] = (b[i] - sum)/coA[diag];
      }
    }
  else
    {
    DEBUG("Lower Tri Solve with"<<A.Label()<<"...");
    int *begA,*jcoA;
    double *coA;
    CHECK_ZERO(A.ExtractCrsDataPointers(begA,jcoA,coA));
    double sum;
    int diag;
    for (int i=0;i<A.NumMyRows();i++)
      {
      diag = begA[i+1]-1;
      sum = 0.0;
      for (int j=0;j<diag;j++)
        {
//        DEBUG(i<<" "<<jcoA[j]<<" "<<coA[j]);
        sum+=coA[j]*x[jcoA[j]];
        }
//      DEBUG("diag: "<<i<<" "<<jcoA[diag]<<" "<<coA[diag]);
      x[i] = (b[i] - sum)/coA[diag];
      }
    }
//  DEBVAR(x);
  }//TriSolve


// make A identity matrix
void MatrixUtils::Identity(mat_ptr A)
  {
  double val =1.0;
  int ind;
  A->PutScalar(0.0);
  for (int i=0;i<A->NumMyRows();i++)
    {
    ind = A->GRID(i);
    CHECK_ZERO(A->ReplaceGlobalValues(ind,1,&val,&ind));
    }
  A->SetLabel("Identity");
  CHECK_ZERO(A->FillComplete());
  }




// write CRS matrix to file
void MatrixUtils::DumpMatrix(const Epetra_CrsMatrix& A, const std::string& filename)
  {
  INFO("Matrix with label "<<A.Label()<<" is written to file "<<filename);
  stream_ptr ofs = rcp(new Teuchos::oblackholestream());
  int my_rank = A.Comm().MyPID();
  if (my_rank==0)
    {
    ofs = rcp(new std::ofstream(filename.c_str()));
    }
  *ofs << std::setw(15) << std::setprecision(15);
  *ofs << *(MatrixUtils::Gather(A,0));
  }


void MatrixUtils::DumpMatrixHDF(const Epetra_CrsMatrix& A, 
                                const std::string& filename, 
                                const std::string& groupname,
                                bool new_file)
  {
#ifndef HAVE_HDF5
  ERROR("HDF format can't be stored, recompile with -DHAVE_HDF5",__FILE__,__LINE__);
#else
  bool verbose=true;
  bool success;
  INFO("Matrix with label "<<A.Label()<<" is written to HDF5 file "<<filename<<", group "<<groupname);
  RCP<EpetraExt::HDF5> hdf5 = rcp(new EpetraExt::HDF5(A.Comm()));
try {
  if (new_file)
    { 
    hdf5->Create(filename.c_str());
    }
  else
    {
    hdf5->Open(filename.c_str());
    }
  hdf5->Write(groupname,A);
  hdf5->Close();
} TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose,std::cerr,success);
#endif
  }

// print row matrix
void MatrixUtils::PrintRowMatrix(const Epetra_RowMatrix& A, std::ostream& os)
  {
  DEBUG("Print Row Matrix: "<<A.Label());
  int nrows = A.NumMyRows();
  int ncols = A.NumMyCols();
  int nnz = A.NumMyNonzeros();
  int nrows_g = A.NumGlobalRows();
  int ncols_g = A.NumGlobalCols();
  int nnz_g = A.NumGlobalNonzeros();
  int maxlen = ncols;
  int len;
  int *indices = new int[maxlen];
  double *values = new double[maxlen];
  int grid,gcid;

  os << "Number of Rows: " << nrows;

  if (nrows!=nrows_g) os << " [g"<<nrows_g<<"]";

  os << std::endl;

  os << "Number of Columns: " << ncols;

  if (ncols!=ncols_g) os << " [g"<<ncols_g<<"]";

  os << std::endl;

  os << "Number of Nonzero Entries: " << nnz;

  if (nnz!=nnz_g) os << " [g"<<nnz_g<<"]";


  os << std::endl;

  for (int i=0;i<nrows;i++)
    {
    grid = A.RowMatrixRowMap().GID(i);
    CHECK_ZERO(A.ExtractMyRowCopy(i,maxlen,len,values,indices));
    for (int j=0;j<len;j++)
      {
      gcid = A.RowMatrixColMap().GID(indices[j]);
//      os << A.Comm().MyPID() << "\t";
      os << i;
      if (grid!=i) os << " [g"<< grid <<"]";
      os << "\t";
      os << indices[j];
      if (gcid!=indices[j]) os << " [g"<< gcid <<"]";
      os << "\t";
      os << values[j] << std::endl;
      }
    }
  delete [] indices;
  delete [] values;
  }

mat_ptr MatrixUtils::TripleProduct(bool transA, const Epetra_CrsMatrix& A,
                                          bool transB, const Epetra_CrsMatrix& B,
                                          bool transC, const Epetra_CrsMatrix& C)
  {

    // trans(A) is not available as we prescribe the row-map of A*B, but if it is needed
    // at some point it can be readily implemented
    if(transA) ERROR("This case is not implemented: trans(A)*op(B)*op(C)\n",__FILE__,__LINE__);

    // temp matrix
    mat_ptr AB = rcp(new Epetra_CrsMatrix(Copy,A.RowMap(),A.MaxNumEntries()) );

    DEBUG("compute A*B...");
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(A,transA,B,transB,*AB));

    // result matrix
    mat_ptr ABC = rcp(new Epetra_CrsMatrix(Copy,AB->RowMap(),AB->MaxNumEntries()) );

    DEBUG("compute ABC...");
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(*AB,false,C,transC,*ABC));

    DEBUG("done!");
    return ABC;
    }

void MatrixUtils::TripleProduct(mat_ptr ABC, bool transA, const Epetra_CrsMatrix& A,
                                          bool transB, const Epetra_CrsMatrix& B,
                                          bool transC, const Epetra_CrsMatrix& C)
  {

    // temp matrix
    mat_ptr AB = rcp(new Epetra_CrsMatrix(Copy,ABC->Graph()) );

    DEBUG("compute A*B...");
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(A,transA,B,transB,*AB));


    DEBUG("compute ABC...");
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(*AB,false,C,transC,*ABC));

    DEBUG("done!");
    }

mat_ptr MatrixUtils::MatrixProduct(bool transA, const Epetra_CrsMatrix& A,
                                          bool transB, const Epetra_CrsMatrix& B)
  {

    mat_ptr AB = rcp(new Epetra_CrsMatrix(Copy,A.RowMap(),A.MaxNumEntries()) );

    DEBUG("compute A*B...");
    DEBVAR(transA);
    DEBVAR(A.NumGlobalRows());
    DEBVAR(A.NumGlobalCols());
    DEBVAR(transB);
    DEBVAR(B.NumGlobalRows());
    DEBVAR(B.NumGlobalCols());


#ifdef TESTING
  if (!A.Filled()) ERROR("Matrix A not filled!",__FILE__,__LINE__);
  if (!B.Filled()) ERROR("Matrix B not filled!",__FILE__,__LINE__);
#endif

DEBVAR(A);
DEBVAR(B);

    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(A,transA,B,transB,*AB));

    DEBUG("done!");
    return AB;
    }

void MatrixUtils::MatrixProduct(mat_ptr AB,bool transA, const Epetra_CrsMatrix& A,
                                          bool transB, const Epetra_CrsMatrix& B)
  {

    DEBUG("compute A*B...");
    DEBVAR(transA);
    DEBVAR(A.NumGlobalRows());
    DEBVAR(A.NumGlobalCols());
    DEBVAR(transB);
    DEBVAR(B.NumGlobalRows());
    DEBVAR(B.NumGlobalCols());
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(A,transA,B,transB,*AB));

    DEBUG("done!");
    }


mat_ptr MatrixUtils::ReadThcmMatrix(std::string prefix, const Epetra_Comm& comm,
              const Epetra_Map& rowmap,
              const Epetra_Map& colmap,
              const Epetra_Map* rangemap,
              const Epetra_Map* domainmap)
  {

  if (comm.NumProc()>1)
    {
    // this routine is only intended for sequential debugging, up to now...
    ERROR("Fortran Matrix input is not possible in parallel case!",__FILE__,__LINE__);
    }

  INFO("Read THCM Matrix with label "<<prefix);
  std::string infofilename = prefix+".info";
  std::ifstream infofile(infofilename.c_str());

  int nnz,nrows;
  infofile >> nrows >> nnz;
  infofile.close();

  int *begA = new int[nrows+1];
  int *jcoA = new int[nnz];
  double *coA = new double[nnz];
  int *indices = new int[nrows];
  double *values = new double[nrows];

  read_fortran_array(nrows+1,begA,prefix+".beg");
  read_fortran_array(nnz,jcoA,prefix+".jco");
  read_fortran_array(nnz,coA,prefix+".co");
  int *len = new int[nrows];
  for (int i=0;i<nrows;i++) 
    {
    len[i] = begA[i+1]-begA[i];
    }

mat_ptr A=rcp(new Epetra_CrsMatrix(Copy, rowmap, colmap, len, true));

  // put CSR arrays in Trilinos Jacobian
  for (int i = 0; i<nrows; i++)
    {
    int row = rowmap.GID(i);
    int index = begA[i]; // note that these arrays use 1-based indexing
    int numentries = begA[i+1] - index;
    for (int j = 0; j <  numentries ; j++)
      {
      indices[j] = colmap.GID(jcoA[index-1+j] - 1);
      values[j] = coA[index - 1 + j];
      }
    CHECK_ZERO(A->InsertGlobalValues(row, numentries, values, indices));
    }
  A->SetLabel(prefix.c_str());
  if (rangemap==NULL || domainmap==NULL)
    {
    CHECK_ZERO(A->FillComplete());
    }
  else
    {
    CHECK_ZERO(A->FillComplete(*domainmap,*rangemap));
    }
  return A;
  }

//! private helper function for THCM I/O
void MatrixUtils::read_fortran_array(int n, int* array, std::string filename)
  {
  std::ifstream ifs(filename.c_str());
  for (int i=0;i<n;i++)
    {
    ifs >> array[i];
    }
  ifs.close();
  }

//! private helper function for THCM I/O
void MatrixUtils::read_fortran_array(int n, double* array, std::string filename)
  {
  std::ifstream ifs(filename.c_str());
  for (int i=0;i<n;i++)
    {
    ifs >> array[i];
    }
  ifs.close();
  }

} //TRIOS
