#ifndef OCEAN_MODEL_IO
#define OCEAN_MODEL_IO

#include "Utils.H"
#include "LOCA_Parameter_Vector.H"
#include "Epetra_Vector.h"
#include <string>

// functionality of the i-EMIC 'Model' and 'Ocean' classes,
// in particular for saving and loading THCM states in HDF5 format.
// This is an add-on to the 'OceanModel' class, which is the LOCA
// interface to THCM.
namespace OceanModelIO 
{

// use fname="current" to get the land mask from THCM
Utils::MaskStruct getLandMask(std::string const &fname);
void additionalExports(EpetraExt::HDF5 &HDF5, std::string const &filename);
int saveStateToFile(std::string const &filename,
        const Epetra_Vector& state,
        const LOCA::ParameterVector& pVector);
int loadStateFromFile(std::string const &filename,
        Epetra_Vector& state,
        LOCA::ParameterVector& pVector,
        bool loadTemFlux=false,
        bool loadSalFlux=false,
        bool loadMask=false);

void additionalImports(EpetraExt::HDF5 &HDF5, std::string const &filename,
                       bool loadTemFlux=false,
                       bool loadSalFlux=false,
                       bool loadMask=false);

}//namespace OceanModelIO

#endif
