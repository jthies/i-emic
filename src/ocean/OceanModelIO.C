#include "Utils.H"
#include "THCM.H"
#include "TRIOS_Domain.H"

#include <string>
#include "Teuchos_RCP.hpp"
#include "Epetra_Vector.h"
#include "EpetraExt_HDF5.h"
#include "LOCA_Parameter_Vector.H"

namespace OceanModelIO {

// use fname="current" to get the mask used in tHCM
Utils::MaskStruct getLandMask(std::string const &fname)
{
  Utils::MaskStruct mask;
  auto domain = THCM::Instance().GetDomain();

  mask.local = THCM::Instance().getLandMask(fname);
  mask.global = THCM::Instance().getLandMask();

  // Copy to full global mask tmp
  std::vector<int> tmp(*mask.global);

    // Erase everything but upper 2 layers
    tmp.erase(tmp.begin(), tmp.begin() + tmp.size() -
              ( 2 * (domain->GlobalM()+2) * (domain->GlobalN()+2) ));


    // Create global surface mask rcp
    mask.global_surface = std::make_shared<std::vector<int> >();

    // Put the first layer of the remaining tmp mask
    // in mask.global_surface, without borders.
    for (int j = 1; j != domain->GlobalM()+1; ++j)
        for (int i = 1; i != domain->GlobalN()+1; ++i)
        {
            mask.global_surface->push_back(tmp[j*(domain->GlobalN()+2) + i]);
        }

    assert( (int) mask.global_surface->size() == N_*M_ );

    // Set label
    mask.label = fname;
    return mask;
}

//=====================================================================
void additionalExports(EpetraExt::HDF5 &HDF5, std::string const &filename)
{
    TIMER_START("Ocean: additionalExports");

    const bool saveSalinityFlux_=true;
    const bool saveTemperatureFlux_=true;
    const bool saveMask_=true;

    std::vector<Teuchos::RCP<Epetra_Vector> > fluxes =
        THCM::Instance().getFluxes();

    if (saveSalinityFlux_)
    {
        // Write emip to ocean output file
        INFO("Writing salinity fluxes to " << filename);
        HDF5.Write("SalinityFlux",       *fluxes[ THCM::_Sal  ]);
        HDF5.Write("OceanAtmosSalFlux",  *fluxes[ THCM::_QSOA ]);
        HDF5.Write("OceanSeaIceSalFlux", *fluxes[ THCM::_QSOS ]);
    }

    if (saveTemperatureFlux_)
    {
        // Write heat fluxes to ocean output file. In the coupled case
        // these are dimensional.
        INFO("Writing temperature fluxes to " << filename);
        HDF5.Write("TemperatureFlux",  *fluxes[ THCM::_Temp ]);
        HDF5.Write("ShortwaveFlux",    *fluxes[ THCM::_QSW  ]);
        HDF5.Write("SensibleHeatFlux", *fluxes[ THCM::_QSH  ]);
        HDF5.Write("LatentHeatFlux",   *fluxes[ THCM::_QLH  ]);
        HDF5.Write("SeaIceHeatFlux",   *fluxes[ THCM::_QTOS ]);
    }

    if (saveMask_)
    {
        Utils::MaskStruct landmask = getLandMask("current");
        INFO("Writing distributed and global mask to "  << filename);
        HDF5.Write("MaskLocal", *(landmask.local));

        HDF5.Write("MaskGlobal", "Global", H5T_NATIVE_INT,
                   landmask.global->size(), &(*landmask.global)[0]);

        HDF5.Write("MaskGlobal", "GlobalSize", (int) landmask.global->size());

        HDF5.Write("MaskGlobal", "Surface", H5T_NATIVE_INT,
                   landmask.global_surface->size(), &(*landmask.global_surface)[0]);

        HDF5.Write("MaskGlobal", "Label", landmask.label);
    }
    TIMER_STOP("Ocean: additionalExports");
}



//=============================================================================
int saveStateToFile(std::string const &filename,
        const Epetra_Vector& state,
        const LOCA::ParameterVector& pVector)
{
    INFO("_________________________________________________________");
    INFO("Writing state and parameters to " << filename);

    // Write state, map and continuation parameter
    EpetraExt::HDF5 HDF5(state.Comm());
    HDF5.Create(filename);
    HDF5.Write("State", state);

    // Interface between HDF5 and the parameters,
    // store all the <npar> parameters in an HDF5 file.
    std::string parName;
    double parValue;
    for (int par = 0; par < pVector.length(); ++par)
    {
        parName  = pVector.getLabel(par);
        parValue = pVector[par];
        INFO("   " << parName << " = " << parValue);
        HDF5.Write("Parameters", parName.c_str(), parValue);
    }

    auto domain = THCM::Instance().GetDomain();

    // Write grid information available in domain object
    HDF5.Write("Grid", "n",   domain->GlobalN());
    HDF5.Write("Grid", "m",   domain->GlobalM());
    HDF5.Write("Grid", "l",   domain->GlobalL());
    HDF5.Write("Grid", "nun", domain->Dof());
    HDF5.Write("Grid", "aux", domain->Aux());

    HDF5.Write("Grid", "xmin", domain->Xmin());
    HDF5.Write("Grid", "xmax", domain->Xmax());
    HDF5.Write("Grid", "ymin", domain->Ymin());
    HDF5.Write("Grid", "ymax", domain->Ymax());
    HDF5.Write("Grid", "hdim", domain->Hdim());


    // Write grid arrays
    std::string gridArrays[6] = {"x", "y", "z",
                                 "xu", "yv", "zw"};

    for (int i = 0; i != 6; ++i)
    {
        std::vector<double> array = (*domain->GetGlobalGrid())[i];
        HDF5.Write("Grid", gridArrays[i], H5T_NATIVE_DOUBLE,
                   array.size(), &array[0]);
    }

    additionalExports(HDF5, filename);
    state.Comm().Barrier();

    INFO("_________________________________________________________");
    return 0;
}

}//namespace OceanModelIO
