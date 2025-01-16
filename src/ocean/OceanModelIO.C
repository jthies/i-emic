#include "OceanModelIO.H"
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

int loadStateFromFile(std::string const &filename,
        Epetra_Vector& state,
        LOCA::ParameterVector& pVector)
{
    INFO("_________________________________________________________");
    bool loadState = true;
    if (loadState)
    {
        INFO("Loading state and parameters from " << filename);
    }
    else
    {
        INFO("Performing only model specific import operations from " << filename);
    }

    // Check whether file exists
    std::ifstream file(filename);
    if (!file)
    {
        WARNING("Can't open " << filename
                << ", continue with trivial state", __FILE__, __LINE__);

        // create trivial state
        state.PutScalar(0.0);
        return 1;
    }
    else file.close();

    // Create HDF5 object
    EpetraExt::HDF5 HDF5(state.Comm());
    Epetra_MultiVector *readState;

    // Open file
    HDF5.Open(filename);

    if (loadState)
    {
        // Check contents
        if (!HDF5.IsContained("State"))
        {
            ERROR("The group <State> is not contained in hdf5 " << filename,
                  __FILE__, __LINE__);
        }

        // Read the state. To be able to restart with different
        // numbers of procs we do not include the Map in the hdf5. The
        // state as read here will have a linear map and we import it
        // into the current domain decomposition.
        HDF5.Read("State", readState);

        if ( readState->GlobalLength() != state.GlobalLength() )
        {
            WARNING("Loading state from differ #procs", __FILE__, __LINE__);
        }

        // Create importer
        // target map: domain StandardMap
        // source map: state with linear map as read by HDF5.Read
        Teuchos::RCP<Epetra_Import> lin2solve =
            Teuchos::rcp(new Epetra_Import(state.Map(),
                                           readState->Map() ));

        // Import state from HDF5 into state_ datamember
        CHECK_ZERO(state.Import(*((*readState)(0)), *lin2solve, Insert));

        delete readState;

        // Interface between HDF5 and the parameters,
        // put all the <npar> parameters back in atmos.
        std::string parName;
        double parValue;

        // Check contents
        if (!HDF5.IsContained("Parameters"))
        {
            ERROR("The group <Parameters> is not contained in hdf5 " << filename,
                  __FILE__, __LINE__);
        }

        for (int par = 0; par < pVector.length(); ++par)
        {
            parName  = pVector.getLabel(par);
            parValue = pVector[par];

            // Read continuation parameter and set them in model
            try
            {
                HDF5.Read("Parameters", parName.c_str(), parValue);
            }
            catch (EpetraExt::Exception &e)
            {
                e.Print();
                continue;
            }

            pVector.setValue(parName, parValue);
            INFO("   " << parName << " = " << parValue);
        }
    }

    additionalImports(HDF5, filename);

    INFO("_________________________________________________________");
    return 0;
}

//=============================================================================
void additionalImports(EpetraExt::HDF5 &HDF5, std::string const &filename)
{
    auto domain = THCM::Instance().GetDomain();

    const bool loadSalinityFlux=false;
    const bool loadTemperatureFlux=false;
    const bool loadMask=false;

    if (loadSalinityFlux)
    {
        INFO("Loading salinity flux from " << filename);

        Epetra_MultiVector *readSalFlux;

        if (!HDF5.IsContained("SalinityFlux"))
        {
            ERROR("The group <SalinityFlux> is not contained in hdf5 " << filename,
                  __FILE__, __LINE__);
        }

        HDF5.Read("SalinityFlux", readSalFlux);

        // Import HDF5 data into THCM. This should not be
        // factorized as we cannot be sure what Map is going to come
        // out of the HDF5 Read call.

        // Create empty salflux vector
        Teuchos::RCP<Epetra_Vector> salflux =
            Teuchos::rcp(new Epetra_Vector(*domain->GetStandardSurfaceMap()));

        Teuchos::RCP<Epetra_Import> lin2solve_surf =
            Teuchos::rcp(new Epetra_Import( salflux->Map(),
                                            readSalFlux->Map() ));

        salflux->Import(*((*readSalFlux)(0)), *lin2solve_surf, Insert);

        // Instruct THCM to set/insert this as the emip in the local model
        //THCM::Instance().setEmip(salflux);
        //TODO: I hink we want to use it as a perturbation, so set spert instead:
        //THCM::Instance().setEmip(salflux, 'P');
        THCM::Instance().setEmip(salflux);

        if (HDF5.IsContained("AdaptedSalinityFlux"))
        {
            INFO(" detected AdaptedSalinityFlux in " << filename);
            Epetra_MultiVector *readAdaptedSalFlux;
            HDF5.Read("AdaptedSalinityFlux", readAdaptedSalFlux);

            assert(readAdaptedSalFlux->Map().SameAs(readSalFlux->Map()));

            Teuchos::RCP<Epetra_Vector> adaptedSalFlux =
                Teuchos::rcp(new Epetra_Vector( salflux->Map() ) );

            adaptedSalFlux->Import( *((*readAdaptedSalFlux)(0)), *lin2solve_surf, Insert);

            delete readAdaptedSalFlux;

            // Let THCM insert the adapted salinity flux
            THCM::Instance().setEmip(adaptedSalFlux, 'A');
        }

        if (HDF5.IsContained("AdaptedSalinityFlux_Mask"))
        {
            INFO(" detected AdaptedSalinityFlux_Mask in " << filename);
            Epetra_MultiVector *readSalFluxPert;
            HDF5.Read("AdaptedSalinityFlux_Mask", readSalFluxPert);

            assert(readSalFluxPert->Map().SameAs(readSalFlux->Map()));

            Teuchos::RCP<Epetra_Vector> salFluxPert =
                Teuchos::rcp(new Epetra_Vector( salflux->Map() ) );

            salFluxPert->Import( *((*readSalFluxPert)(0)), *lin2solve_surf, Insert);

            delete readSalFluxPert;

            // Let THCM insert the salinity flux perturbation mask
            THCM::Instance().setEmip(salFluxPert, 'P');
        }

        delete readSalFlux;

        INFO("Loading salinity flux from " << filename << " done");
    }

    if (loadTemperatureFlux)
    {
        INFO("Loading temperature flux from " << filename);
        if (!HDF5.IsContained("TemperatureFlux"))
        {
            ERROR("The group <SalinityFlux> is not contained in hdf5 " << filename,
                  __FILE__, __LINE__);
        }

        Epetra_MultiVector *readTemFlux;
        HDF5.Read("TemperatureFlux", readTemFlux);

        // This should not be factorized as we cannot be sure what Map
        // is going to come out of the HDF5.Read call.

        // Create empty temflux vector
        Teuchos::RCP<Epetra_Vector> temflux =
            Teuchos::rcp(new Epetra_Vector(*domain->GetStandardSurfaceMap()));

        Teuchos::RCP<Epetra_Import> lin2solve_surf =
            Teuchos::rcp(new Epetra_Import( temflux->Map(),
                                            readTemFlux->Map() ));

        temflux->Import(*((*readTemFlux)(0)), *lin2solve_surf, Insert);

        // Instruct THCM to set/insert this as tatm in the local model
        THCM::Instance().setTatm(temflux);

        delete readTemFlux;

        INFO("Loading temperature flux from " << filename << " done");
    }

    if (loadMask)
    {
        INFO("Loading local mask from " << filename);
        if (!HDF5.IsContained("MaskLocal") ||
            !HDF5.IsContained("MaskGlobal")
            )
        {
            WARNING("The group <Mask*> is not contained in hdf5, continue with standard mask...\n  "
                    << filename, __FILE__, __LINE__);
        }
        else
        {
            //__________________________________________________
            // We begin with the local (distributed) landmask
            Epetra_IntVector *readMask;

            // Obtain current mask to get distributed map with current
            // domain decomposition.
            Teuchos::RCP<Epetra_IntVector> tmpMask =
                THCM::Instance().getLandMask("current");

            // Read mask in hdf5 with distributed map
            HDF5.Read("MaskLocal", readMask);

            Teuchos::RCP<Epetra_Import> lin2dstr =
                Teuchos::rcp(new Epetra_Import( tmpMask->Map(),
                                                readMask->Map() ));

            tmpMask->Import(*readMask, *lin2dstr, Insert);

            delete readMask;

            // Put the new mask in THCM
            THCM::Instance().setLandMask(tmpMask, true);

            //__________________________________________________
            // Get global mask
            int globMaskSize;
            INFO("Loading global mask from " << filename);
            HDF5.Read("MaskGlobal", "GlobalSize", globMaskSize);

            std::shared_ptr<std::vector<int> > globmask =
                std::make_shared<std::vector<int> >(globMaskSize, 0);

            HDF5.Read("MaskGlobal", "Global", H5T_NATIVE_INT,
                      globMaskSize, &(*globmask)[0]);

            // Put the new global mask in THCM
            THCM::Instance().setLandMask(globmask);
        }
    }
}


}//namespace OceanModelIO

