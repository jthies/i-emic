# North Atlantic AMOC collapse with LOCA

This is an 0.5 degree resolution model of the AMOC with continental geometry and realistic forcing from data.

## Step 1: Compute the reference solution

Start from a zero solution and perform continuation in "Combined Forcing" (Surface wind, Temperature and Salinity) under restoring conditions.
Produces the input for Step 2 (ReferenceSolution.h5)

## Step 2: Increase freshwater forcing

- Copy the final state of Step 1 to "ReferenceSolution.h5"
- Set the "Continuation Parameter" in `loca_params.xml` to "Salinity Perturbation" (SPERT internally)
- Set the following parameters in `thcm_params.xml`
    - name="Starting Solution File", type="string", value="ReferenceSolution.h5"
    - name="Restoring Salinity Profile", type="bool", value="0" (known as SRES internally)
    - name="Continuation Parameter Scaling": "-1.0" (to go from 0 to negative values, i.e., freshwater rather than salinity forcing)
    - name="Read Salinity Perturbation Mask" type="bool" value="1"
    - name="Salinity Perturbation Mask" type="string" value="spertm.scale_0.5deg"

The mask file contains zeros in the window where the freshwater flux should be increased, adn ones everywhere ese. It should be placed in `i-emic/data/mkmask`.
The applied surface salinity flux is diagnosed from the reference solution and does not have to be provided separately.

