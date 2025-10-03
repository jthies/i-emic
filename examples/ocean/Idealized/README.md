# Idealized Ocean Basin

This is a simple setup without geometry and with only wind forcing from data.
Salinity and Temperature forcing are constructed synthetically (idealized).
The example starts from no forcing and increases the forcing in all three parameters
step-wise to 1 ("Combined Forcing").

The main purpose of this example is testing and developing solvers.

# Step 1

Start from a 0 solution at 0 forcing, and increase the "Combined Forcing" parameter to get to the "Reference Solution":
We run this with the MUMPS direct solver on four MPI processes here.

```bash
cp thcm_params.xml.step1 thcm_params.xml
cp solver_params.xml.MUMPS solver_params.xml
mpirun -np 4 run_ocean_loca
```

# Step 2: increase freshwater forcing, leading to collapse of the AMOC

**This part is not verified/complete**, for idealized forcing you may need to do this
differently in the ocean model THCM. For an AMOC collapse example, see the NorthAtlantic case.

We will switch to non-restoring salinity forcing and add an increasing, constant (negative)
salinity flux perturbation until the AMOC collapses.

- Create the perturbation mask. It should be of size n x m, with n the number of grid points in the 
  x- (meridional) direction, and m the number of grid points in the y- (zonal) direction. A matlab/octave
  script make_spertm.m to create this mask and store it to an ASCII file is included in this directory.
  The script reads the file fort.44, which is stored by Step 1 above. Copy the result to the i-EMIC data dir, e.g.:
  ```bash
  cp spert.txt ${IEMIC_HOME}/data/mkmask/spertm.id32``
  ```
- In ``thcm_params.xml`` we need to set ``Restoring Salinity Forcing`` to 0, and ``Levitus S`` to 1 to get the mechanism
  to work in THCM. Furthermore, set "Read Salinity Perturbation Mask" to 1, and "Salinity Perturbation Mask File" to "spertm.id32"
  (without the prefix).
- The parameter par(SPER) ("Salinity Perturbation" in the XML file) should go from 0 to -10,
  however, LOCA only allows continuation in the positive direction. To circumvent this,
  we set in ``thcm_params.xml`` the parameter "Continuation Parameter Scaling" to -1.0, and then
  perform continuation to a maximum value of +10.0. In case we want to restart while on a branch that
  goes "backward", we have to change the sign and set the minimum value to -10 in ``loca_params.xml``.
  It is therefore convenient to set both values right away, and only adjust the sign if necessary when restarting.
