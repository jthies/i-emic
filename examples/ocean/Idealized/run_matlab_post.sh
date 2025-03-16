#!/bin/bash
#
# Bash script to create plots of the state in FinalConfig.h5
#

if ! test -f fort.44; then
  echo "Missing fort.44 file. This is written at the end of the run_ocean-loca driver."
  echo "You can create it by restarting and setting the number of continuation steps"
  echo "to 0 in loca_params.xml."
  exit 1;
fi

if ! test -f FinalConfig.h5; then
  echo "Missing FinalConfig.h5 file. This is written at the end of the run_ocean-loca driver.\n"
  echo "You can rename any intermediate State_*.h5 file to FinalConfig.h5 to visualize it."
  exit 1;
fi

matlab_exe="matlab -nodesktop -nosplash -softwareopengl"
${matlab_exe} -batch "run('make_plots'); exit;"
