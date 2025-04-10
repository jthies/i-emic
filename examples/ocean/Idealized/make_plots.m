clear all, close all;
LW=3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Final solution                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('~/ocean/i-emic/matlab/');

% Reading the HDF5 output only works with Matlab:

[n m l la nun xmin xmax ymin ymax hdim x y z xu yv zw landm] = ...
        readfort44('fort.44');

fname='ReferenceSolution.h5';
%fname='State.h5';

opts.readParameters=true;
opts.readFluxes=true;
opts.everything=true;

[sol, add, fluxes, pars] = plot_ocean(fname, opts);

% read from matlab state instead
%load('NorthAtlantic.mat');
plot_ocean2(sol, add, fluxes, pars, opts);

nfig = num_figures();

for i=1:nfig
  saveas(i, sprintf('fig%.2d.png',i));
end


%%%%%%%%%%%%%%%%%%%%%%%%%
% Bifurcation diagram   %
%%%%%%%%%%%%%%%%%%%%%%%%%

dat=load('fort.7');

figure;
title('Bifurcation Diagram');

param = abs(dat(:,3));
% meridional overturning streamfunction
psiM_min = dat(:,4);
psiM_max = dat(:,5);
psiM = psiM_max + psiM_min;

psiB_min = dat(:,6);
psiB_max = dat(:,7);

hold on;
%plot(param, psiB_max, 'b-','LineWidth',LW);
%plot(param, psiB_min, 'b--','LineWidth',LW);
plot(param, psiM_max,'k-', 'LineWidth', LW);
plot(param, psiM_min,'k:', 'LineWidth', LW);
plot(param, psiM,'k-.', 'LineWidth', LW);
xlabel('-par(SPER)');
ylabel('\Psi_{M,max}-\Psi_{M,min}');
legend('\Psi_{B,max}', '-\Psi_{B,min}', '\Psi_{M,max}-\Psi_{M,min}');

saveas(nfig+1, sprintf('bif.png',i));
