clear all, close all;
LW=3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Final solution                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%addpath('../../i-emic/matlab/');

% Reading the HDF5 output only works with Matlab:

[n m l la nun xmin xmax ymin ymax hdim x y z xu yv zw landm] = ...
        readfort44('fort.44');

fname='State_A_par28_-0.0.h5';

% this generates EPS files
opts.exportfig=false;
opts.readParameters=true;
opts.readFluxes=true;
opts.everything=true;

% scale salinity flux to freshwater flux [mm/year]
opts.scale_salflux = -21.1;

[sol, add, fluxes, pars] = plot_ocean(fname, opts);

% mark the region of freshwater perturbation in the Salinity Flux figure:
figure(7), hold on;
x = [300, 336, 336, 300, 300];
y = [ 54,  54,  66,  66,  54];
plot(x,y,'k-','LineWidth',1);
                       
% read from matlab state instead
%load('NorthAtlantic.mat');
add = plot_ocean2(sol, add, fluxes, pars, opts);

nfig = num_figures();

for i=1:nfig
  saveas(i, sprintf('fig%.2d.png',i));
end


%%%%%%%%%%%%%%%%%%%%%%%%%
% Bifurcation diagram   %
%%%%%%%%%%%%%%%%%%%%%%%%%

dat=load('fort.7');

figure;
%title('Bifurcation Diagram');

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
%ylabel('\Psi_{M,max}-\Psi_{M,min}');
ylabel('AMOC');
legend('\Psi_{B,max}', '-\Psi_{B,min}', '\Psi_{M,max}-\Psi_{M,min}');

%saveas(nfig+1, sprintf('bif.png',i));
