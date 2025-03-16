function make_plots(fname)

LW=3; % line width for plots

if ~exist('fname')
    fname='FinalConfig.h5';
end

% helper function to check how many figures are currently open
num_figures = @() length(findobj('type','figure'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Final solution                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../../../matlab/');

% Reading the HDF5 output only works with Matlab:

[n m l la nun xmin xmax ymin ymax hdim x y z xu yv zw landm] = ...
        readfort44('fort.44');

opts.readParameters=true;
opts.readFluxes=false;
opts.everything=false;
opts.mstream=true;
opts.bstream=true;
opts.temperature=true;
opts.salinity=true;

[sol, add, fluxes, pars] = plot_ocean(fname, opts);

% some additional plots
opts.arrows=true;
opts.T_slice=true;
opts.S_slice=true;
opts.rho_slice=true;


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

end
