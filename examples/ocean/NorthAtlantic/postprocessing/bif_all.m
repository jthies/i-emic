clear all, close all;

% this gives us the data for the bifurcation diagram.
% We can't use the values from fort.7 for the original run
% because the PsiM_min/max there are taken over the whole domain
% whereas we want them only above 30N.

use_new_curve=true;
zoom_in_on = 1;

sper2Sv=0.076;
%sper2Sv=1;

% sets MS, FS, LW etc.
plot_settings;

addpath('~/ocean/i-emic/matlab/');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bifurcation diagram                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fort7 = load('fort.7.1');

for f=1:2
    figure(f), hold on;
    orig_curve{f} = plot(abs(fort7(:,3))*sper2Sv, fort7(:,5), 'k--', 'LineWidth', LW);
    set(gca,'FontSize',FS);
    xlabel('Freshwater forcing $\gamma_p$ [Sv]', 'interpreter','latex');
    %ylabel('$\Psi_{M,max}$ [Sv]', 'interpreter','latex');
    ylabel('AMOC', 'interpreter','latex');
    set(gca, 'XLim',[0, 1/0.076]*sper2Sv);
end

figure(1), hold on;
%title('Dependence of snaking on convective transition steepness $\lambda$');
%% lambda/2
fort7 = load('fort.7.5');
curve2=plot(abs(fort7(:,3))*sper2Sv, fort7(:,5),'k-','LineWidth',LW);
%% new/revised: lambda * 2:
if (use_new_curve)
  fort7 = load('fort.7.6');
  curve3=plot(abs(fort7(:,3))*sper2Sv, fort7(:,5),'k:','LineWidth',LW);
  % original (1, 0.5, 0.2)lambda
  h=legend('ref. params', '$\lambda / 2$', '$\lambda \times 2$');
else
  %% In originally submitted version: lambda/5
  fort7 = load('fort.7.4');
  curve3=plot(abs(fort7(:,3))*sper2Sv, fort7(:,5),'k:','LineWidth',LW);
  % (1, 0.5, 2.0)lambda
  h=legend('ref. params', '$\lambda / 2$', '$\lambda \times 2$');
end
% never used, I think this was lambda/10, not sure.
%fort7 = load('fort.7.3');
%plot(abs(fort7(:,3))*sper2Sv, fort7(:,5),'k:','LineWidth',LW);

set(h,'Interpreter','latex');
set(h,'Location','NorthEast');
set(h,'FontSize', FS);

% add sub-axes zoomed-in on region C
% Source: https://nl.mathworks.com/matlabcentral/answers/33779-zooming-a-portion-of-figure-in-a-figure
if (zoom_in_on>0)
  axC=axes('position',[.175 .175 .38 .38]);
  box on % put box around new pair of axes
  copyobj(orig_curve{1},axC);
  copyobj(curve2,axC);
  copyobj(curve3,axC);
  if (zoom_in_on==1)
    % a) both snaking regimes (B, C)
    if (use_new_curve)
      set(axC, 'XLim',[0.45,0.725]);
      set(axC, 'YLim',[8.0,13]);
    else
      set(axC, 'XLim',[0.5,0.8]);
      set(axC, 'YLim',[8.0,12]);
    end
  elseif (zoom_in_on==2)
    % b) only B
    set(axC, 'XLim',[0.45,0.6]);
    set(axC, 'YLim',[10.0,12.8]);
  elseif (zoom_in_on==3)
    % c) only C
    set(axC, 'XLim',[0.6,0.725]);
    set(axC, 'YLim',[8.5,12]);
  end
end

figure(2), hold on;

%fort7 = load('fort.7.2e');
%plot(abs(fort7(:,3))*sper2Sv, fort7(:,5),'k-', 'LineWidth',LW);
fort7 = load('fort.7.2f');
plot(abs(fort7(:,3))*sper2Sv, fort7(:,5),'k-', 'LineWidth',LW);
fort7 = load('fort.7.2d');
plot(abs(fort7(:,3))*sper2Sv, fort7(:,5),'k:', 'LineWidth',LW);
%fort7 = load('fort.7.2c');
%plot(abs(fort7(:,3))*sper2Sv, fort7(:,5),'k:', 'LineWidth',LW);
%fort7 = load('fort.7.2b');
%plot(abs(fort7(:,3))*sper2Sv, fort7(:,5),'k:', 'LineWidth',LW);
%fort7 = load('fort.7.2');
%plot(abs(fort7(:,3))*sper2Sv, fort7(:,5),'k:', 'LineWidth',LW);

h=legend('ref. params', '$K_H \times 1.2$', '$K_H \times 1.5$');
set(h,'Interpreter','latex');
set(h,'Location','NorthEast');
set(h,'FontSize', FS);

saveas(1, 'bif_SPL1.png')
saveas(2, 'bif_hPe.png')
