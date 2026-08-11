clear all, close all;

% this gives us the data for the bifurcation diagram.
% We can't use the values from fort.7 for the original run
% because the PsiM_min/max there are taken over the whole domain
% whereas we want them only above 30N.
load 'bif_data.mat';
clear add opts;


LW=2;
FS=12;
footnoteFS=8;
MS=4;


addpath('~/ocean/i-emic/matlab/');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bifurcation diagram                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

npoints = size(PsiM_min,1);
%idx=1:npoints
idx=find(Sv<=1.0);
sper2Sv=0.076;

figure(1), hold on;
%title('Meridional Streamfunction');
%plot(p, PsiM_min, '--', 'LineWidth', LW);
plot(Sv(idx), PsiM_max(idx), 'o', 'LineWidth', LW);
set(gca,'FontSize',FS);
xlabel('Freshwater forcing $\varphi$ [Sv]', 'interpreter','latex');
%ylabel('\Psi_{M,max}');
ylabel('AMOC');

%add lines for other runs
fort7 = load('fort.7.1');
plot(abs(fort7(:,3))*sper2Sv, fort7(:,5),'k-','LineWidth',LW);
%fort7 = load('fort.7.1b');
%plot(abs(fort7(:,3))*sper2Sv, fort7(:,5),'k-','LineWidth',LW);
%fort7 = load('fort.7');
%plot(abs(fort7(:,3))*sper2Sv, fort7(:,5),'k-', 'LineWidth',LW);
