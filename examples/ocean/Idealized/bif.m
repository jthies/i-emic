dat=load('fort.7');

LW=3;
FS=16;

npoints = size(dat,1);
idx=1:npoints;
% To only plot the last part:
%idx=npoints-250:npoints;

p = abs(dat(idx,3));

PsiM_min = dat(idx,4);
PsiM_max = dat(idx,5);
PsiB_min = dat(idx,6);
PsiB_max = dat(idx,7);

hold on;
%plot(p, PsiB_min, '--', 'LineWidth', LW);
%plot(p, PsiB_max, ':', 'LineWidth', LW);
plot(p, PsiM_min, '--', 'LineWidth', LW);
plot(p, PsiM_max, '-', 'LineWidth', LW);
plot(p, PsiM_max+PsiM_min, ':', 'LineWidth', LW);
set(gca,'FontSize',FS);
xlabel('Combined forcing');

%legend('\Psi_{B,min}', '\Psi_{B,max}', '\Psi_{M,max}+\Psi_{M,min}', 'Location','NorthEastOutside');
legend('\Psi_{M,max}+\Psi_{M,min}', 'Location','NorthEastOutside');

saveas(1,'bif.png');
