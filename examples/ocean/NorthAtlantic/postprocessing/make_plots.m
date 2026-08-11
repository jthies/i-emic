clear all, close all;

load('bif_data.mat');

only_refsol = false;

% sets FS, MS, LW, etc.
plot_settings;

opts.only_contour = false;
opts.readParameters=true;
% which fields to we want to plot?
opts.mstream=true;
opts.surface_velocity=true;
opts.mixdepth=true;


%addpath('~/ocean/i-emic/matlab/');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bifurcation diagram                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sper2Sv=0.076;

fort7=load('fort.7.1');

p = abs(fort7(:,3))*sper2Sv;
v = fort7(:,5);

idx_sets = split_branches(p);

LS={'k:','k-'};

figure(1), hold on;
for k=1:length(idx_sets)
    idx = idx_sets{k};
    plot(p(idx), v(idx),LS{mod(k,2)+1},'LineWidth',LW);
end
set(gca,'FontSize', FS);
xlabel('Freshwater forcing $\gamma_p$ [Sv]', 'interpreter','latex');
%ylabel('\Psi_{M,max} [Sv]');
ylabel('AMOC [Sv]');

%title('max meridional streamfunction, y>30N, z<-500m')

pdx = 0.005;
pdy=0.2;

states = {{'A',0,'State_A_par28_-0.0.h5',  {0,    0}},
          {'B',1,'State_B1_par28_-6.45.h5',{-2*pdx,-pdy}},
          {'B',2,'State_B2_par28_-6.45.h5',{-pdx, 0}},
          {'B',3,'State_B3_par28_-6.45.h5',{ pdx, pdy}},
          {'C',1,'State_C1_par28_-8.67.h5',{ pdx, pdy}},
          {'C',2,'State_C2_par28_-8.67.h5',{-2.3*pdx, 0}},
          {'C',3,'State_C3_par28_-8.67.h5',{ pdx, pdy}},
          {'C',4,'State_C4_par28_-8.67.h5',{-1.1*pdx,0}},
          {'C',5,'State_C5_par28_-8.67.h5',{ pdx, -pdy}}};

% note: we restrict PsiM_min/max to the area above 30N and a depth>500m
psiM_z_idx = find(zw*hdim < -500);
RtD=180/pi;
psiM_y_idx = find(yv*RtD > 30);

if only_refsol
  nstates = 1;
else
  nstates = length(states);
end

for s=1:nstates
    S = states{s};
    label =  S{1};
    number = S{2};
    fname =  S{3};
    [pdx,pdy] = S{4}{:};
    opts.fig_ctr=2;
    if (number==1)
        if isfield(opts,'mixdepth_diff')
            opts = rmfield(opts, 'mixdepth_diff');
        end
        if isfield(opts,'PSIG_diff')
            opts = rmfield(opts, 'PSIG_diff');
        end
    end
    [sol, add, fluxes, pars] = plot_ocean(fname, opts);
    saveas(2, ['AMOC_diff_',label,'_',num2str(number),'.png']);
    sper = abs(pars.Salinity_Perturbation);
    p = sper*sper2Sv;
    PsiM_max = max(max(add.PSIG(psiM_y_idx,psiM_z_idx)));
    figure(1), hold on;
    % this plots a number and a 'x' or 'o'
    % in the bifurcation diagram. Unfortunately,
    % the value of PsiM_max computed here is slightly
    % off compared to what comes out of THCM in fort.7,
    % so we disable this here and place the markers at the
    % end of this script based on fort.7
    %if mod(number,2)==1 % stable steady state
    %  plot(p,PsiM_max,'ko','MarkerSize',MS-1,'LineWidth',LW);
    %  if (number~=0)
    %    text(p+pdx,PsiM_max+pdy,num2str(number),'VerticalAlignment','middle','HorizontalAlignment','left', 'FontSize',footnoteFS);
    %  end
    %else % unstable steady state
    %  plot(p,PsiM_max,'kx','MarkerSize',MS+1,'LineWidth',2*LW);
    %  if (number~=0)
    %    text(p+pdx,PsiM_max+pdy,num2str(number),'VerticalAlignment','middle','HorizontalAlignment','right', 'FontSize',footnoteFS);
    %  end
    %end
    add=plot_ocean2(sol, add, fluxes, pars, opts);
    if (number==1)
        opts.mixdepth_diff = add.mixdepth;
        opts.PSIG_diff = add.PSIG;
    end
    %figure(3), %title(['Mixed Layer Depth (State ',label,'_',num2str(number),')']);
    saveas(3, ['SurfaceVelocity_',label,'_',num2str(number),'.png']);
    saveas(4, ['MixedLayerDepth_',label,'_',num2str(number),'.png']);
    close(3);
    close(4);
end

figure(1), hold on;

p = abs(fort7(:,3))*sper2Sv;
v = fort7(:,5);


main=gca;

% add sub-axes zoomed-in on region C
% Source: https://nl.mathworks.com/matlabcentral/answers/33779-zooming-a-portion-of-figure-in-a-figure
axB=axes('position',[.25 .25 .25 .25])
box on % put box around new pair of axes
idxB = (p>=0.45) & (p <=0.525); % range of t near perturbation
plot(p(idxB),v(idxB)) % plot on new axes
%axis tight;
set(axB, 'YLim',[9.5, 12.5]);
hold on;

axC=axes('position',[.65 .675 .25 .25])
box on % put box around new pair of axes
idxC = (p>=0.625) & (p <=0.68); % range of t near perturbation
plot(p(idxC),v(idxC)) % plot on new axes
%axis tight;
set(axC, 'YLim',[8.5,10.2]);
hold on;

    text(main, 0, 15,'A', 'VerticalAlignment','middle','HorizontalAlignment','center','FontSize',FS);
    text(main, 6.45*sper2Sv,10,'B','VerticalAlignment','middle','HorizontalAlignment','center','FontSize', FS);
    text(axB, 6.45*sper2Sv,8.75,'B','VerticalAlignment','middle','HorizontalAlignment','center','FontSize', FS);
    text(main, 8.67*sper2Sv,8,'C','VerticalAlignment','middle','HorizontalAlignment','center', 'FontSize', FS);
    text(axC, 8.67*sper2Sv,8,'C','VerticalAlignment','middle','HorizontalAlignment','center', 'FontSize', FS);
    text(main, 0.73,7.5,'D','VerticalAlignment','middle','HorizontalAlignment','center', 'FontSize', FS);

    interp = @(x1, x2, y1, y2, xc) y1 + ((xc - x1).*(y2-y1))./(x2-x1);
    pdy=0;
    marked_points = [0, 6.45, 8.67]*sper2Sv;
    for p0 = marked_points
      ax = main;
      if p0==marked_points(2)
          ax = axB;
      elseif p0==marked_points(3)
          ax = axC;
      end
      number = 1;
      for i=1:length(p)-1
        if (p0>=p(i)) & (p0<p(i+1))
          % stable steady state
          xx = p0;
          yy = interp(p(i),p(i+1),v(i),v(i+1),p0);
          plot(ax, xx,yy,'ko','MarkerSize',MS,'LineWidth',LW);
          if (ax~=main)
            plot(main, xx,yy,'ko','MarkerSize',MS,'LineWidth',LW);
          end
          if (p0~=0.0)
            text(ax, p(i)+pdx,v(i)+pdy,num2str(number),'VerticalAlignment','middle','HorizontalAlignment','left', 'FontSize',footnoteFS);
          end
          number=number+1;
        elseif (p0<=p(i)) & (p0>p(i+1))
          xx = p0;
          yy = interp(p(i+1),p(i),v(i+1),v(i),p0);
          plot(ax, xx, yy,'kx','MarkerSize',MS+1,'LineWidth',2*LW);
          if (ax~=main)
            plot(main, xx, yy,'kx','MarkerSize',MS+1,'LineWidth',2*LW);
          end
          text(ax, p(i)-pdx,v(i)+pdy,num2str(number),'VerticalAlignment','middle','HorizontalAlignment','right', 'FontSize',footnoteFS);
          number=number+1;
        end
      end
    end


saveas(1,'BifurcationDiagram.png');
%saveas(1,'BifurcationDiagram.eps');

