function add=plot_ocean2(sol, add, fluxes, pars, opts)
%---------------------------------------------------------------------
% PLOTTHCM - Mother script for plotting THCM output
%  usage: plot_ocean(solfile, maskfile, opts)
%
%  Father is M. den Toom, who conceived it 06-11-08
%  Messed up by Erik, 2015/2016/2017 -> t.e.mulder@uu.nl
%---------------------------------------------------------------------

    plot_everything = false;
    if (isfield(opts, 'everything'))
      plot_everything = opts.everything;
    end

    only_contour = false;
    if (isfield(opts, 'only_contour'))
      only_contour = opts.only_contour;
    end

    plot_spert  = plot_everything;
    plot_arrows = plot_everything;
    plot_surface_velocity = plot_everything;
    plot_S_slice = plot_everything;
    plot_T_slice = plot_everything;
    plot_rho_slice = plot_everything;
    plot_mixdepth  = plot_everything;
    mixdepth_diff = [];
    if isfield(opts,'mixdepth_diff')
        mixdepth_diff = opts.mixdepth_diff;
    end
    
    
    if (isfield(opts, 'spert'))
      plot_spert = opts.spert;
    end
    if (isfield(opts, 'arrows'))
      plot_arrows = opts.arrows;
    end
    if (isfield(opts, 'surface_velocity'))
      plot_surface_velocity = opts.surface_velocity;
    end
    if (isfield(opts, 'T_slice'))
      plot_T_slice = opts.T_slice;
    end
    if (isfield(opts, 'S_slice'))
      plot_S_slice = opts.S_slice;
    end
    if (isfield(opts, 'rho_slice'))
      plot_rho_slice = opts.rho_slice;
    end
    if (isfield(opts, 'mixdepth'))
      plot_mixdepth = opts.mixdepth;
    end

    maskfile='fort.44';

    [n m l la nun xmin xmax ymin ymax hdim x y z xu yv zw landm] = ...
        readfort44(maskfile);

    fprintf(1,'----------------------------------------------\n')

    % - DEFINE CONSTANTS - ----------------------------------------------

    udim  = 0.1;       %[m/s]    Velocity scale
    r0dim = 6.4e6;     %[m]      Radius of Earth
    T0    = 15;        %[deg C]  Reference temperature
    S0    = 35;        %[psu]    Reference salinity
    RtD   = 180/pi;    %[-]      Radians to degrees

    c1 = 3.8e-3;
    c2 = 21.87;
    c3 = 265.5;
    c4 = 17.67;
    c5 = 243.5;

    % - READ MASK - -----------------------------------------------------

    surfm      = landm(2:n+1,2:m+1,l+1);  %Only interior surface points
    landm_int  = landm(2:n+1,2:m+1,2:l+1);
    dx         = (xu(n+1)-xu(1))/n;
    dy         = (yv(m+1)-yv(1))/m;
    dz         = (zw(l+1)-zw(1))/l;

    % - Create surface landmask image
    summask = sum(landm_int,3);
    summask = summask / max(max(abs(summask)));
    summask = summask.^3;

    % - EXTRACT SOLUTION COMPONENTS - -----------------------------------
    [u,v,w,p,T,S] = extractsol(sol);

    % --- Create colormaps


    % Salinity Flux perturbation
    if (plot_spert)
      figure; % note: if it is not contained we create an empty plot so the figure numbering doesn't change
      if isfield(add, 'SalinityPerturbation')
        spert = reshape(add.SalinityPerturbation, n, m);
        plot_mask(summask,x,y); hold on
        plot_mask(spert,x,y,'FaceColor','green');
        %title('Salinity perturbation mask');
      end
    end
    % arrow plot of surface velocity field
    if (plot_arrows || plot_surface_velocity)
      U = udim*reshape(u, n, m, l);
      U = U(:,:,l);
      V = udim*reshape(v, n, m, l);
      V = V(:,:,l);
      [Y, X] = meshgrid(RtD*yv(1:m), RtD*xu(1:n));
      if (plot_arrows)
          figure;
          plot_mask(summask,x,y); hold on
          quiver(X, Y, U, V); hold on;
          %title('Surface velocity field');
          xlabel('Longitude')
          ylabel('Latitude');
      end
      if (plot_surface_velocity)
          figure, hold on;
          img = sqrt(U.^2+V.^2);
          %imagesc(RtD*xu,RtD*(yv),img); hold on; set(gca,'ydir','normal');
          if only_contour
              [C,h]=contour(X,Y,img,15,'k-'); set(gca,'ydir','normal');
              h.LevelList=round(h.LevelList,1);
              v = h.LevelList;
              clabel(C,h,v(1:2:end));
          else
              contourf(X,Y,img,15);
              crange = [min(img(:)),max(img(:))];
              cmap = [my_colmap(crange)];
              hold off
              colormap(cmap)
              colorbar
          end
          set(gca,'ydir','normal');
          plot_mask(surfm,x,y);
          %title('Surface velocity field');
          xlabel('Longitude')
          ylabel('Latitude');
      end
    end


    % plot T/S/rho slices at 320E
    ipos = find(abs(xu*RtD-320)<0.01);
    ipos_str = sprintf('x=%d', round((x(ipos-1)+x(ipos))*0.5*RtD));

    lamb = pars.LAMB;
    T = T0+reshape(T,n,m,l);
    S = S0+reshape(S,n,m,l);
    T(landm_int==1) = NaN;
    S(landm_int==1) = NaN;
    Rho = lamb*S - T;

    Tp1 = squeeze(T(ipos-1, :,:));
    Tp2 = squeeze(T(ipos, :,:));
    Tp = (Tp1+Tp2)/2;
    Sl1 = squeeze(S(ipos-1, :,:));
    Sl2 = squeeze(S(ipos, :,:));
    Sl = (Sl1+Sl2)/2;
    rho1 = squeeze(Rho(ipos-1, :,:));
    rho2 = squeeze(Rho(ipos, :,:));
    rho = (rho1+rho2)/2;

    % make dimensional
    rhodim=1024;   % see usr.F90
    alphaT = 1e-4; %      "
    rho = alphaT*rhodim*rho;

    if (plot_T_slice)
      figure;
      contourf(RtD*y,z*hdim,Tp',15);
      colorbar;
      %title(['Temperature at ',ipos_str,'E'])
      xlabel('Latitude')
      ylabel('z (m)')
    end
    if (plot_S_slice)
      figure;
      contourf(RtD*y,z*hdim,Sl',15);
      colorbar;
      %title(['Salinity at ',ipos_str,'E'])
      xlabel('Latitude')
      ylabel('z (m)')
    end
    if (plot_rho_slice)
      figure;
      contourf(RtD*y,z*hdim,rho',15);
      colorbar
      %title(['Density at ',ipos_str,'E'])
      xlabel('Latitude')
      ylabel('z (m)')
    end
    if (plot_mixdepth)
        % -------------------------------------------------------
        % mixdepth(x,y) is the depth at which the density is 1% higher than at the surface
        transform = @(rho_in) alphaT*rhodim*rho_in;
        rho_surf = transform(Rho(:,:,l));
        rho_ref = 1.001*rho_surf;
        %z-index k at which rho(i,j,k) is >= rho_ref(i,j)
        mix_k = zeros(n,m);
        for k=l-1:-1:1
          rho_k = transform(Rho(:,:,k));
          mix_k(mix_k==0 & ( rho_k >= rho_ref))=k;
        end
        skip=find(mix_k==0);
        mix_k(skip) = 1;
        zk=z(mix_k);
        zkp1=z(mix_k+1);
        rho_k = zeros(n,m);
        rho_kp1 = zeros(n,m);
        for i=1:n
          for j=1:m
            k = mix_k(i,j);
            rho_k(i,j)   = transform(Rho(i,j,k));
            rho_kp1(i,j) = transform(Rho(i,j,k+1));
          end
        end

        % linear interpolation between grid cells at depth zk and zkp1 to find mixdepth(:,:)
        mixdepth = hdim*(zk + ((rho_ref - rho_k).*(zkp1-zk))./(rho_kp1-rho_k));
        %mixdepth = hdim*zk;

        mixdepth(skip) = NaN;
        add.rho_ref = rho_ref;
        add.Rho = Rho;
        add.mixdepth=mixdepth;

        topo_depth = landm2depth(landm, zw, hdim);
          mixdepth(isnan(mixdepth)) = -hdim;
        mixdepth = max(topo_depth, mixdepth);

        if ~isempty(mixdepth_diff)
          mixdepth_diff(isnan(mixdepth_diff)) = -hdim;
          mixdepth_diff = max(mixdepth_diff, topo_depth);
          img = (mixdepth - mixdepth_diff)';
        else
          deep = zeros(size(mixdepth));
          deep(mixdepth==NaN)==1;
          img  = mixdepth';
        end

        figure;

        if (~only_contour)
            h=imagesc(RtD*x,RtD*(y),img); hold on;
            %surf(RtD*x,RtD*y, deep');
            %set(h, 'AlphaData', surfm');
        end

        [C,h] = contour(RtD*x,RtD*(y),img,15,'k-');
        %plot_mask(summask,x,y); hold on
        plot_mask(surfm,x,y); hold on

        if only_contour
            h.LevelList=round(h.LevelList,1);
            v = h.LevelList;
            clabel(C,h,v(1:2:end));
        else
            if ~isempty(mixdepth_diff)
                crange = [min(img(:)),max(img(:))];
                %cmean = mean(img(~isnan(img)));
                cmean = 0.0;
                cmap = [my_colmap(crange,cmean)];
            else
              if isfield(add, 'mixdepth_cmap')
                cmap = add.mixdepth_cmap;
              else
                crange = [min(img(:)), max(img(:))];
                %crange = [-4000, 0];
                cmean = mean(img(~isnan(img)));
                cmap = [my_colmap(crange,cmean)];
                add.mixdepth_cmap = cmap;
              end
            end
            colormap(cmap);
            colorbar;
        end
        set(gca,'ydir','normal')
        if ~isempty(mixdepth_diff)
             %title('Mixed Layer Depth (diff)', 'interpreter', 'none');
        else
             %title('Mixed Layer Depth', 'interpreter', 'none');
        end
        xlabel('Longitude');
        ylabel('Latitude');
    end        
end
