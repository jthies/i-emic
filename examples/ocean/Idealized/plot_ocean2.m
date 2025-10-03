function plot_ocean2(sol, add, fluxes, pars, opts)
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

    plot_spert  = plot_everything;
    plot_arrows = plot_everything;
    plot_S_slice = plot_everything;
    plot_T_slice = plot_everything;
    plot_rho_slice = plot_everything;
    
    if (isfield(opts, 'spert'))
      plot_spert = opts.spert;
    end
    if (isfield(opts, 'arrows'))
      plot_arrows = opts.arrows;
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
        title('Salinity perturbation mask');
      end
    end
    % arrow plot of surface velocity field
    if (plot_arrows)
      figure;
      plot_mask(summask,x,y); hold on
      U = udim*reshape(u, n, m, l);
      U = U(:,:,l);
      V = udim*reshape(v, n, m, l);
      V = V(:,:,l);
      [Y, X] = meshgrid(RtD*yv(1:m), RtD*xu(1:n));
      quiver(X, Y, U, V); hold on;
      title('Surface velocity field');
      xlabel('Longitude')
      ylabel('Latitude');
    end


    % plot T/S/rho slices at 320E
    ipos = find(abs(xu*RtD-320)<0.01);
    ipos_str = sprintf('x=%4.2f', (x(ipos-1)+x(ipos))*0.5*RtD);

    lamb = pars.LAMB;
    T = T0+reshape(T,n,m,l);
    S = S0+reshape(S,n,m,l);
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

    if (plot_T_slice)
      figure;
      contourf(RtD*y,z*hdim,Tp',15);
      colorbar;
      title(['Temperature at ',ipos_str,'E'])
      xlabel('Latitude')
      ylabel('z (m)')
    end
    if (plot_S_slice)
      figure;
      contourf(RtD*y,z*hdim,Sl',15);
      colorbar;
      title(['Salinity at ',ipos_str,'E'])
      xlabel('Latitude')
      ylabel('z (m)')
    end
    if (plot_rho_slice)
      figure;
      contourf(RtD*y,z*hdim,rho',15);
      colorbar
      title(['Density at ',ipos_str,'E'])
      xlabel('Latitude')
      ylabel('z (m)')
    end
end
