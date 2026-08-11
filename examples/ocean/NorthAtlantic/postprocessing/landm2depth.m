function depth=landm2depth(landm, zw, hdim)
    [n,m,l] = size(landm);
    depth=ones(n,m)*hdim;
    for k=1:l-1
      layer = landm(:,:,k);
      depth(layer==1) = zw(k)*hdim;
    end

  depth = depth(2:end-1, 2:end-1);
end
