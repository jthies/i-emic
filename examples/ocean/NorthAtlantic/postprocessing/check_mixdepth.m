[n,m]=size(add.mixdepth);

surfm = landm(2:n+1,2:m+1,l+1);

for i=1:n
    for j=1:m
        if (isnan(add.mixdepth(i,j)) && surfm(i,j)==0)
            disp(sprintf('Non-land column (%d,%d) has mixdepth=NaN',i,j));
            disp(squeeze(add.T(i,j,l:-1:1).*(1-landm(i+1,j+1,l+1:-1:2))))
        end
    end
end
