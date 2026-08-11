function bif(fort7_file, npoints, linecolor)
LW=2;
FS=16;
footnoteFS=8;
MS=4;

fort7 = load(fort7_file);
n=size(fort7,1);
if ~exist('npoints')
    npoints=n;
elseif (npoints<0)
    npoints=n;
end

idx = n-npoints+1:n;

p = abs(fort7(idx,3));
v = fort7(idx,5);

idx_sets = split_branches(p);

LS={'k--','k-'};

hold on;
for k=1:length(idx_sets)
    idx = idx_sets{k};
    h=plot(p(idx), v(idx),LS{mod(k,2)+1},'LineWidth',LW);
    if (exist('linecolor'))
      set(h, 'Color',linecolor);
    end
end
set(gca,'FontSize', FS);

end
