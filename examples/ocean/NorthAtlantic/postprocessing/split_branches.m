function idx_sets = split_branches(p)
% given parameter values along a branch (in our case, column 3 of a fort.7 file),
% returns a cell array where each cell contains the index set of a range in which
% the parameter is monotonically increasing or decreasing. The union of the index
% sets is the full range 1:length(p), index sets are overlapping by one index, e.g.
% {[1,2,3,4], [4,5,...,12],[12,13,...29]}

% the result data structure to be filled dynamically
idx_sets = {};

if length(p)==0
     return;
end

dp = sign(diff([2*p(1)-p(2);p]));

idx = find(dp==0)
% allo0w only "+" or "-" branches
dp(idx)=dp(idx-1);

% current set index
s = 1;

idx_sets{1} = [1];

for j=2:length(p);
  if dp(j) ~= dp(j-1)
    idx_sets{s} = [idx_sets{s}, j];
    s=s+1;
    idx_sets{s}=[];
  end
  idx_sets{s} = [idx_sets{s}, j];
end

end
