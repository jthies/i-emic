beg  = importdata('atmos_beg.txt');
ico  = importdata('atmos_ico.txt');
jco  = importdata('atmos_jco.txt');

n   = numel(beg)-1;
nnz = beg(end)-1;

ivals = zeros(nnz,1);
jvals = jco;
vals  = ico;

row = 1;
idx = 1;
while row <= n
  for k = beg(row):beg(row+1)-1
	ivals(idx) = row;
	idx        = idx + 1;
  end
  row = row + 1;
end
A = sparse(ivals, jvals, vals, n, n);

rhs    = importdata('atmos_rhs.txt');
frc    = importdata('atmos_frc.txt');
state  = importdata('atmos_state.txt');
sol    = importdata('atmos_sol.txt');
otemp  = importdata('atmos_oceanTemp.txt');
tatm   = importdata('tatm.txt');
