function X = GenerateGaussMixture(N, alpha, mu, sig)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate gaussian mixture data
% N : no of data to be generated
% alpha : 1-by-ncomp weight vector
% mu : ncomp-by-dim mean vectors
% sig : dim-by-dim-by-ncomp covariance matrices


[ncomp,dim] = size(mu);

% normalize alpha
alpha = alpha ./ sum(alpha,2);

assert(size(sig,3) == ncomp)
assert(size(sig,2) == dim)
assert(size(sig,1) == dim)


X = zeros(N,dim);
rnum = rand(N,1);

flag1 = bsxfun(@le,rnum, cumsum(alpha));
comInd = sum(flag1,2);

MU = mu(comInd,:);
SIG = sig(:,:,comInd);

X = mvnrnd(MU, SIG);


