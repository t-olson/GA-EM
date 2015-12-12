function [MDL, loglhood] = Evaluate_MDL(alpha, p, d)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate MDL and log-likelihood lhood
% alpha : cluster weight
% p : N-by-M matrix of gaussian pdf of data n from cluster m
% d : dimension of data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% total probability for each data point
ptotal = sum( bsxfun(@times, p, alpha), 2);

% log-likelihood
loglhood = sum(log(ptotal),1);


% Degrees of freedom
dof = d*(d+1)/2;
[N,M] = size(p);

% Calculate MDL
MDL = -loglhood + M*(dof+1)/2*log(N);









