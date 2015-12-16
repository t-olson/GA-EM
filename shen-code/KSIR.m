% This is the implementation of KSIR we found online.
% We didn't use KSIR in the end

function [SIR, vopts] = KSIR(K, Y, d, s, opts)
% Input: 
% K: n X n kernel matrix; K(i,j)=K(x_i,x_j)
% Y: response variable
% d: number of SIR directions
% s: tuning parameter
% opts: structrue containing parameters
%       pType: Type of problem: 'c' for classification
%                               'r' for regression
%       H: number of slices, defalt: 10 for 'r' and class number for 'c'
%       
%
% Output:
% SIR: a structure containing
%      C : row vector of length n contains
%          coefficients for compute sir variates for new data
%      b : center parameter
%  for a given new data, the SIR variate is 
%          C*(K[x_i, x]-mean(K[x_i,x])) + b;
% 
% vopts: structure of parameters used
%
% Wu: 8/22/2007


n=length(K(:,1));
if nargin<5
    opts=[];
end

if ~isfield(opts, 'pType')
    if (~isnumeric(Y))|| (length(unique(Y))<5)
        opts.pType = 'c';
    else
        opts.pType = 'r';
    end
end

J=zeros(n,n);

if opts.pType=='c'
    labels = unique(Y);
    opts.H = length(labels);
    
    for i = 1:opts.H
        ind = find(Y==labels(i));
        J(ind,ind)=1/length(ind);
    end
end

if opts.pType=='r'
    if ~isfield(opts, 'H')
        opts.H = 10;
    end
    [Yvals, YI] = sort(Y);
    Hn = round(n/opts.H);
    for i = 1:opts.H-1
        J(YI((i-1)*Hn+1:i*Hn),YI((i-1)*Hn+1:i*Hn))=1/Hn;
    end
    J(YI((opts.H-1)*Hn+1:n), YI((opts.H-1)*Hn+1:n)) = 1/length(YI((opts.H-1)*Hn+1:n));
end

tic
disp('In KSIR rountine, computing starts.');

Kmean = mean(K);
cK = K - repmat(Kmean,n,1) - repmat(mean(K,2),1,n) + mean(Kmean);
[U, D, V] = svd(.5*(cK+cK'));

disp('SVD Done');

D = diag(D);
Dvals = sort(D,'descend');

if nargin<4 || isempty(s)
    s = .1*Dvals(d);
    if s==0
        warning('parameter d is too large')
    end
end

ind  = find(D>1.0e-15*Dvals(1));
D = D(ind); 
U = U(:,ind);
Ds = diag(D+s);
D = diag(sqrt(D));
[V, L] = eig(D*U'*J*U*D, Ds);

disp('Decomposition done.');

[Lvals, LI] = sort(diag(L),1,'descend');
V = V(:,LI(1:d));
C = U*Ds^(-1/2)*V;

toc;


SIR.C = C';
SIR.b = C'*(mean(Kmean)-Kmean)';
SIR.X = C'*cK;

vopts = opts;
vopts.d = d;
vopts.s = s;
