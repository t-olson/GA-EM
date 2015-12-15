% EECS 545 F15 Final project
% Genetic-based EM Algorithm for Learning Gaussian Mixture Models

% This script import the pendigits data from a .txt file
% perform the Kernel Principle Component Analysis (KPCA)
% to reduce the feature dimensions

clear, clc
rng(0);

fileName = 'pendigits_all.txt';
rawData = importdata(fileName);
rawData = rawData(randperm(length(rawData)), :);
Ndigit = 5;

% choose the digits from 0 to Ndigt-1
X = rawData(rawData(:,17)< Ndigit,:);
% label for each entry
label = X(:,17);
% each entry is characterized by 16 features
X = X(:,1:16);
[n, d0] = size(X);

% center and scale data
for k=1:d0
    xtemp = X(:,k);
    xbar = mean(xtemp);
    xstd = std(xtemp);
    xtemp = (xtemp - xbar) / xstd;
    X(:,k) = xtemp;
end

%{
% another way of standardizing data
X = bsxfun(@minus, X, mean(X)); % center the data
sigma = cov(X); % sample covariance matrix
X = X * sqrtm(inv(sigma))'; % standardize the data
%}

% function handle to calculate gaussian kernel
gs = @(x1, x2, s) exp(-dist2(x1, x2)/(2*s*s));

% calculate the train-train kernel matrix
K = gs(X, X, 20);
% center the train-train kernel matrix
A = ones(n, n) / n;
KBar = K - K*A - A*K - A*K*A;

% perform the singular value decomposition
[U, S, ~] = svd(KBar);
S = diag(S);

d = 2;
for k = 1:d
    U(:, d) = U(:,d) / sqrt(S(d));
end
UR = U(:, 1:d);
XR = UR' * K; 
% scatter plot with groups
figure;
gscatter(XR(1,:), XR(2,:),label,'brmygck','xo+d^vs', 7);

% dump the reduced feature and label to a .mat file
X = XR';
save('pendigit_kpca_2', 'X', 'label');