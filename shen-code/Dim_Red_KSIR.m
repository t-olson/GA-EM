% EECS 545 F15 Final project
% Genetic-based EM Algorithm for Learning Gaussian Mixture Models
% This script import the pendigits data from a .txt file
% perform the PCA to reduce the feature dimension

clear, clc

fileName = 'pendigits_all.txt';
rawData = importdata(fileName);
Ndigit = 5;

% choose the digits from 0 to Ndigt-1
X = rawData(rawData(:,17)< Ndigit,:);
X = X(1:1000,:);
% label for each entry
label = X(:,17);
% each entry is characterized by 16 features
X = X(:,1:16);
[n, d] = size(X);

% center and scale data
for k=1:d
    xtemp = X(:,k);
    xbar = mean(xtemp);
    xstd = std(xtemp);
    xtemp = (xtemp - xbar) / xstd;
    X(:,k) = xtemp;
end

% function handle to calculate gaussian kernel
gs = @(x1, x2, s) exp(-dist2(x1, x2)/(2*s*s));

% calculate the kernel matrix
K = gs(X, X, 20);

% center the kernel matrix
A = ones(n, n) / n;
K = K - 2*K*A - A*K*A;

% perform the KSIR dimension reduction
opts.pType = 'c';
opts.H = 5;
[SIR] = KSIR(K, label, 2, 0.1, opts);

% get the reduced data
X2 = zeros(n, 2);
for k=1:n
    X2(k, :)=SIR.C*(K(:,k)-mean(K(:,k)))+SIR.b;
end

% draw the clusters
figure;
gscatter(X2(:,1), X2(:,2),label,[],'.',10);
 
% dump the reduced feature and label to a .mat file
X = X2';
ReadMe = ['X is the feature matrix.' ...
          'X contains 1000 observations and each observation is 2 dimensional.'...
          'label contains the labels (i.e. the real digit) of that observation'];
save('pendigit_ksir_2', 'X', 'label', 'ReadMe');