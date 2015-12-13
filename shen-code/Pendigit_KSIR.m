% EECS 545 F15 Final project
% Genetic-based EM Algorithm for Learning Gaussian Mixture Models

% This script import the pendigits data from a .txt file
% perform the Kernel Sliced Inverse Regression (KSIR)
% to reduce the feature dimension

clear, clc
rng(0);

fileName = 'pendigits_all.txt';
rawData = importdata(fileName);
Ndigit = 5;

% choose the digits from 0 to Ndigt-1
X = rawData(rawData(:,17)< Ndigit,:);
X = X(randperm(length(X)), :);

% seperate data and label
label = X(:,17);
X = X(:,1:16);

% seperate data into training and testing
ntr = 3000;
nts = 2000;
X = X(1:ntr+nts, :);
label = label(1:ntr+nts, :);


% center and scale data
[~, d] = size(X);
% center and scale data
for k=1:d
    xtemp = X(:,k);
    xbar = mean(xtemp);
    xstd = std(xtemp);
    xtemp = (xtemp - xbar) / xstd;
    X(:,k) = xtemp;
end

Xtr = X(1:ntr, :);
Xts = X(ntr+1:end, :);
label_tr = label(1:ntr, :);
label_ts = label(ntr+1:end, :);

% function handle to calculate gaussian kernel
gs = @(x1, x2, s) exp(-dist2(x1, x2)/(2*s*s));

% calculate the train-train kernel matrix
K = gs(Xtr, Xtr, 20);
% center the train-train kernel matrix
A = ones(ntr, ntr) / ntr;
K = K - 2*K*A - A*K*A;

% perform the KSIR on traing data
opts.pType = 'c';
opts.H = 5;
[SIR] = KSIR(K, label_tr, 2, 0.1, opts);

% calculate the KxBar matrix
Kx = gs(Xtr, Xts, 20);
B = ones(ntr, nts) / ntr;
Kx = Kx - A*Kx - K*B + A*K*B;

% reduced dimensionality of test data
X2 = zeros(nts, 2);
for k=1:nts
    X2(k, :)=SIR.C*(Kx(:,k)-mean(Kx(:,k)))+SIR.b;
end

% draw the clusters
figure;
gscatter(X2(:,1), X2(:,2),label_ts,[],'.',10);

% dump the reduced feature and label to a .mat file
X = X2;
save('pendigit_ksir_2', 'X', 'label_ts');
