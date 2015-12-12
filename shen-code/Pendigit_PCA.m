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

% pca using matlab build-in function
[U, ~, S] = pca(X);
X = X';
U2 = U(:,1:2);
X = U2' * X;
% scatter plot with groups
figure;
gscatter(X(1,:), X(2,:),label,[],'.',5);

% dump the reduced feature and label to a .mat file
X = X';
ReadMe = ['X is the feature matrix.' ...
          'X contains 5629 observations and each observation is 2 dimensional.'...
          'label contains the labels (i.e. the real digit) of that observation'];
save('pendigit_pca_2', 'X', 'label', 'ReadMe');
