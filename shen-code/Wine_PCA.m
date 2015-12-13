% EECS 545 F15 Final project
% Genetic-based EM Algorithm for Learning Gaussian Mixture Models

% This script import the wine data from a .txt file
% perform the Principle Component Analysis (PCA)
% to reduce the feature dimensions

clear, clc
rng(0);

fileName = 'wine_data.txt';
X = importdata(fileName);
X = X(randperm(length(X)),:);

% label for each entry
label = X(:,1);
% each entry is characterized by 13 features
X = X(:,2:14);
[n, d0] = size(X);

% center and scale data
for k=1:d0
    xtemp = X(:,k);
    xbar = mean(xtemp);
    xstd = std(xtemp);
    xtemp = (xtemp - xbar) / xstd;
    X(:,k) = xtemp;
end

% pca using matlab build-in function
[U, ~, S] = pca(X);
X = X';
d = 2;
U2 = U(:,1:2);
X = U2' * X;
% scatter plot with groups
figure;
gscatter(X(1,:), X(2,:),label,[],'o',10);

% dump the reduced feature and label to a .mat file
X = X';
save('wine_pca_2', 'X', 'label');