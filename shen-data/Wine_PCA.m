% EECS 545 F15 Final project
% Genetic-based EM Algorithm for Learning Gaussian Mixture Models
% This script import the wine data from a .txt file
% perform the PCA to reduce the feature dimension

clear, clc

fileName = 'wine_data.txt';
X = importdata(fileName);

% label for each entry
label = X(:,1);
% each entry is characterized by 13 features
X = X(:,2:14);
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
gscatter(X(1,:), X(2,:),label,[],'o',10);


% dump the reduced feature and label to a .mat file
X = X';
ReadMe = ['X is the feature matrix.' ...
          'X contains 178 observations and each observation is 2 dimensional.'...
          'label contains the labels (i.e. the real digit) of that observation'];
save('wine_pca_2', 'X', 'label', 'ReadMe');
