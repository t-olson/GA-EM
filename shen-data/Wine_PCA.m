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
%{
C = 3; % number of clusters
d = 2;
weights = ones(1,C);  % get the weights
means = ones(C,d);  % calculate sample mean
sigmas = ones(d,d,C); % calculate sample cov
for k=1:C
    Xk = X(label==k);
    weights(k) = length(Xk) / length(X);
    means(k,:) = mean(Xk);
    sigmas(:,:,k) = cov(Xk);
end
%}
ReadMe = ['X is the feature matrix.' ...
          'X contains 178 observations and each observation is 2 dimensional.'...
          'label contains the labels (i.e. the real digit) of that observation'];
save('wine_pca_2', 'X', 'label','ReadMe');
%save('wine_pca_2', 'X', 'label', 'weights', 'means', 'sigmas','ReadMe');
