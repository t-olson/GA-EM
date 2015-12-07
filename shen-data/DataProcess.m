% This script import the pendigits data from a .txt file
% perform the PCA to reduce the feature dimension

clear, clc

fileName = 'pendigits_all.txt';
X = importdata(fileName);
Ndigit = 5;

% choose the digits from 0 to Ndigt-1
X = X(X(:,17)< Ndigit,:);
% label for each entry
label = X(:,17);
% each entry is characterized by 16 features
X = X(:,1:16);

% perform PCA
X = X';
X = bsxfun(@minus, X, mean(X,2));
[U,S,V] = svd(X,'econ');
S = diag(S);
U2 = U(:,1:2);
X = U2'*X;

% scatter plot with groups
gscatter(X(1,:), X(2,:),label,[],'.',2);


