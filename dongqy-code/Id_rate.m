% EECS 545 Final project
% Genetic-based EM Algorithm for Learning Gaussian Mixture Models

% This script takes the weight, mu, and sigma of a 
% Gaussian mixture model and the true label ids to 
% calculate the rate of correctly clustered data points

function [rate] = Id_rate(weight, mu, sigma, X, label)

% for each data point calculate the probablity of 
% belonging to a certain cluster
M = length(weight);
n = length(label);
pc = ones(M, n);
for k =1:M
    pc(k,:) = weight(k)*mvnpdf(X', mu(:, k)', sigma(:,:,k));
end

% find the cluster id with highest probability
[~, dc] = max(pc); 
% another definition of identification rate
n_id = 0;   % number of corrected identified points
for k = 1:M
    label_k = label(dc==k);
    label_k_maj = mode(label_k);
    n_id = n_id + length(find(label_k == label_k_maj)); 
end
rate = n_id / n;

end