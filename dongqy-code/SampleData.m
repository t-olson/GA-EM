
function [data, means, sigmas] = SampleData(d, N, C)

means = 8*rand(C, d) - 4;

sigmas = generateSPDmatrices(C, d);
sigmas = sigmas / (5*max(sigmas(:)));

data = zeros(N, d);
% Generate points with uniform weight in each component
for i=1:C
    data(floor(N/C)*(i-1)+1:floor(N/C)*i,:) = mvnrnd(means(i,:), sigmas(:,:,i), floor(N/C));
end

% Generate all remaining points in one random cluster
remain = N - C*floor(N/C);
i = randi(C);
data(end - remain + 1:end, :) = mvnrnd(means(i,:), sigmas(:,:,i), remain);

% shuffle
data = data(randperm(N), :)';

means = means';
end

function A = generateSPDmatrices(K, n)
% Generate K dense n x n symmetric, positive definite matrices
A = rand(n,n,K); % generate K random n x n matrices

for i=1:K
    % A*A' is symmetric, PSD, add eye(n) to make it PD
    A(:,:,i) = A(:,:,i) * A(:,:,i)' + eye(n);
end
end