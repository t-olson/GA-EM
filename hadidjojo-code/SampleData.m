function [data, means, sigmas, varargout] = SampleData(N, C, d)

rng(10)

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

% if user requests MDL output, compute it
if (nargout > 3)
    weights = repmat(floor(N/C)/N, 1, C);
    weights(i) = weights(i) + remain/N;
    % encode data
    P = struct('code', ones(1,C), 'weights', weights, 'means', means', 'covs',...
        sigmas);
    varargout{1} = MDLencode(P,data); % compute MDL value for sample data
end

% shuffle data
data = data(randperm(N), :);
end

function A = generateSPDmatrices(K, n)
% Generate K dense n x n symmetric, positive definite matrices
A = rand(n,n,K); % generate K random n x n matrices

for i=1:K
    % A*A' is symmetric, PSD, add eye(n) to make it PD
    A(:,:,i) = A(:,:,i) * A(:,:,i)' + eye(n);
end
end