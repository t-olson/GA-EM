% Generate Sample data
%   d : dimension
%   N : number of points to be generated
%   M : number of Gaussian distributions
%   c : (optional) make sure the data is c-separated
%       !!!caution : if c is too big the loop won't stop

function [data, means, sigmas] = SampleData(d, N, M, c)

means = 8*rand(d, M) - 4;

sigmas = generateSPDmatrices(M, d);
sigmas = sigmas / (5*max(sigmas(:)));

restart = true;
if nargin == 4          % if we require the data are c-separated
    while restart
        restart = false;
        for i = 1:M
            for j = i+1:M
                c_sep = norm(means(:,i) - means(:,j)) > c*sqrt(d*max([eigs(sigmas(:,:,i)); eigs(sigmas(:,:,j))]));
                if ~c_sep
                    fprintf('%d and %d : c_sep is %d\n', i, j, c_sep);
                    while ~c_sep
                        % Regenerate component i
                        means(:,i) = (8*rand(1, d) - 4).';
                        sigmas(:,:,i) = generateSPDmatrices(1, d);
                        % Regenerate component j
                        means(:,j) = (8*rand(1, d) - 4).';
                        sigmas(:,:,j) = generateSPDmatrices(1, d);
                        c_sep = norm(means(:,i) - means(:,j)) > c*sqrt(d*max([eigs(sigmas(:,:,i)); eigs(sigmas(:,:,j))]));
                    end
                    restart = true;
                    break;
                end
            end
            if restart
                break;
            end
        end
    end
end

data = zeros(d, N);
% Generate points with uniform weight in each component
for i=1:M
    data(:, floor(N/M)*(i-1)+1:floor(N/M)*i) = mvnrnd(means(:,i), sigmas(:,:,i), floor(N/M))';
end

% Generate all remaining points in one random cluster
remain = N - M*floor(N/M);
i = randi(M);
data(:, end - remain + 1:end) = mvnrnd(means(:,i), sigmas(:,:,i), remain)';

% shuffle
data = data(:, randperm(N));

end

function A = generateSPDmatrices(K, n)
% Generate K dense n x n symmetric, positive definite matrices
A = rand(n,n,K); % generate K random n x n matrices

for i=1:K
    % A*A' is symmetric, PSD, add eye(n) to make it PD
    A(:,:,i) = A(:,:,i) * A(:,:,i)' + eye(n);
end
end