function children = Recombine(P, H, data)
thresh = 0.05; % threshold for annihilating component (should set according to dimension of data)
children = repmat( struct('code', [], 'weights', [], 'means', [], 'covs',...
        []), 2*ceil(H/2), 1); % preallocate for speed

K = size(P, 1);
pairs = zeros(ceil(H/2), 2);
numbers = randperm(K);
for h=1:ceil(H/2)
    i = numbers(h);
    pairs(h, :) = [numbers(i), numbers(mod(i, K) + 1)];
    children(2*h-1:2*h) = Mate(P(pairs(h,:)), data, thresh);
end
children = children(1:H);

end

% combine two parents to get two children
function children = Mate(parents, data, thresh)
M = length(parents(1).code);
m = randi(M); % split position

children = repmat( struct('code', [], 'weights', [], 'means', [], 'covs',...
        []), 1, 2);
    
% copy data assuming uniform weights for new components
children(1).code = [parents(1).code(1:m-1), parents(2).code(m:end)];
children(1).weights = zeros(1, M);
n = length(children(1).code(logical(children(1).code))); % number of components
children(1).weights(logical(children(1).code)) = 1/n;
children(1).means = [parents(1).means(:, 1:m-1), parents(2).means(:, m:end)];
children(1).covs = cat(3, parents(1).covs(:, :, 1:m-1), parents(2).covs(:, :, m:end));

children(2).code = [parents(1).code(m:end), parents(2).code(1:m-1)];
children(2).weights = zeros(1, M);
n = length(children(2).code(logical(children(2).code))); % number of components
children(2).weights(logical(children(2).code)) = 1/n;
children(2).means = [parents(1).means(:, m:end), parents(2).means(:, 1:m-1)];
children(2).covs = cat(3, parents(1).covs(:, :, m:end), parents(2).covs(:, :, 1:m-1));

% update binary code by annihilating components with low probability
children(1).code = UpdateCode(children(1), data, thresh);
children(2).code = UpdateCode(children(2), data, thresh);
end

function code = UpdateCode(child, data, thresh)
code = child.code;
ws = child.weights;
mus = child.means;
sigs = child.covs;

M = length(code);
gamma = zeros(size(data,1),M);
for k=1:M
    if (code(k) == 0)
        continue;
    end
    gamma(:,k) = ws(k) .* mvnpdf(data, mus(:,k)', sigs(:,:,k));
end
gamma(:,logical(code)) = gamma(:,logical(code)) ./ repmat(sum(gamma(:,logical(code)),2), 1, sum(code));
code(sum(gamma, 1) < thresh) = 0;
end