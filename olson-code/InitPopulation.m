%% Initialize Population %%
 % Author: T.Olson
function P = InitPopulation(data, M, K)
    % Initializes population
    [N, d] = size(data);
    
    % Each member of the population is stored as a struct
    % A component is included if b_i = 1.
    % code: [b_1, ..., b_M]
    % weights: [alpha_1, ..., alpha_M]
    % means: [mu_1, ..., mu_M]
    % covs: [Sigma_1, ..., Sigma_M]
    P = repmat( struct('code', [], 'weights', [], 'means', [], 'covs',...
        []), K, 1); % preallocate
    
    % Initialize using uniform weights, random means, Sigma = 1/sigSQ * I
    % Each member has different number active, from 1...K (requires M > K)
%     sigSQ = 1 / 10 * mean(var(data));
    dataCov = cov(data);
    
    % The i^th member has the first i components enabled, with uniform
    % weights
    for i=1:K
        indices = randperm(N, M); % initial means
        P(i).code = [ones(1, i), zeros(1, M - i)];
        P(i).weights = [repmat(1/i, 1, i), zeros(1, M - i)];
        P(i).means = data(indices, :)';
        P(i).covs = repmat(dataCov, 1, 1, M); % repmat(dataCov, 1, 1, M);
    end
 
end