function Pop = EM(Pop, data, R)
% Pop is a K x 1 population of structs

for i = 1:size(Pop,1)
    Pop(i) = runEM(Pop(i), data, R); % Run EM on each member of population
end
end

function P = runEM(P, data, R)
code=P.code;
ws=P.weights;
mus=P.means;
sigs=P.covs;

delta = 1; % this will be the change in relative log likelihood
eps = 10^-6; % if delta < eps, terminate early
logLike = Inf; % init log likelihood

M = length(code);
M_true = sum(code);
index = 1:M;
index = index(logical(code));

gamma = zeros(size(data, 1), M_true);

% M-step (update weights, means, sigma)
    function M_step()
        for k=1:M_true
            
            % covariances
            centered = bsxfun(@minus, data, mus(:,index(k))'); % center data
            sigs(:,:,index(k)) = bsxfun(@times, centered, gamma(:,k))' * centered /...
                sum(gamma(:,k));
        end
    end
ct = 0;
while (ct < R && delta > eps)
    ct = ct + 1;
    oldLogLike = logLike;
    
    % E-step (update gamma)
    for k=1:M_true;
        gamma(:,k) = ws(index(k)) * mvnpdf(data, mus(:,index(k))', sigs(:,:,index(k)));
    end
    gamma = gamma ./ repmat(sum(gamma,2), 1, M_true);
    
    for k = 1:M_true
        % weights
        ws(index(k)) = mean(gamma(:, k));
        
        % means
        mus(:, index(k)) = data' * gamma(:, k) / sum(gamma(:, k));
    end
    M_step();
    
    % compute relative change in log likelihood
    logLike = logLikelihood(P, data);
    delta = abs(1 - logLike/oldLogLike);
end

P.weights = ws;
P.means = mus;
P.covs = sigs;
end