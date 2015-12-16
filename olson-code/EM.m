%% EM algorithm acting on a population %%
 % Author: T.Olson
function Pop = EM(Pop, data, R)
% Pop is a K x 1 population of structs

for i = 1:size(Pop,1)
    Pop(i) = runEM(Pop(i), data, R); % Run EM on each member of populatin
end
end

function P = runEM(P, data, R)
code=P.code;
M = sum(code);

logLike = Inf; % dummy log-likelihood
eps = 10^-5; % if relative change in logLikelihood < eps, terminate early

gamma = zeros(size(data, 1), M);

ct = 0;
ws=P.weights(logical(code));
mus=P.means(:,logical(code));
sigs=P.covs(:,:,logical(code));

while (ct < R) % Can also break after E-step if logLikelihood doesn't change
    ct = ct + 1;
    
    % E-step (update gamma)
    for k=1:M
        gamma(:,k) = ws(k) * mvnpdf(data, mus(:,k)', sigs(:,:,k));
    end
    
    kSum = sum(gamma,2);
    
    oldLogLike = logLike;
    logLike = sum(log(kSum));
    if(abs(1 - logLike/oldLogLike) < eps)
        break;
    end
    
    gamma = gamma ./ repmat(kSum, 1, M);
    
    % M-step (update weights, means, sigma)
    for k=1:M
        % weights
        ws(k) = mean(gamma(:, k));
        
        % means
        mus(:, k) = data' * gamma(:, k) / sum(gamma(:, k));
        
        % covariances
        centered = bsxfun(@minus, data, mus(:,k)'); % center data
        sigs(:,:,k) = bsxfun(@times, centered, gamma(:,k))' * centered /...
            sum(gamma(:,k));% add small stabilizer
    end
end
    
% store updated values
P.weights(logical(code)) = ws;
P.means(:,logical(code)) = mus;
P.covs(:,:,logical(code)) = sigs;
end