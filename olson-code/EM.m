function Pop = EM(Pop, data, R)
% Pop is a K x 1 population of structs

for i = 1:size(Pop,1)
    Pop(i) = runEM(Pop(i), data, R); % Run EM on each member of populatin
end
end

function P = runEM(P, data, R)
code=P.code;
M = length(code);

logLike = Inf; % dummy log-likelihood
eps = 10^-5; % if relative change in logLikelihood < eps, terminate early

gamma = zeros(size(data, 1), M);

ct = 0;
while (ct < R) % Can also break after E-step if logLikelihood doesn't change
    ct = ct + 1;
    ws=P.weights;
    mus=P.means;
    sigs=P.covs;
    
    % E-step (update gamma)
    for k=1:M
        if (code(k) == 0)
            continue;
        end
        gamma(:,k) = ws(k) * mvnpdf(data, mus(:,k)', sigs(:,:,k));
    end
    
    kSum = sum(gamma(:,logical(code)),2);
    
    oldLogLike = logLike;
    logLike = sum(log(kSum));
    if(abs(1 - logLike/oldLogLike) < eps)
        break;
    end
    
    gamma(:,logical(code)) = gamma(:,logical(code)) ./ repmat(kSum, 1, sum(code));
    
    % M-step (update weights, means, sigma)
    for k=1:M
        if (code(k) == 0)
            continue;
        end
        % weights
        ws(k) = mean(gamma(:, k));
        
        % means
        mus(:, k) = data' * gamma(:, k) / sum(gamma(:, k));
        
        % covariances
        centered = bsxfun(@minus, data, mus(:,k)'); % center data
        sigs(:,:,k) = bsxfun(@times, centered, gamma(:,k))' * centered /...
            sum(gamma(:,k)) + (1e-6)*eye(size(data,2)); % add small stabilizer
    end
    
    % store updated values
    P.weights = ws;
    P.means = mus;
    P.covs = sigs;
end

end