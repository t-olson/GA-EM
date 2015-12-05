function Pop_prime = EM(Pop, data, R)
% Pop is a K x 1 population of structs, preallocate for speed
Pop_prime = repmat( struct('code', [], 'weights', [], 'means', [], 'covs',...
    []), size(Pop,1), 1);

for i = 1:size(Pop,1)
    newP = runEM(Pop(i), data, R); % Run EM on each member of populatin
    
    % Update new members
    Pop_prime(i).code = newP.code;
    Pop_prime(i).weights = newP.weights;
    Pop_prime(i).means = newP.means;
    Pop_prime(i).covs = newP.covs;
end
end

function newP = runEM(P, data, R)
code=P.code;
ws=P.weights;
mus=P.means;
sigs=P.covs;

delta = 1; % this will be the change in relative log likelihood
eps = 10^-5; % if delta < eps, terminate early
newP = P;

M = length(code);

newWs = ws;
newMus = mus;
newSigs = sigs;
gamma = zeros(size(data, 1), M);

ct = 0;
while (ct < R && delta > eps)
    oldP = newP;
    ct = ct + 1;
    ws = newWs;
    mus = newMus;
    sigs = newSigs;
    
    % E-step (update gamma)
    for k=1:M
        if (code(k) == 0)
            continue;
        end
        gamma(:,k) = ws(k) * mvnpdf(data, mus(:,k)', sigs(:,:,k));
    end
    
    kSum = sum(gamma(:,logical(code)),2);
    gamma(:,logical(code)) = gamma(:,logical(code)) ./ repmat(kSum, 1, sum(code));
    if(sum(isnan(gamma(:)))>0)
        % This was causing problems before, but shouldn't happen anymore.
        error('Error computing gamma, NaN detected');
    end
    
    % M-step (update weights, means, sigma)
    for k=1:M
        if (code(k) == 0)
            continue;
        end
        % weights
        newWs(k) = mean(gamma(:, k));
        
        % means
        newMus(:, k) = data' * gamma(:, k) / sum(gamma(:, k));
        if(sum(isnan(newMus(:,k)))>0)
            error(['err ', num2str(k)]);
        end
        
        % covariances
        centered = bsxfun(@minus, data, newMus(:,k)'); % center data
        newSigs(:,:,k) = centered' * diag(gamma(:,k)) * centered /...
            sum(gamma(:,k))+eye(size(data,2))*(1e-6);
    end
    
    % store updated values
    newP.code = code;
    newP.weights = newWs;
    newP.means = newMus;
    newP.covs = newSigs;
    
    % compute relative change in log likelihood
    delta = abs(1 - logLikelihood(newP, data)/logLikelihood(oldP, data));
end

end