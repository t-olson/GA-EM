function newP = Mutate(P, data, p_m)
newP = P;
M = length(P(1).code);
[~,d] = size(data);
L = d + (d * (d + 1) / 2);
meanRange = repmat(sqrt(var(data))',1,M);
covData = cov(data);

for i = 2:size(P,1) % skip best candidate
    % Randomly flip binary value
    mask = rand(1, M) < p_m;
    newP(i).code(mask) = (~logical(newP(i).code(mask)));
    % reset weights to uniform with new code
    newP(i).weights = zeros(1, M);
    newP(i).weights(logical(newP(i).code)) = 1 / sum(newP(i).code);
    
    % mutate means
    mask = rand(d, M) < p_m / L;
    shifts = (rand(size(mask(mask))) - 1/2) .* meanRange(mask);
    newP(i).means(mask) = newP(i).means(mask) + shifts;
    
    % DON'T MUTATE COVARIANCES!
    % mutate covariances by changing eigenvalues to ensure it remains PD
%     mask = rand(d, 1) < p_m * d / L;
%     for k = 1:M;
%         [V,D] = eig(newP(i).covs(:,:,k));
%         lambda = diag(D);
%         lambda(mask) = max(max(covData(:)) * rand(length(mask(mask)),1), .1);
%         newP(i).covs(:,:,k) = V * diag(sort(lambda)) * V';
%     end
end
end