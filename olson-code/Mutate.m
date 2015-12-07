function P = Mutate(P, data, p_m)
M = length(P(1).code);
[~,d] = size(data);
L = d + (d * (d + 1) / 2);
minData = repmat(min(data)', 1, M);
maxData = repmat(max(data)', 1, M);
% covData = cov(data);

for i = 2:size(P,1) % skip best candidate
    % Randomly flip binary value
    mask = rand(1, M) < p_m;
    P(i).code(mask) = (~logical(P(i).code(mask)));
    % reset weights to uniform with new code
    P(i).weights = zeros(1, M);
    P(i).weights(logical(P(i).code)) = 1 / sum(P(i).code);
    
    % mutate means
    mask = rand(d, M) < p_m / L;
    P(i).means(mask) = minData(mask) + (maxData(mask)-minData(mask)) .* rand(size(mask(mask)));
    
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