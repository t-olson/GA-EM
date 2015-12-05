function logLike = logLikelihood(P, data)
code=P.code;
ws=P.weights;
mus=P.means;
sigs=P.covs;

N = size(data, 1);
M = size(code, 2);

p = zeros(N,1);
for k=1:M
    if (code(k) == 0)
        continue;
    end
    p = p + ws(k) .* mvnpdf(data, mus(:,k)', sigs(:,:,k));
end
logLike = sum(log(p));
end