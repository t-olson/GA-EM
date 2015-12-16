%% Compute the MDL value for members of a population %%
 % Author: T.Olson
function MDL = MDLencode(P, data)
[N, d] = size(data);
MDL = zeros(size(P));
for i = 1:size(P,1)
    MDL(i) = -logLikelihood(P(i), data)...
        + sum(P(i).code) * (1 + d + (d * (d + 1)) / 2) / 2 * log(N);
end

end