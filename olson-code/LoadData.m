%% Load data from file and process for GA-EM %%
 % Author: T.Olson
function [X, means, sigmas, C, N, d, varargout] = LoadData(filename)

load(filename,'X','label');
labelSet = unique(label);
C = length(labelSet);
[N,d] = size(X);

weights = zeros(1,C);
means = zeros(C, d);
sigmas = zeros(d,d,C);

for i=1:C
    myX = X(label(end-N+1:end) == labelSet(i),:);
    weights(i) = size(myX,1)/N;
    means(i,:) = mean(myX);
    sigmas(:,:,i) = cov(myX);
end

% if user requests MDL output, compute it
if (nargout > 6)
    % encode data
    P = struct('code', ones(1,C), 'weights', weights, 'means', means', 'covs',...
        sigmas);
    varargout{1} = MDLencode(P,X); % compute MDL value for sample data
end

end