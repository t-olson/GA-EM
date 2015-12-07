function P = Enforce(P, data)
t_corr = 0.95; % maximum correlation allowed between components

M = length(P(1).code);
N = size(data,1);
K = size(P,1);
for i = 2:K % skip best candidate
    % store local copies of data
    code = P(i).code;
    n = sum(code); % number of active components
    gamma = zeros(N,M);
    
    % compute gamma
    for k=1:M
        if (code(k) == 0)
            continue;
        end
        gamma(:,k) = P(i).weights(k) .* mvnpdf(data, P(i).means(:,k)', P(i).covs(:,:,k));
    end
    kSum = sum(gamma(:,logical(code)),2);
    gamma(:,logical(code)) = gamma(:,logical(code)) ./ repmat(kSum,1,sum(code));
    
    % compute correlation coefficients for supported components
    Ind = logical(code);
    r = abs(corrcoef(gamma(:,Ind)));
    
    % get set of components to update
    [rows,cols] = find(r > t_corr); % must be highly correlated
    rlist=rows(rows<cols); % only take points with row < col
    clist=cols(rows<cols); % get corresponding cols
    components = rlist; % store 1st component of pair
    % with 50% probability, select the other component instead
    randvals = rand(length(rlist),1);
    components(randvals < .5) = clist(randvals<.5);
    
    % conversions to real indices of components
    realComponents = 1:M;
    realComponents = realComponents(Ind);
    components = unique(components);
    % update forced components (repeats will be skipped because we check
    % the new code each time)
    for t = 1:length(components)
        j = realComponents(components(t));
        if(rand < .5)
            % clear component and set weights to uniform
            n = n - 1;
            P(i).code(j) = 0;
            P(i).weights = zeros(1, M);
            P(i).weights(logical(P(i).code)) = 1 / n;
        else
            % set mean to random data point
            P(i).means(:,j) = data(randi(N),:)';
        end
    end
end
end