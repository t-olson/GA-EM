function [P_B_mu, P_B_sig, Alpha, MDL, Gamma] = EM_steps(X, P_A, P_B_mu, P_B_sig, nIter, tolLL)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do EM iteration nIter times

if isempty(P_A)
    Alpha = [];
    MDL = [];
    Gamma = [];
    return
end

[Mmax,K] = size(P_A);
[N,d] = size(X);

% Start with uniform alpha
Alpha = bsxfun(@rdivide, P_A, sum(P_A,1));

Gamma = cell(K,1);



%% Main Loop

% Initialize var to store MDL and LL
MDL = zeros(1,K);

for k = 1:K     % for each individual
    % index of active components
    compInd = find(P_A(:,k) == 1);
    
    % Extract mu and sigma with P_A == 1
    mu = P_B_mu(:, compInd, k);
    sig = P_B_sig(:, :, compInd, k);
    
    % Weight alpha
    alpha = Alpha(compInd, k);
    alpha = alpha';                 % alpha is 1-by-ncomp row vector
    
    %%% Initial calculation: calculate gaussian pdf for each data %%%%%%%%%
    % Repeat data matrix X by ncomp times (ncomp = no of components)
    % so we can vectorize call to mvnpdf. (Goal: make N*ncomp-by-d matrix)
    ncomp = size(mu,2);
    X_rep = repmat(X,ncomp,1);
    % Repeat mu (Goal: make N*ncomp-by-1 vector)
    mu_rep = repmat(mu,N,1);
    mu_rep = reshape(mu_rep(:), d, ncomp*N)';        % now mu is row vector
    % Same thing with sigma (Goal: make d-by-d-by-M*ncomp matrix)
    sig_rep = repmat(sig, 1, 1, 1, N);
    sig_rep = permute(sig_rep, [1, 2, 4, 3]);
    sig_rep = reshape(sig_rep, d, d, N*ncomp);
    % Calculate gaussian pdf
    p = mvnpdf(X_rep, mu_rep, sig_rep);
    p = reshape(p, N, ncomp);           % row: datapoints n = 1:N, col: components m = 1:ncomp
    
    gamma = zeros(N,Mmax);
    % Iterate EM for nIter times or until convergence
    for i = 1:nIter        
        
        %%% E-step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        % Calculate gamma (cluster responsibility for each data point)
        p_num = bsxfun(@times, alpha, p);
        p_denom = sum(p_num,2);
        gamma = bsxfun(@rdivide, p_num, p_denom);   % row: datapoints, col: components
        
 
        
        %%% M-step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % update alpha
        sum_gamma_n = sum(gamma,1);
        alpha_new = sum_gamma_n ./ N;
        
        % update mu   
        gamma_perm = permute(gamma, [1 3 2]);       % put gauss components along 3rd dim
        gamma_perm = repmat(gamma_perm, 1, 2, 1);
        X_rep1 = repmat(X, 1, 1, ncomp);
        temp = gamma_perm .* X_rep1;                % gamma_1 * x_i
        mu_num = permute( sum(temp, 1), [3 2 1]);
        mu_new = bsxfun(@rdivide, mu_num, sum_gamma_n');
        
        % update sigma
        temp = permute(mu_new,[3,2,1]);
        X_rep2 = X_rep1 - repmat(temp, N, 1, 1);   % mean-subtracted X_rep
        temp = bsxfun(@times, sqrt(gamma_perm), X_rep2);
        sig_new = zeros(size(sig));
        for j = 1:ncomp
            sig_new(:,:,j) = cov(temp(:,:,j)) ./ sum_gamma_n(j) * (N-1);        % cov normalizes by (N-1) by default
        end
        
        
        %%% Replace old values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        alpha = alpha_new;
        mu = mu_new';
        sig = sig_new;
        
        
        
        %%% Calculate MDL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Repeat mu (Goal: make N*ncomp-by-1 vector)
        mu_rep = repmat(mu,N,1);
        mu_rep = reshape(mu_rep(:), 2, ncomp*N)';        % now mu is row vector
        % Same thing with sigma (Goal: make d-by-d-by-M*ncomp matrix)
        sig_rep = repmat(sig, 1, 1, 1, N);
        sig_rep = permute(sig_rep, [1, 2, 4, 3]);
        sig_rep = reshape(sig_rep, d, d, N*ncomp);
        % Calculate gaussian pdf
        p = mvnpdf(X_rep, mu_rep, sig_rep);
        p = reshape(p, N, ncomp);           % row: datapoints n = 1:N, col: components m = 1:ncomp
        % MDL and LL
        [MDLtemp, LL] = Evaluate_MDL(alpha, p, d);
        MDL(k) = MDLtemp;
        
        
        
        %%% Check termination criterion %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if LL < tolLL
            % terminate if log-likelihood falls below threshold
            p_num = bsxfun(@times, alpha, p);
            p_denom = sum(p_num,2);
            gamma = bsxfun(@rdivide, p_num, p_denom);   % row: datapoints, col: components
            break
        end
    end
    
    %%% Update P_B_mu, P_B_sig, and alpha for this individual %%%%%%%%%%%%%
    P_B_mu(:, compInd, k) = mu;
    P_B_sig(:, :, compInd, k) = sig;
    Alpha(compInd,k) = alpha;
    Gamma{k} = gamma;

end





