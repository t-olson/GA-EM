function [B_mu, B_sig, alphaFittest, MDLFittest] =  GA_EM(X)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X = (M,d) data matrix [m-data points, d-dimensional]

%rng(0);
tic;

%% Parameters

R = 3;                  % No of EM iterations performed in Step 1 and 3
K = 5;                  % Population size
Mmax = 10;              % Max number of Gaussian components
minLL = -inf;           % minimum log-likelihood for EM steps
pr = .4;                % recombination probability
pm = .4;                % mutation probability for part A
pm_L = .1;              % probability of mean mutation (part B), relative to pm
tCorr = .8;             % correlation threshold for Enforced Mutation



%% Initialization

[N,d] = size(X);            % M = no of data points, d = dimension

%%% Random Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize K solutions with 1 component centered at ranomly-picked data
% points and identity covariance matrix

P_A = zeros(Mmax, K);                % Part A of encoding (row = components, col = different individuals)
P_B_mu = zeros(d, Mmax, K);            % Part B encoding mean mu
P_B_sig = zeros(d, d, Mmax, K);     % Part B encoding dxd cov matrix Sigma

% Initialize part B with random data points and identity variance
rind = randi(N, Mmax*K, 1);
P_B_mu = (X(rind,:))';
P_B_mu = reshape(P_B_mu, 2, Mmax, K);
P_B_sig = repmat(eye(d), 1, 1, Mmax, K);

% Initialize part A by randomly flipping random bits into 1
rind = randi(Mmax,K,randi(Mmax));
for k = 1:K
    P_A(rind(k,:),k) = 1;
end
%%% End Random Init %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%% Main Loop

cend = 0;
ncompFittest = 0;
generation = 0;

while cend < 5
    generation = generation + 1;
    
    %%% Step 1: perform EM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [P_B_mu, P_B_sig, ~, MDL1, Gamma1] = EM_steps(X, P_A, P_B_mu, P_B_sig, R, minLL);


    %%% Step 2: recombine %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [C_A, C_B_mu, C_B_sig] = Recombine(P_A, P_B_mu, P_B_sig, pr);


    %%% Step 3: perform another EM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [C_B_mu, C_B_sig, ~, MDL2, Gamma2] = EM_steps(X, C_A, C_B_mu, C_B_sig, R, minLL);

      
    %%% Step 4: selection %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Q_A = cat(2, P_A, C_A);
    Q_B_mu = cat(3, P_B_mu, C_B_mu);
    Q_B_sig = cat(4, P_B_sig, C_B_sig);
    MDL = cat(2, MDL1, MDL2);
    Gamma = [Gamma1; Gamma2];
    [R_A, R_B_mu, R_B_sig, MDL3, Gamma3] = Selection(Q_A, Q_B_mu, Q_B_sig, MDL, Gamma, K);

    
    %%% Step 5: Enforced Mutation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [R_A, R_B_mu] = EnforcedMutation(R_A, R_B_mu, Gamma3, tCorr);
    
    
    %%% Step 6: Mutation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [S_A, S_B_mu, S_B_sig] = Mutation(R_A, R_B_mu, R_B_sig, MDL3, pm, pm_L);

    
    %%% Step 7: perform EM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [R_B_mu, R_B_sig, Alpha4, MDL4, Gamma4] = EM_steps(X, S_A, S_B_mu, S_B_sig, 10*R, minLL);

    
    
    % Fittest individual
    [MDLFittest,ind] = min(MDL4);
    
    % No of component of fittest individual
    A = S_A(:,ind);
    mask = A==1;
    
    % Record alpha of fittest individual
    alphaFittest = Alpha4(mask,ind);
    
    B_mu = S_B_mu(:,mask,ind);
    B_sig = S_B_sig(:,:,mask,ind);
    ncompFittest_new = sum(A);
    
    
    if ncompFittest_new == ncompFittest
        cend = cend + 1;
    else
        ncompFittest = ncompFittest_new;
        cend = 0;
    end
    
    txt = ['Generation = ', num2str(generation), ', ncomp = ', num2str(ncompFittest_new), ', cend = ', num2str(cend), ', MDL = ', num2str(MDLFittest)];
    disp(txt)
    
    
end

    









