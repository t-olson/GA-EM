function [B_mu, B_sig, alpha, MDL] = Run_EM(X, ncomp, nIter)

rng(0)

%% Initialization
K = 3;
[N,d] = size(X);

P_A = ones(ncomp, K);
% Initialize part B with random data points and identity variance
rind = randi(N, ncomp*K, 1);
P_B_mu = (X(rind,:))';
P_B_mu = reshape(P_B_mu, 2, ncomp, K);
P_B_sig = repmat(eye(d), 1, 1, ncomp, K);


[P_B_mu, P_B_sig, alpha, MDL] = EM_steps(X, P_A, P_B_mu, P_B_sig, nIter, -inf);

[MDL,ind] = min(MDL);
B_mu = P_B_mu(:,:,ind);
B_sig = P_B_sig(:,:,:,ind);
alpha = alpha(:,ind);

txt = ['MDL = ', num2str(MDL)];
disp(txt)


