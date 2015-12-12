function [C_A, C_B_mu, C_B_sig] = Recombine(P_A, P_B_mu, P_B_sig, pr)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Recombination for genetic algorithm
% pr = recombination probability
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[Mmax,K] = size(P_A);
Npairs = nchoosek(K,2);
pairind = nchoosek(1:K,2);

% Pick pairs with probability pr
rind = find(rand(Npairs,1)<pr);

% Recombine each pair
for i = 1:length(rind)
    % Figure out index of individual 1 and 2 from rind(i)
    ind1 = pairind(rind(i), 1);
    ind2 = pairind(rind(i), 2);
    
    % pick random crossover pos [1, Mmax] and recombine into children
    rpos = randi(Mmax);
    C_A1 = cat(1, P_A(1:rpos, ind1), P_A(rpos+1:end, ind2));
    C_A2 = cat(1, P_A(1:rpos, ind2), P_A(rpos+1:end, ind1));
    C_B_mu1 = cat(2, P_B_mu(:, 1:rpos, ind1), P_B_mu(:, 1+rpos:end, ind2));
    C_B_mu2 = cat(2, P_B_mu(:, 1:rpos, ind2), P_B_mu(:, 1+rpos:end, ind1));
    C_B_sig1 = cat(3, P_B_sig(:, :, 1:rpos, ind1), P_B_sig(:, :, 1+rpos:end, ind2));
    C_B_sig2 = cat(3, P_B_sig(:, :, 1:rpos, ind2), P_B_sig(:, :, 1+rpos:end, ind1));
    
    % Add new children to population
    P_A = cat(2, P_A, C_A1, C_A2);
    P_B_mu = cat(3, P_B_mu, C_B_mu1, C_B_mu2);
    P_B_sig = cat(4, P_B_sig, C_B_sig1, C_B_sig2);
    
end
    

% Return only children
C_A = P_A(:, K+1:end);
C_B_mu = P_B_mu(:,:, K+1:end);
C_B_sig = P_B_sig(:,:,:, K+1:end);




