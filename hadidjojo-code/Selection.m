function [P_A1, P_B_mu1, P_B_sig1, MDL1, Gamma1] = Selection(P_A, P_B_mu, P_B_sig, MDL, Gamma, K)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Select K individuals with the lowest MDL

[MDL1,ind] = sort(MDL,'ascend');
sel = ind(1:K);

MDL1 = MDL1(1:K);
P_A1 = P_A(:, sel);
P_B_mu1 = P_B_mu(:,:, sel);
P_B_sig1 = P_B_sig(:,:,:, sel);
Gamma1 = Gamma(sel);




