function [R_A, R_B_mu, R_B_sig] = Mutation(R_A, R_B_mu, R_B_sig, MDL, pm, L)


[Mmax,K] = size(R_A);

%% Sort individual based on MDL
[MDLsort,ind] = sort(MDL, 'ascend');
R_A = R_A(:,ind);
R_B_mu = R_B_mu(:,:,ind);
R_B_sig = R_B_sig(:,:,:,ind);



%% Invert value of R_A

rind = find(rand(K,1) < pm);

for i = 1:length(rind)
    if rind(i) == 1
        continue                % Don't mutate fittest individual
    end
    rpos = randi(Mmax);
    R_A(rpos, rind(i)) = ~R_A(rpos, rind(i));
end



%% Mutate mean value

rind = find(rand(K,1) < pm*L);
for i = 1:length(rind)
    if rind(i) == 1
        continue                % Don't mutate fittest individual
    end
    rpos = randi(Mmax);
    R_B_mu(:, rpos, rind(i)) = R_B_mu(:, rpos, rind(i)) + rand*5 - 1;
end






