function [R_A, R_B_mu] = EnforcedMutation(R_A, R_B_mu, Gamma3, tCorr)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

K = size(R_A,2);

for k = 1:K
    C = xcorr(Gamma3{k},0,'coeff');
    n = size(Gamma3{k},2);
    C = reshape(C, n, n);
    C = C - tril(C);
    [row,col] = find(C > tCorr);
    
    for j = 1:length(row)
        if rand < 0.5
            if rand < 0.5
                R_A(row(j),k) = 0;
            else
                R_A(col(j),k) = 0;
            end
        else
            % Generate random mean
            
        end
    end
    
end












