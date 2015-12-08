function [best, MDL_list] = GA_EM(data, R, M, K, H, p_m)
% Initialization
t = 0;
t_max = 1000; % max number of iterations (this would take a very long time)
MDL_list = zeros(t_max, 1);
OldSize = 0;
c_end = 0;
P = InitPopulation(data, M, K);

while (c_end ~= 5 && t < t_max)
    t = t + 1;
    
    % EM loop on parent population
    P_prime = EM(P, data, R);
    % Compute MDL for updated parents
    MDL = MDLencode(P_prime, data);
    
    % Produce offspring
    P_2prime = Recombine(P_prime, H, data);
    % EM loop on offspring population
    P_3prime = EM(P_2prime, data, R);
    % Compute MDL for updated offspring
    MDL_prime = MDLencode(P_3prime, data);
    
    % Sort combined population according to MDL value and extract top K
    [MDL_list(t), P_4prime] = FitSelect([P_prime; P_3prime],[MDL; MDL_prime], K);
    
    % Store best candidate
    a_min = P(1);
    size_amin = sum(a_min.code);
    
    % If best candidate has different number of components, reset counter
    if (size_amin ~= OldSize)
        c_end = 0;
        OldSize = size_amin;
    else
        c_end = c_end + 1;
    end
    
    % Mutate new population (neither function affects best candidate)
    P_5prime = Enforce(P_4prime, data);
    P = Mutate(P_5prime, data, p_m);
end
% fprintf('Finished after %d iterations\n', t);

t = t + 1;
% Run EM on best candidate until converged
best = EM(P(1), data, Inf);
% compute new MDL value
MDL_list(t) = MDLencode(best, data);

MDL_list = MDL_list(1:t);
end


