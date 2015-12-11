% EECS 545 Final project
% Genetic-based EM Algorithm for Learning Gaussian Mixture Models
%   X : d x n training data matrix
%   K : number of individual in the population
%   H : number of offsprings in the new population, assumed to be
%       multiples of 2
%   R : number of iterations
%   Mmax : maximal number of allowed components
%   t : correlation threshold
%   pm : mutation probability
%   mode : Initialization mode, random or k-means

function [best_M_new, best_mdl, best_individual] = GA_EM(X, K, H, R, Mmax, t, pm, mode)

d = size(X, 1);
N = size(X, 2);
best_M = 0;
best_M_new = 1;
best_individual_ind = 0;
best_consecutive = 0;
pop = Population(K, Mmax, d, X, mode);
new_pop = Population(K, Mmax, d, X, mode);

while best_consecutive < 5
    % Step 1 : Perform R EM steps on current population
    pop.EM_run(X, R);
    
    % Step 2 : Evaluate MDL's on current population 
    % (Done in the EM_gmm function)
    
    % Step 3 : Recombination
    children = pop.recombine(H, X);
    
    % Step 4 : Perform R EM steps on children
    children.EM_run(X, R);
    
    % Step 5 : Evaluate MDL's on children
    % (Done in the EM_gmm function)
    
    % Step 6 : Selection K best individual among pop and children
    all_mdl = [arrayfun(@(x) x.Value.mdl, pop), arrayfun(@(x) x.Value.mdl, children)];
    [~,ind] = sort(all_mdl);
    ind = ind(1:K);
    for i = 1:K
        if ind(i) > K
            new_pop(i).Value = Individual(children(ind(i)-K).Value);
        else
            new_pop(i).Value = Individual(pop(ind(i)).Value);
        end
    end
    
    % Step 7 : Determine the best MDL and the best individual
    [~, ind] = min(arrayfun(@(x) x.Value.mdl, new_pop));
    best_individual_ind = ind;
    new_pop(ind).Value.best_ind = true;
    % Set all other individuals'best_ind in new_pop to be false
    for i = 1:K
        if i ~= ind
            new_pop(i).Value.best_ind = false;
        end
    end
    best_M_new = new_pop(ind).Value.num();
    if best_M == best_M_new
        best_consecutive = best_consecutive + 1;
    end
    best_M = best_M_new;
    
    % Step 8 : Perform enforced mutation on new_pop
    new_pop.enforced_mutation_run(X, t);
    
    % Step 9 : Perform mutation on new_pop
    new_pop.mutation_run(X, pm);
    
    % Let pop = new_pop and go to the next iteration
    pop = Population(new_pop);                      % Deep copy
end

best_individual = Individual(new_pop(best_individual_ind).Value);

% Final EM run until convergence on the best_individual selected
tol = 1e-6;
best_mdl = best_individual.mdl;
best_individual.EM_gmm(X, 500);
while abs(best_individual.mdl - best_mdl)/best_mdl > tol
    best_mdl = best_individual.mdl;
    best_individual.EM_gmm(X, 100);
end
best_mdl = best_individual.mdl;

end
