% Author: Qiaoyuan Dong

classdef Population
    properties
        Value
    end
    methods
        % Constructor
        %   K : number of individual in the population
        %   Mmax : maximal number of components
        %   d : dimension of trainning data
        %   X : d x n data matrix
        %   mode : Initialization mode, random or k-means
        function obj = Population(K, Mmax, d, X, mode)
            if nargin > 1
                obj(1, K) = Population;
                for i = 1:K
                    if nargin == 5
                        % Random initialization of each individual
                        obj(i).Value = Individual(Mmax, d, X, mode);
                    elseif nargin == 3
                        % Initialize each individual as bare ones with
                        % right sizes
                        obj(i).Value = Individual(Mmax, d);
                    end
                end
            % Deep Copy Constructor (because Individual's are passed by ref)
            %   K : a Population object
            elseif nargin == 1 && isa(K, 'Population')
                size_K = size(K, 2);
                obj(1, size_K) = Population;
                for i = 1:size_K
                    obj(i).Value = Individual(K(i).Value);
                end
            end
        end
        
        % Perform EM_gmm on each individual in the population
        %   !!!Caution : obj changed
        %   obj : Population
        %   X : d x n data matrix
        %   R : number of iterations
        function EM_run(obj, X, R)
            K = size(obj, 2);
            for i = 1:K
                obj(i).Value.EM_gmm(X, R);
            end
        end
        
        % Recombination step : selects two parent individuals randomly from
        % the population and recombines them to form two offsprings, repeat
        % this step H/2 times to generate totally H offsprings
        %   H : number of offsprings in the new population, assumed to be
        %       multiples of 2
        %   X : d x n data matrix
        function pop_children = recombine(pop_parent, H, X)
            Mmax = length(pop_parent(1).Value.Binary);
            d = size(pop_parent(1).Value.Mu, 1);
            K = size(pop_parent, 2);
            pop_children = Population(H, Mmax, d);
            for i = 1:(H/2)
                parent1 = pop_parent(randi([1,K])).Value;
                parent2 = pop_parent(randi([1,K])).Value;
                crossover_pos = randi([1, Mmax]);
                % Swap Binary part
                pop_children(2*i-1).Value.Binary = [parent1.Binary(1:crossover_pos), parent2.Binary(crossover_pos+1:end)];
                pop_children(2*i).Value.Binary = [parent2.Binary(1:crossover_pos), parent1.Binary(crossover_pos+1:end)];
                assert(length(pop_children(2*i-1).Value.Binary) == Mmax);
                assert(length(pop_children(2*i).Value.Binary) == Mmax);
                % Swap Mu part
                pop_children(2*i-1).Value.Mu(:, 1:crossover_pos) = parent1.Mu(:, 1:crossover_pos);
                pop_children(2*i-1).Value.Mu(:, crossover_pos+1:end) = parent2.Mu(:, crossover_pos+1:end);
                pop_children(2*i).Value.Mu(:, 1:crossover_pos) = parent2.Mu(:, 1:crossover_pos);
                pop_children(2*i).Value.Mu(:, crossover_pos+1:end) = parent1.Mu(:, crossover_pos+1:end);
                assert(size(pop_children(2*i-1).Value.Mu, 2) == Mmax);
                assert(size(pop_children(2*i).Value.Mu, 2) == Mmax);
                % Swap Sigma part
                pop_children(2*i-1).Value.Sigma(:, :, 1:crossover_pos) = parent1.Sigma(:, :, 1:crossover_pos);
                pop_children(2*i-1).Value.Sigma(:, :, crossover_pos+1:end) = parent2.Sigma(:, :, crossover_pos+1:end);
                pop_children(2*i).Value.Sigma(:, :, 1:crossover_pos) = parent2.Sigma(:, :, 1:crossover_pos);
                pop_children(2*i).Value.Sigma(:, :, crossover_pos+1:end) = parent1.Sigma(:, :, crossover_pos+1:end);
                assert(size(pop_children(2*i-1).Value.Sigma, 3) == Mmax);
                assert(size(pop_children(2*i).Value.Sigma, 3) == Mmax);
                % Reset the weight in the 2 children
                M1 = pop_children(2*i-1).Value.num();
                M2 = pop_children(2*i).Value.num();
                pop_children(2*i-1).Value.weight(pop_children(2*i-1).Value.Binary == 1) = 1/M1;
                pop_children(2*i-1).Value.weight(pop_children(2*i-1).Value.Binary == 0) = 0;
                pop_children(2*i).Value.weight(pop_children(2*i).Value.Binary == 1) = 1/M2;
                pop_children(2*i).Value.weight(pop_children(2*i).Value.Binary == 0) = 0;
                if M1~=0 assert(abs(sum(pop_children(2*i-1).Value.weight) - 1) < 1e-6); end
                if M2~=0 assert(abs(sum(pop_children(2*i).Value.weight) - 1) < 1e-6); end
                % Calculate the MDL values for the 2 children
                pop_children(2*i-1).Value.mdl = pop_children(2*i-1).Value.MDL(X);
                pop_children(2*i).Value.mdl = pop_children(2*i).Value.MDL(X);
            end
        end
        
        % Perform Enforced Mutation on each individual in the population
        %   !!!Caution : obj changed
        %   obj : Population
        %   X : d x n data matrix
        %   t : correlation threshold
        function enforced_mutation_run(obj, X, t)
            K = size(obj, 2);
            for i = 1:K
                if obj(i).Value.best_ind ~= true            % Elitist doesn't enter
                    obj(i).Value.enforced_mutation(X, t);
                end
            end
        end
        
        % Perform Mutation on each individual in the population
        %   !!!Caution : obj changed
        %   obj : Population
        %   X : d x n data matrix
        %   pm : mutation probability
        function mutation_run(obj, X, pm)
            K = size(obj, 2);
            for i = 1:K
                if obj(i).Value.best_ind ~= true            % Elitist doesn't enter
                    obj(i).Value.mutation(X, pm);
                end
            end
        end
    end
end
