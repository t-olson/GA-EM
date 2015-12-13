% EECS 545 Final project
% Genetic-based EM Algorithm for Learning Gaussian Mixture Models

clear, clc
% Define the parameter values
K = 6;                  %   K : number of individual in the population
H = 4;                  %   H : number of offsprings in the new population, assumed to be multiples of 2
R = 7;                  %   R : number of iterations
Mmax = 20;              %   Mmax : maximal number of allowed components
t = 0.95;               %   t : correlation threshold
pm = 0.05;              %   pm : mutation probability

init_mode = {'random'};     % Initialization mode : random or k-means
file_name = {'wine_pca_2'}; % which data file to load
for file_k=1:length(file_name)
    for init_k=1:length(init_mode)
        % load data
        load(file_name{file_k});
        X = X';
        [~, n] = size(X);
        % run GA-EM couple of time and find the best one
        n_rep = 10;
        final_mdl = 1e6;
        mdl_all = zeros(1, n_rep);
        for k = 1:n_rep
            % set the random seed
            rng(k);
            % Run GA_EM on the data
            [best_M, best_mdl, best_individual, ~] = ...
                      GA_EM(X, K, H, R, Mmax, t, pm, init_mode{init_k});
            mdl_all(k) = best_mdl;
            if best_mdl < final_mdl
                final_mdl = best_mdl;
                final_M = best_M;
                final_individual = best_individual;
            end
            disp(best_mdl);
        end
        % use the best result to get identification rate
        mu_GA = final_individual.Mu(:, final_individual.Binary == 1);
        sigma_GA = final_individual.Sigma(:,:,final_individual.Binary == 1);
        weight_GA = final_individual.weight(final_individual.Binary == 1);
        % for each data point calculate the probablity of 
        % belonging to a certain cluster
        pc = ones(final_M, n);
        for k =1:final_M
            pc(k,:) = weight_GA(k)*mvnpdf(X', mu_GA(:, k)', sigma_GA(:,:,k));
        end
        % find the cluster id with highest probability
        [~, dc] = max(pc); 
        % another definition of identification rate
        n_id = 0;   % number of corrected identified points
        for k = 1:final_M
            label_k = label(dc==k);
            label_k_maj = mode(label_k);
            n_id = n_id + length(find(label_k == label_k_maj)); 
        end
        rate = n_id / n;
        % report the final result
        fprintf('%s \n', file_name{file_k});
        fprintf('    %s initialization\n', init_mode{init_k});
        fprintf('        AvgMDL = %6.5f (+/-) %6.5f\n', ...
                        mean(mdl_all), std(mdl_all));
        fprintf('        MinMDL = %6.5f\n', min(mdl_all));
        fprintf('        ID rate = %6.5f\n\n', rate);
    end
end














