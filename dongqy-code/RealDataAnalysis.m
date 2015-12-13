% EECS 545 Final project
% Genetic-based EM Algorithm for Learning Gaussian Mixture Models

% This script evaluate the performance of EM and GA-EM
% and produce figures as illustration

clear, clc
% Define the parameter values
K = 6;                  %   K : number of individual in the population
H = 4;                  %   H : number of offsprings in the new population, assumed to be multiples of 2
R = 7;                  %   R : number of iterations
Mmax = 20;              %   Mmax : maximal number of allowed components
t = 0.95;               %   t : correlation threshold
pm = 0.05;              %   pm : mutation probability

init_mode = 'k-means';     % Initialization mode : random or k-means
file_name = 'wine_pca_2';   % which data file to load

% load data
load(file_name);
X = X';
[~, n] = size(X);

% run GA-EM couple of time to find the best one
n_rep = 10;
final_mdl = 1e10;
mdl_all = zeros(1, n_rep);
for k = 1:n_rep
    % set the random seed
    rng(k);
    % Run GA_EM on the data
    [best_M, best_mdl, best_individual, ~] = ...
               GA_EM(X, K, H, R, Mmax, t, pm, init_mode);
    mdl_all(k) = best_mdl;
    if best_mdl < final_mdl
        final_mdl = best_mdl;
        final_M = best_M;
        final_individual = best_individual;
    end
end

% use the best result to get identification rate
weight_GA = final_individual.weight(final_individual.Binary == 1);
mu_GA = final_individual.Mu(:, final_individual.Binary == 1);
sigma_GA = final_individual.Sigma(:,:,final_individual.Binary == 1);

% get the correct rate of clustering
rate = Id_rate(weight_GA, mu_GA, sigma_GA, X, label);

% report the final result
fprintf('%s \n', file_name);
fprintf('    %s initialization\n', init_mode);
fprintf('        AvgMDL = %6.5f (+/-) %6.5f\n', ...
                        mean(mdl_all), std(mdl_all));
fprintf('        MinMDL = %6.5f\n', min(mdl_all));
fprintf('        ID rate = %6.5f\n\n', rate);


