% EECS 545 Final project
% Genetic-based EM Algorithm for Learning Gaussian Mixture Models

% This script evaluate the performance of EM and GA-EM
% and produce figures as illustration

% author: Wenbo Shen (shenwb@umich.edu)

clear, clc
% Define the parameter values
d = 2;                  %   data, dimension
K = 6;                  %   K : number of individual in the population
H = 4;                  %   H : number of offsprings in the new population, assumed to be multiples of 2
R = 7;                  %   R : number of iterations
Mmax = 20;              %   Mmax : maximal number of allowed components
t = 0.95;               %   t : correlation threshold
pm = 0.05;              %   pm : mutation probability

init_mode = 'k-means';     %   Initialization mode : random or k-means
file_name = 'pendigit_kpca_2';  %   which data file to load

% load data
load(file_name);
X = X';
[~, n] = size(X);

labelSet = unique(label);
true_M = length(labelSet);           %   true number of clusters


% run GA-EM a couple of times to find the best one
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
    disp(best_mdl);
end

% use the best result to get identification rate
weight_GA = final_individual.weight(final_individual.Binary == 1);
mu_GA = final_individual.Mu(:, final_individual.Binary == 1);
sigma_GA = final_individual.Sigma(:,:,final_individual.Binary == 1);

% get the correct rate of clustering
rate_GA = Id_rate(weight_GA, mu_GA, sigma_GA, X, label);

% report the result of GA-EM
fprintf('%s \n', file_name);
fprintf('    GA-EM, %s initialization\n', init_mode);
fprintf('        M = %2d\n', final_M);
fprintf('        AvgMDL = %6.5f (+/-) %6.5f\n', ...
                        mean(mdl_all), std(mdl_all));
fprintf('        MinMDL = %6.5f\n', min(mdl_all));
fprintf('        ID rate = %6.5f\n\n', rate_GA);


% run EM a couple of times with different M to find the best one
tol = 1e-6;
final_mdl = 1e10;
mdl_all = zeros(1, Mmax);
for Mk = 1:Mmax
    rng(0);
    individual_EM = Individual(Mk, d, X, init_mode);
    
    while individual_EM.num() ~= Mk
        individual_EM = Individual(Mk, d, X, init_mode);
    end
    
    assert(individual_EM.num() == Mk);
    mdl_EM = individual_EM.mdl;
    individual_EM.EM_gmm(X, 500);
    
    while abs(individual_EM.mdl - mdl_EM)/mdl_EM > tol
        mdl_EM = individual_EM.mdl;
        individual_EM.EM_gmm(X, 100);
    end
    
    mdl_all(Mk) = mdl_EM;
    if mdl_EM < final_mdl
        final_mdl = mdl_EM;
        final_M = Mk;
        final_individual = individual_EM;
    end
    disp(mdl_EM);
end

% use the best result to get identification rate
weight_EM = final_individual.weight(final_individual.Binary == 1);
mu_EM = final_individual.Mu(:, final_individual.Binary == 1);
sigma_EM = final_individual.Sigma(:,:,final_individual.Binary == 1);

% get the correct rate of clustering
rate_EM = Id_rate(weight_EM, mu_EM, sigma_EM, X, label);

% report the result of GA-EM
fprintf('%s \n', file_name);
fprintf('    EM, %s initialization\n', init_mode);
fprintf('        M = %2d\n', final_M);
fprintf('        AvgMDL = %6.5f (+/-) %6.5f\n', ...
                        mean(mdl_all), std(mdl_all));
fprintf('        MinMDL = %6.5f\n', min(mdl_all));
fprintf('        ID rate = %6.5f\n\n', rate_EM);

% plotting section
% define the mesh
dx = (max(X(1, :)) - min(X(1,:)))*0.05;
dy = (max(X(2, :)) - min(X(2,:)))*0.05;
xi = linspace(min(X(1, :))-dx, max(X(1, :))+dx);
yi = linspace(min(X(2, :))-dy, max(X(2, :))+dy);
[xm,ym] = meshgrid(xi,yi);


%{
% plot the true data
figure;
gscatter(X(1,:), X(2,:),label,'brmygck','xo+d^vs', 7);
hold on;
for k=1:true_M
    Xk = X(:, label==labelSet(k));
    mu_true(:,k) = mean(Xk,2);
    sigma_true(:,:,k) = cov(Xk');
end
for i=1:true_M
    pm = mvnpdf( [xm(:), ym(:)], mu_true(:,i)', sigma_true(:,:,i));
    pm = reshape(pm, size(xm));
    contour(xm, ym, pm);
end
title('True Data');

% plot the figure from GA-EM results
figure;
gscatter(X(1,:), X(2,:),label,'brmygck','xo+d^vs', 7);
hold on;
for i=1:length(mu_GA)
    pm = mvnpdf( [xm(:), ym(:)], mu_GA(:,i)', sigma_GA(:,:,i));
    pm = reshape(pm, size(xm));
    contour(xm, ym, pm);
end
title('GA-EM');

% plot the figure from EM results
figure;
gscatter(X(1,:), X(2,:),label,'brmygck','xo+d^vs', 7);
hold on;
for i=1:length(mu_EM)
    pm = mvnpdf( [xm(:), ym(:)], mu_EM(:,i)', sigma_EM(:,:,i));
    pm = reshape(pm, size(xm));
    contour(xm, ym, pm);
end
title('EM');
%}
