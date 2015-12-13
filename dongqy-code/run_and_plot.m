% EECS 545 Final project
% Genetic-based EM Algorithm for Learning Gaussian Mixture Models

% This is the main script to generate data, run GA_EM and plot

rng(3);
% Load/Generate data from true_M 2D distributions
gen_data = false;
true_M = 3;
d = 2;
if gen_data
    N = 300*true_M;
    [X, true_mu, true_sigma] = SampleData(d, N, true_M, 1.2);
end

% Run GA_EM on the data
init_mode = 'k-means';  % Initialization mode : random or k-means
K = 6;                  %   K : number of individual in the population
H = 4;                  %   H : number of offsprings in the new population, assumed to be multiples of 2
R = 3;                  %   R : number of iterations
Mmax = 15;              %   Mmax : maximal number of allowed components
t = 0.95;               %   t : correlation threshold
pm = 0.02;              %   pm : mutation probability
[best_M, best_mdl, best_individual, ~] = GA_EM(X, K, H, R, Mmax, t, pm, init_mode);
mu = best_individual.Mu(:, best_individual.Binary == 1);
sigma = best_individual.Sigma(:,:,best_individual.Binary == 1);

% Run EM on the data with the right number of clusters : true_M
tol = 1e-6;
individual_EM = Individual(true_M, d, X, init_mode);
while individual_EM.num() ~= true_M
    individual_EM = Individual(true_M, d, X, init_mode);
end
assert(individual_EM.num() == true_M);
mdl_EM = individual_EM.mdl;
individual_EM.EM_gmm(X, 500);
while abs(individual_EM.mdl - mdl_EM)/mdl_EM > tol
    mdl_EM = individual_EM.mdl;
    individual_EM.EM_gmm(X, 100);
end

% Make plots
figure
%%%%%% GA_EM %%%%%%
% subplot(2, 2, 1);
% Display a scatter plot of the original data
scatter(X(1,:),X(2,:),'b', 'LineWidth', 1.5)
hold on

set(gcf,'color','white') % White background for the figure.

for k = 1:best_M
    plot(mu(1,k), mu(2,k), 'kx');
end

% First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
% the input values over the grid.
gridSize = 100;
lower_bound = min(min(X,[],2));
upper_bound = max(max(X,[],2));
u = linspace(lower_bound, upper_bound, gridSize);
[A B] = meshgrid(u, u);
gridX = [A(:), B(:)];

% Calculate the Gaussian response for every value in the grid.
z = cell(1,best_M);
for k = 1:best_M
    z{k} = mvnpdf(gridX, mu(:, k)', sigma(:,:,k));
end

% Reshape the responses back into a 2D grid to be plotted with contour.
Z = cell(1, best_M);
for k = 1:best_M
    Z{k} = reshape(z{k}, gridSize, gridSize);
end

% Plot the contour lines to show the pdf over the data.
for k = 1:best_M
    [C, h] = contour(u, u, Z{k}, 5, '--', 'LineWidth', 2);
    caxis([-Inf, Inf]);
end
axis([lower_bound upper_bound lower_bound upper_bound])
set(gca,'FontSize',12, 'FontWeight', 'bold');
title(sprintf('Original Data and Estimated GA-EM PDFs with MDL = %e', best_mdl), 'FontSize', 16, 'FontWeight', 'bold');

hold off

%%%%%% True configuration %%%%%%
if gen_data
    true_individual = Individual(true_M, d);
    true_individual.Binary = ones(1, true_M);
    true_individual.weight = ones(1, true_M)/true_M;
    true_individual.Mu = true_mu;
    true_individual.Sigma = true_sigma;
    true_individual.mdl = true_individual.MDL(X);
    
    figure
%     subplot(2, 2, 3);
    % Display a scatter plot of the original data
    scatter(X(1,:),X(2,:),'b', 'LineWidth', 1.5)
    hold on

    set(gcf,'color','white') % White background for the figure.

    for k = 1:true_M
        plot(true_mu(1,k), true_mu(2,k), 'kx');
    end

    % Calculate the Gaussian response for every value in the grid.
    z_true = cell(1,true_M);
    for k = 1:true_M
        z_true{k} = mvnpdf(gridX, true_mu(:, k)', true_sigma(:,:,k));
    end

    % Reshape the responses back into a 2D grid to be plotted with contour.
    Z_true = cell(1, true_M);
    for k = 1:true_M
        Z_true{k} = reshape(z_true{k}, gridSize, gridSize);
    end

    % Plot the contour lines to show the pdf over the data.
    for k = 1:true_M
        [C, h] = contour(u, u, Z_true{k}, 5, 'LineWidth', 2);
        caxis([-Inf, Inf]);
    end
    axis([lower_bound upper_bound lower_bound upper_bound])
    set(gca,'FontSize',12, 'FontWeight', 'bold');
    title(sprintf('Original Data and true PDFs with MDL = %e', true_individual.mdl), 'FontSize', 16, 'FontWeight', 'bold');

    hold off
end

%%%%%% EM with the right # components %%%%%%
mu_EM = individual_EM.Mu;
sigma_EM = individual_EM.Sigma;
mdl_EM = individual_EM.mdl;

figure
% subplot(2, 2, 4);
% Display a scatter plot of the original data
scatter(X(1,:),X(2,:),'b', 'LineWidth', 1.5)
hold on

set(gcf,'color','white') % White background for the figure.

for k = 1:true_M
    plot(mu_EM(1,k), mu_EM(2,k), 'kx');
end

% Calculate the Gaussian response for every value in the grid.
z_EM = cell(1,true_M);
for k = 1:true_M
    z_EM{k} = mvnpdf(gridX, mu_EM(:, k)', sigma_EM(:,:,k));
end

% Reshape the responses back into a 2D grid to be plotted with contour.
Z_EM = cell(1, true_M);
for k = 1:true_M
    Z_EM{k} = reshape(z_EM{k}, gridSize, gridSize);
end

% Plot the contour lines to show the pdf over the data.
for k = 1:true_M
    [C, h] = contour(u, u, Z_EM{k}, 5, '-.', 'LineWidth', 2);
    caxis([-Inf, Inf]);
end
axis([lower_bound upper_bound lower_bound upper_bound])
set(gca,'FontSize',12, 'FontWeight', 'bold');
title(sprintf('Original Data and Estimated EM PDFs with MDL = %e', mdl_EM), 'FontSize', 16, 'FontWeight', 'bold');

hold off

