% EECS 545 Final project
% Genetic-based EM Algorithm for Learning Gaussian Mixture Models

% This is the main script to generate data, run GA_EM and plot

% Load/Generate data from true_M 2D distributions
true_M = 5;
N = 1000;
d = 2;
[X, true_mu, true_sigma] = SampleData(d, N, true_M);

% Run GA_EM on the data
K = 6;          %   K : number of individual in the population
H = 4;          %   H : number of offsprings in the new population, assumed to be multiples of 2
R = 3;          %   R : number of iterations
Mmax = 15;      %   Mmax : maximal number of allowed components
t = 0.95;       %   t : correlation threshold
pm = 0.02;      %   pm : mutation probability
[best_M, best_mdl, best_individual] = GA_EM(X, K, H, R, Mmax, t, pm);
mu = best_individual.Mu(:, best_individual.Binary == 1);
sigma = best_individual.Sigma(:,:,best_individual.Binary == 1);

% Make plots
figure
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
    [C, h] = contour(u, u, Z{k}, '--', 'LineWidth', 2);
    caxis([-1 1]);
end
axis([lower_bound upper_bound lower_bound upper_bound])
set(gca,'FontSize',12, 'FontWeight', 'bold');
title(sprintf('Original Data and Estimated PDFs with MDL = %e', best_mdl), 'FontSize', 16, 'FontWeight', 'bold');

hold off

% True configuration
true_individual = Individual(true_M, d);
true_individual.Binary = ones(1, true_M);
true_individual.weight = ones(1, true_M)/true_M;
true_individual.Mu = true_mu;
true_individual.Sigma = true_sigma;
true_individual.mdl = true_individual.MDL(X);
figure
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
    [C, h] = contour(u, u, Z_true{k}, 'LineWidth', 2);
    caxis([-1 1]);
end
axis([lower_bound upper_bound lower_bound upper_bound])
set(gca,'FontSize',12, 'FontWeight', 'bold');
title(sprintf('Original Data and true PDFs with MDL = %e', true_individual.mdl), 'FontSize', 16, 'FontWeight', 'bold');

hold off

