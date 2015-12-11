% EECS 545 Final project
% Genetic-based EM Algorithm for Learning Gaussian Mixture Models

% % This is the main script to generate graphs for model selection

% Load/Generate data from true_M 2D distributions
gen_data = true;
true_M = 7;
d = 2;
if gen_data
    N = 300*true_M;
    [X, true_mu, true_sigma] = SampleData(d, N, true_M);
end

% Run GA_EM on the data with different model parameters
init_mode = 'random';  % Initialization mode : random or k-means
K = 6;                  %   K : number of individual in the population
H = 4;                  %   H : number of offsprings in the new population, assumed to be multiples of 2
R = 3;                  %   R : number of iterations
R_array = [1,3,10,50,100];
mdl_cell = cell(1,5);
Mmax = 15;              %   Mmax : maximal number of allowed components
t = 0.95;               %   t : correlation threshold
pm = 0.02;              %   pm : mutation probability
for i = 1:length(R_array)
    R = R_array(i);
    [~, ~, ~, mdl_array] = GA_EM(X, K, H, R, Mmax, t, pm, init_mode);
    fprintf('i = %d\n', i);
    mdl_array
    mdl_cell{i} = mdl_array;
end

% Make plots
figure
hold on
for i = length(R_array)
    iter_array = 1:length(mdl_cell{i});
    plot(iter_array, mdl_cell{i}, '--', 'LineWidth', 2);
end
set(gca,'FontSize',12, 'FontWeight', 'bold');
hold off