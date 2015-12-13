% EECS 545 Final project
% Genetic-based EM Algorithm for Learning Gaussian Mixture Models

% % This is the main script to generate graphs for initialization

% Load/Generate data from true_M 2D distributions
gen_data = false;
true_M = 7;
d = 2;
if gen_data
    N = 300*true_M;
    [X, true_mu, true_sigma] = SampleData(d, N, true_M, 1.2);
end

% Run GA_EM & EM on the data with different initialization mode
figure
hold on
mdl_gaem = cell(1,2);
mdl_em = cell(1,2);
for p = 1:2
    if p == 1
        init_mode = 'random';  % Initialization mode : random or k-means
    elseif p == 2
        init_mode = 'k-means';  % Initialization mode : random or k-means
    end
    K = 6;                  %   K : number of individual in the population
    H = 4;                  %   H : number of offsprings in the new population, assumed to be multiples of 2
    R = 3;                  %   R : number of iterations
    Mmax = 15;              %   Mmax : maximal number of allowed components
    t = 0.95;               %   t : correlation threshold
    pm = 0.02;              %   pm : mutation probability
    
    % Run GA-EM
    [~, ~, ~, mdl_array] = GA_EM(X, K, H, R, Mmax, t, pm, init_mode);
    mdl_gaem{p} = mdl_array(2:end);
    
    % Run EM on the data with the right number of clusters : true_M
    tol = 1e-6;
    individual_EM = Individual(true_M, d, X, init_mode);
    while individual_EM.num() ~= true_M
        individual_EM = Individual(true_M, d, X, init_mode);
    end
    assert(individual_EM.num() == true_M);
    mdl_array = individual_EM.EM_gmm(X, 500);
    mdl_em{p} = mdl_array(2:end);
    
    % Make plots
    symbol_array = ['s-';'--';'o-';'-.'];
    iter_array_gaem = 1:length(mdl_gaem{p});
    iter_array_em = linspace(1,length(mdl_gaem{p}), length(mdl_em{p}));
    plot(iter_array_gaem, mdl_gaem{p}, symbol_array(2*p-1,:), 'LineWidth', 3, 'MarkerSize', 16);
    plot(iter_array_em, mdl_em{p}, symbol_array(2*p,:), 'LineWidth', 3, 'MarkerSize', 16);
end

legend({'GA-EM (random)', 'EM (random)', 'GA-EM (k-means)', 'EM (k-means)'}, ...
    'Location', 'NorthEast', 'FontSize', 18, 'FontWeight', 'bold');
xlabel('(Scaled) iteration');
ylabel('MDL value');
grid on
grid minor
set(gcf,'color','white')                        % White background for the figure.
set(gca,'FontSize',12, 'FontWeight', 'bold');
hold off
