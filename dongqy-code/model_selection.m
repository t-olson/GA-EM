% EECS 545 Final project
% Genetic-based EM Algorithm for Learning Gaussian Mixture Models

% % This is the main script to generate graphs for model selection

% Load/Generate data from true_M 2D distributions
gen_data = false;
true_M = 7;
d = 2;
if gen_data
    N = 300*true_M;
    [X, true_mu, true_sigma] = SampleData(d, N, true_M, 1.2);
end

% Run GA_EM on the data with different model parameters
figure
for p = 1:6
    init_mode = 'random';  % Initialization mode : random or k-means
    K = 6;                  %   K : number of individual in the population
    H = 4;                  %   H : number of offsprings in the new population, assumed to be multiples of 2
    R = 3;                  %   R : number of iterations
    mdl_cell = cell(1,5);
    Mmax = 15;              %   Mmax : maximal number of allowed components
    t = 0.95;               %   t : correlation threshold
    pm = 0.02;              %   pm : mutation probability
    if p == 1           % R
        para_array = [1,3,10,50,100];
    elseif p == 2       % K
        para_array = [4,6,10,20,30];
    elseif p == 3       % H (H = pc*K)
        K = 20;
        para_array = [4,8,10,16,20];
    elseif p == 4       % Mmax
        para_array = [6,10,15,20,25];
    elseif p == 5       % pm
        para_array = [0.001,0.02,0.005,0.1,0.3];
    elseif p == 6       % t_corr
        para_array = [0.7,0.8,0.9,0.95,0.98];
    end
        
    for i = 1:length(para_array)
        if p == 1           % R
            R = para_array(i);
        elseif p == 2       % K
            K = para_array(i);
        elseif p == 3       % H (H = pc*K)
            H = para_array(i);
        elseif p == 4       % Mmax
            Mmax = para_array(i);
        elseif p == 5       % pm
            pm = para_array(i);
        elseif p == 6       % t_corr
            t = para_array(i);
        end
        
        [~, ~, ~, mdl_array] = GA_EM(X, K, H, R, Mmax, t, pm, init_mode);
        fprintf('i = %d \n', i);
        mdl_array
        mdl_cell{i} = mdl_array;
        
%         mdl_cell{i} = zeros(1, 15);
%         for j = 1:10        % Average over 10 indepedent runs
%             [~, ~, ~, mdl_array] = GA_EM(X, K, H, R, Mmax, t, pm, init_mode);
%             fprintf('i = %d, j = %d', i, j);
%             mdl_array
%             tmp = zeros(1,15);
%             tmp(1,1:length(mdl_array)) = mdl_array;
%             tmp(length(mdl_array)+1:end) = mdl_array(end);
%             mdl_cell{i} = mdl_cell{i} + tmp;
%         end
%         mdl_cell{i} = mdl_cell{i}./10;
    end

    % Make plots
    symbol_array = ['s-';'+-';'*-';'x-';'o-'];
    legend_array = cell(1,length(para_array));
    subplot(2, 3, p)
    hold on
    for i = 1:length(para_array)
        if p == 1           % R
            legend_array{i} = sprintf('R = %d', para_array(i));
        elseif p == 2       % K
            legend_array{i} = sprintf('K = %d', para_array(i));
        elseif p == 3       % H (H = pc*K)
            legend_array{i} = sprintf('H = %d', para_array(i));
        elseif p == 4       % Mmax
            legend_array{i} = sprintf('M_{max} = %d', para_array(i));
        elseif p == 5       % pm
            legend_array{i} = sprintf('p_m = %.3f', para_array(i));
        elseif p == 6       % t_corr
            legend_array{i} = sprintf('t_{corr} = %.2f', para_array(i));
        end

        iter_array = 0:length(mdl_cell{i})-1;
        plot(iter_array, mdl_cell{i}, symbol_array(i,:), 'LineWidth', 2, 'MarkerSize', 12);
    end
    legend(legend_array, 'Location', 'NorthEast', 'FontSize', 13, 'FontWeight', 'bold');
    xlabel('Iteration');
    ylabel('MDL value');
    grid on
    grid minor
    set(gcf,'color','white')                        % White background for the figure.
    set(gca,'FontSize',12, 'FontWeight', 'bold');
    hold off
end