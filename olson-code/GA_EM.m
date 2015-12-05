function [best, MDL_list] = GA_EM(data)
% Set Parameters
R = 3; % number of EM steps for each GA iteration
M = 15; % max number of components
K = 6; % size of parent population
H = floor(.8*K); % number of offspring
p_m = 0.02; % mutation rate

% Initialization
t = 0;
t_max = 1000; % max number of iterations (this would take a very long time)
MDL_list = zeros(t_max, 1);
OldSize = 0;
c_end = 0;
P = InitPopulation(data, M, K);

% data for updating plots as it runs
subplot(2,2,3);
xRange = min(data(:,1))-1:.1:max(data(:,1))+1; % x axis
yRange = min(data(:,2))-1:.1:max(data(:,2))+1; % y axis
[X, Y] = meshgrid(xRange,yRange); % all combinations of x, y

% plot min MDL vs iteration
subplot(2,2,4);
hold off;
plot(1:t, MDL_list(1:t));

while (c_end ~= 5 && t < t_max)
    t = t + 1;
    
    % EM loop on parent population
    P_prime = EM(P, data, R);
    % Compute MDL for updated parents
    MDL_prime = MDLencode(P_prime, data);
    
    % Produce offspring
    P_2prime = Recombine(P_prime, H, data);
    % EM loop on offspring population
    P_3prime = EM(P_2prime, data, R);
    % Compute MDL for updated offspring
    MDL_2prime = MDLencode(P_3prime, data);
    
    % Sort combined population according to MDL value and extract top K
    [MDL_list(t), P_4prime] = FitSelect([P_prime; P_3prime], [MDL_prime; MDL_2prime], K);
    
    % Store best candidate
    a_min = P_4prime(1);
    size_amin = sum(a_min.code);
    
    % If best candidate has different number of components, reset counter
    if (size_amin ~= OldSize)
        c_end = 0;
        OldSize = size_amin;
    else
        c_end = c_end + 1;
    end
    
    % Mutate new population (neither function affects best candidate)
    P_5prime = Enforce(P_4prime, data);
    P = Mutate(P_5prime, data, p_m);
    
    % plot current mixture
    subplot(2,2,3);
    scatter(data(:,1), data(:,2));
    hold on;
    index = 1:M;
    index = index(logical(a_min.code));
    for i=1:length(index)
        Z = mvnpdf([X(:) Y(:)], a_min.means(:,index(i))', a_min.covs(:,:,index(i))); % compute pdf
        Z = reshape(Z,size(X));
        contour(X,Y,Z,[.1,.1], 'LineColor', [1 0 0], 'LineWidth', 2);  % contour plot
    end
    title(['Best GA-EM Mixture, MDL = ', num2str(MDL_list(t))]);
    hold off;
    
    % update MDL plot
    subplot(2,2,4);
    hold off;
    plot(1:t, MDL_list(1:t))
    title('MDL vs iterations');
    drawnow;
end
fprintf('Finished after %d iterations\n', t);

t = t + 1;
% Run EM on best candidate until converged
best = EM(P(1), data, Inf);
% compute new MDL value
MDL_list(t) = MDLencode(best, data);

% plot best mixture
subplot(2,2,3);
scatter(data(:,1), data(:,2));
hold on;
index = 1:M;
index = index(logical(a_min.code));
for i=1:length(index)
    Z = mvnpdf([X(:) Y(:)], a_min.means(:,index(i))', a_min.covs(:,:,index(i))); % compute pdf
    Z = reshape(Z,size(X));
    contour(X,Y,Z,[.1,.1], 'LineColor', [1 0 0], 'LineWidth', 2);  % contour plot
end
title(['Best GA-EM Mixture, MDL = ', num2str(MDL_list(t))]);
hold off;

% update MDL plot
subplot(2,2,4);
hold off;
plot(1:t, MDL_list(1:t))
title('MDL vs iterations');
drawnow;

MDL_list = MDL_list(1:t);
end


