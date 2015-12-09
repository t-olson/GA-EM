function GA_EM_pendigit()
close all;
SEED = 3; % change this to get different data sets

% Set GA-EM Parameters
R = 4; % number of EM steps for each GA iteration
M = 15; % max number of components
K = 6; % size of parent population
H = 4;%floor(.8*K); % number of offspring
p_m = 0.02; % mutation rate

load('pendigit_pca_2.mat', 'X', 'label');
data = X';
C = 5; % actual number of clusters expected

% plot actual 2d data
scrsz = get(groot,'ScreenSize');
figure('Position',[1 scrsz(4)/6 scrsz(3)*2/3 scrsz(4)*2/3])
subplot(2,2,1);
gscatter(data(:,1), data(:,2),label,[],'.',5);

% Run EM algorithm 
rng(SEED); % reset random seed to ensure reproducibility
disp('Running EM');
tic
Pop = InitPopulation(data, 15, C); % initialize (requires same M and K >= C to match GA-EM init)
newP = Pop(C); % only use the C^th one, which has correct components enabled
EM_result = EM(newP, data, Inf);
MDL_EM = MDLencode(EM_result,data); % compute MDL value for EM result
toc
disp(['EM: ', num2str(MDL_EM)]); % print MDL value

% plot EM-mixture
xymin = min(data);
xymax = max(data);
xRange = xymin(1)-1:(xymax(1)-xymin(1)+2)/100:xymax(1)+1; % x axis
yRange = xymin(2)-1:(xymax(2)-xymin(2)+2)/100:xymax(2)+1; % y axis
[X, Y] = meshgrid(xRange,yRange); % all combinations of x, y
subplot(2,2,2);
gscatter(data(:,1), data(:,2),label,[],'.',5);
hold on;
for i=1:C
    Z = mvnpdf([X(:) Y(:)], EM_result.means(:,i)', EM_result.covs(:,:,i)); % compute Gaussian pdf
    Z = reshape(Z,size(X));
    contour(X,Y,Z, 'LineColor', [1 0 0], 'LineWidth', 1);  % contour plot
end
title(['Best EM Mixture (', num2str(C), ' clusters), MDL = ',  num2str(MDL_EM)]);

% Run GA_EM algorithm
rng(SEED); % reset random seed
disp('Running GA-EM');
tic
[GA_EM_result, MDL_list] = GA_EM(data, R, M, K, H, p_m);
toc
disp(['GA_EM: ', num2str(MDL_list(end))]); % print final MDL value

% plot GA_EM-mixture
subplot(2,2,3);
gscatter(data(:,1), data(:,2),label,[],'.',5);
hold on;
index = 1:M;
index = index(logical(GA_EM_result.code));
for i=1:length(index)
    Z = mvnpdf([X(:) Y(:)], GA_EM_result.means(:,index(i))', GA_EM_result.covs(:,:,index(i))); % compute Gaussian pdf
    Z = reshape(Z,size(X));
    contour(X,Y,Z, 'LineColor', [0 1 0], 'LineWidth', 1);  % contour plot
end
title(['Best GA-EM Mixture (', num2str(sum(GA_EM_result(1).code)), ' clusters), MDL = ',  num2str(MDL_list(end))]);

% show MDL plot
subplot(2,2,4);
hold off;
plot(1:length(MDL_list), MDL_list(1:length(MDL_list)))
title('MDL vs iterations');
end
