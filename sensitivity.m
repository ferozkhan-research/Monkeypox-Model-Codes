% Main sensitivity analysis script

% Define the baseline parameter values
params = [11731.91, 0.0386, 0.00139, 0.000034, 0.00111, 0.021, 0.2, 0.0694, 0.04];  % Pi_H, beta_2, m, mu_1, d, theta, Pi_R, beta_3, mu_2

% Define parameter names for the plot
param_names = {'\Pi_H', '\beta_2', 'm', '\mu_1', 'd', '\theta', '\Pi_R', '\beta_3', '\mu_2'};

% Calculate the baseline R0 value
[R0_baseline, R0_H_baseline, R0_R_baseline] = calculate_R0mpox(params);

% Define the perturbation amount 
perturbation_factor = 0.01;

% Initialize vectors to store the sensitivities for both R0_H and R0_R
sensitivities_H = zeros(size(params));
sensitivities_R = zeros(size(params));

% Loop over each parameter to calculate its sensitivity
for i = 1:length(params)
    
    perturbed_params = params;
    perturbed_params(i) = params(i) * (1 + perturbation_factor);  % 1% increase
    
    % Calculate the new R0 with the perturbed parameter
    [R0_perturbed, R0_H_perturbed, R0_R_perturbed] = calculate_R0mpox(perturbed_params);
    
    % Calculate the sensitivity index for both R0_H and R0_R using finite difference approximation
    sensitivities_H(i) = ((R0_H_perturbed - R0_H_baseline) / (params(i) * perturbation_factor)) * (params(i) / R0_H_baseline);
    sensitivities_R(i) = ((R0_R_perturbed - R0_R_baseline) / (params(i) * perturbation_factor)) * (params(i) / R0_R_baseline);
end

% Display the results
disp('Sensitivity indices for each parameter (R0_H):');
disp(sensitivities_H);
disp('Sensitivity indices for each parameter (R0_R):');
disp(sensitivities_R);

% Plot the sensitivity indices as a combined bar chart for both R0_H and R0_R
figure;
hold on;

% Plot R0_H sensitivities in red
bar(1:length(param_names), sensitivities_H, 'r', 'BarWidth', 0.4);

% Plot R0_R sensitivities in blue, slightly offset for clarity
bar(1:length(param_names) + 0.4, sensitivities_R, 'r', 'BarWidth', 0.4);


set(gca, 'XTick', 1:length(param_names), 'XTickLabel', param_names);
xlabel('Parameters');
ylabel('Sensitivity Index');
title(' Sensitivity Analysis');

ylim([-1 1]);

ax = gca; 
ax.Box = 'on'; 
ax.TickDir = 'in'; 

hold off;
