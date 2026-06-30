clear all; clc;
% SDE Model Parameters 
randn('state', 100)
pih = 4; pir = 3; beta1 = 0.02; beta2 = 0.2; beta3 = 0.25; m = 0.7; gamma = 0.8;
tau = 0.1; alpha = 0.1; d = 0.01; mu1 = 1; mu2 = 1; theta = 0.001; 
% Noise intensities for each compartment
sigma1 = 0.05;   % SH
sigma2 = 0.05;   % EH
sigma3 = 0.05;   % IH
sigma4 = 0.05;   % QH
sigma5 = 0.05;   % RH
sigma6 = 0.05;   % SR
sigma7 = 0.05;   % IR
Shzero = 1; Ehzero = 1; Ihzero = 1; Jhzero = 1; Qhzero = 1; Rhzero = 1; Srzero = 1;
Irzero = 1;
T = 60;

N = 2^7;           
L = 256;           % number of patches across [0,T] 
Nc = 3;            % LGL collocation order per patch
dtpatch = T/L;

[xi, Dleg] = lglnodes(Nc);
Np1 = Nc+1;
nvar = 7;

dWpatch = sqrt(dtpatch)*randn(1,L);   
                                      
Dphys = -Dleg * (2/dtpatch);         
                         
drift = @(X) [ ...
    pih - beta1*X(7)*X(1) - beta2*X(3)*X(1) - mu1*X(1) + tau*alpha*X(4); ...
    beta1*X(7)*X(1) + beta2*X(3)*X(1) - mu1*X(2) - m*X(2); ...
    m*X(2) - mu1*X(3) - d*X(3) - theta*X(3); ...
    theta*gamma*X(3) - mu1*X(4) - d*X(4) - tau*X(4); ...
    theta*(1-gamma)*X(3) + tau*(1-alpha)*X(4) - mu1*X(5); ...
    pir - beta3*X(6)*X(7) - mu2*X(6); ...
    beta3*X(6)*X(7) - mu2*X(7) ];

%  Jacobian (fast Newton solves)
jacdrift = @(X) [ ...
  -beta1*X(7)-beta2*X(3)-mu1,  0,           -beta2*X(1),      tau*alpha,        0,    0,                -beta1*X(1); ...
   beta1*X(7)+beta2*X(3),     -mu1-m,        beta2*X(1),       0,               0,    0,                 beta1*X(1); ...
   0,                          m,           -mu1-d-theta,      0,               0,    0,                 0; ...
   0,                          0,            theta*gamma,     -mu1-d-tau,       0,    0,                 0; ...
   0,                          0,            theta*(1-gamma),  tau*(1-alpha),  -mu1,  0,                 0; ...
   0,                          0,            0,                0,              0,   -beta3*X(7)-mu2,   -beta3*X(6); ...
   0,                          0,            0,                0,              0,    beta3*X(7),        beta3*X(6)-mu2 ];

% Initialize storage to match original variable names exactly
Shem = zeros(1, L);
Ehem = zeros(1, L);
Ihem = zeros(1, L);
Qhem = zeros(1, L);
Rhem = zeros(1, L);
Srem = zeros(1, L);
Irem = zeros(1, L);

Xcur = [Shzero; Ehzero; Ihzero; Qhzero; Rhzero; Srzero; Irzero];

% --- Stochastic LSCM main loop ---
for j = 1:L

    Xg = repmat(Xcur, 1, Np1);  

    for newt = 1:60
        F = zeros(nvar, Np1);
        for k = 1:Np1
            F(:,k) = drift(Xg(:,k));
        end
        dXdt = (Dphys * Xg')';
        Res = dXdt - F;
        Res(:,1) = Xg(:,1) - Xcur;

        nunk = nvar*Np1;
        Jac = zeros(nunk, nunk);
        for v = 1:nvar
            rows = v:nvar:nunk;
            Jac(rows, rows) = Jac(rows, rows) + Dphys;
        end
        for k = 1:Np1
            idx = (k-1)*nvar + (1:nvar);
            Jac(idx, idx) = Jac(idx, idx) - jacdrift(Xg(:,k));
        end
        idx0 = (1:nvar);
        Jac(idx0,:) = 0;
        Jac(idx0, idx0) = eye(nvar);

        Resvec = Res(:);
        dXvec = -Jac \ Resvec;
        Xvec = Xg(:) + dXvec;
        Xg = reshape(Xvec, nvar, Np1);

        if max(abs(dXvec)) < 1e-9
            break;
        end
    end

   Xend_det = Xg(:, Np1);      % spectral end-of-patch value
Winc = dWpatch(j);          

% Different noise intensities for each compartment
Sigma = [sigma1;
         sigma2;
         sigma3;
         sigma4;
         sigma5;
         sigma6;
         sigma7];

% Multiplicative stochastic perturbation
Xnext = Xend_det + (Sigma .* Xcur) * Winc;
    Shem(j) = Xnext(1);
    Ehem(j) = Xnext(2);
    Ihem(j) = Xnext(3);
    Qhem(j) = Xnext(4);
    Rhem(j) = Xnext(5);
    Srem(j) = Xnext(6);
    Irem(j) = Xnext(7);

    Xcur = Xnext;
end

time_days = linspace(0, T, L);

% Prepare training data for the neural network
X = [Shem(1:end-1)', Ehem(1:end-1)', Ihem(1:end-1)', Qhem(1:end-1)', Rhem(1:end-1)', Srem(1:end-1)', Irem(1:end-1)'];
Y = [Shem(2:end)', Ehem(2:end)', Ihem(2:end)', Qhem(2:end)', Rhem(2:end)', Srem(2:end)', Irem(2:end)'];

% Define neural network architecture
net = feedforwardnet([25, 25, 25]);
net = configure(net, X', Y');

% Train the neural network and retrieve the training record
[net, tr] = train(net, X', Y');

% Extract MSE values for each dataset from the training record
trainMSE = tr.perf(tr.best_epoch);      % MSE for training data at the best epoch
valMSE = tr.vperf(tr.best_epoch);       % MSE for validation data at the best epoch
testMSE = tr.tperf(tr.best_epoch);      % MSE for test data at the best epoch

% Display the MSE values
fprintf('Training MSE: %e\n', trainMSE);
fprintf('Validation MSE: %e\n', valMSE);
fprintf('Test MSE: %e\n', testMSE);

% Use the trained network to predict future states
predicted_Y = net(X');

marker_interval = 20;

fig_handle = findobj('Type', 'Figure');  

if isempty(fig_handle)
    
    figure;
end
hold on;  

plot(time_days(2:end), Y(:,1), 'Color', 'k', 'LineWidth', 1);   
plot(time_days(2:end), Y(:,2), 'Color', 'c', 'LineWidth', 1); 
plot(time_days(2:end), Y(:,3), 'Color',  [0 0.45 0.74], 'LineWidth', 1);   
plot(time_days(2:end), Y(:,4), 'Color', 'r', 'LineWidth', 1);  
plot(time_days(2:end), Y(:,5), 'Color', [128, 0, 0]/255, 'LineWidth', 1); 
plot(time_days(2:end), Y(:,6), 'Color', 'g', 'LineWidth', 1); 
plot(time_days(2:end), Y(:,7), 'Color', [0.3 0.75 0.93], 'LineWidth', 1); 

plot(time_days(2:marker_interval:end), predicted_Y(1,1:marker_interval:end), 'Color', [0 0.75 0.75], 'Marker', 'o', 'MarkerFaceColor', [0 0.75 0.75], 'LineStyle', 'none'); % Predicted Sh(t)
plot(time_days(2:marker_interval:end), predicted_Y(2,1:marker_interval:end), 'Color', [0.85 0.33 0.1], 'Marker', 'o', 'MarkerFaceColor', [0.85 0.33 0.1], 'LineStyle', 'none');  % Predicted Eh(t)
plot(time_days(2:marker_interval:end), predicted_Y(3,1:marker_interval:end), 'Color', [255, 140, 0]/255, 'Marker', 'o', 'MarkerFaceColor', [255, 140, 0]/255, 'LineStyle', 'none'); % Predicted Ih(t)
plot(time_days(2:marker_interval:end), predicted_Y(4,1:marker_interval:end), 'Color', [0.3 0.75 0.93], 'Marker', 'o', 'MarkerFaceColor', [0.3 0.75 0.93], 'LineStyle', 'none');  % Predicted Qh(t)
plot(time_days(2:marker_interval:end), predicted_Y(5,1:marker_interval:end), 'Color', [50, 205, 50]/255, 'Marker', 'o', 'MarkerFaceColor', [50, 205, 50]/255, 'LineStyle', 'none'); % Predicted Rh(t)
plot(time_days(2:marker_interval:end), predicted_Y(6,1:marker_interval:end), 'Color', 'm', 'Marker', 'o', 'MarkerFaceColor', 'm', 'LineStyle', 'none');  % Predicted Sr(t)
plot(time_days(2:marker_interval:end), predicted_Y(7,1:marker_interval:end), 'Color', [0.49 0.18 0.56], 'Marker', 'o', 'MarkerFaceColor', 'b', 'LineStyle', 'none'); % Predicted Ir(t)

xlabel('Time (days)', 'FontSize', 12);
ylabel('N(t)', 'FontSize', 12, 'Rotation', 0, 'HorizontalAlignment', 'right');
legend('Sh(t)', 'Eh(t)', 'Ih(t)', 'Qh(t)', 'Rh(t)', 'Sr(t)', ...
       'NN Sh(t)', 'NN Eh(t)', 'NN Ih(t)', 'NN Qh(t)', 'NN Rh(t)', 'NN Sr(t)');
axis([0 T 0 5]);
ax = gca; 
ax.Box = 'on'; 
ax.TickDir = 'in';

hold off;

