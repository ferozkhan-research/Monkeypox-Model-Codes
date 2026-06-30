clear all; clc;
randn('state',100)
pih = 4; pir = 3; beta3 = 0.8; beta2 = 0.8; gamma = 0.8; tau = 0.5;
alpha = 0.1; d = 0.01; mu1=1; mu2 = 1; theta = 0.3; m = 0.7;
sigma1 = 0.02;   % SH noise intensity
sigma2 = 0.02;   % EH noise intensity
sigma3 = 0.02;   % IH noise intensity
sigma4 = 0.02;   % QH noise intensity
sigma5 = 0.02;   % RH noise intensity
sigma6 = 0.02;   % SR noise intensity
sigma7 = 0.02;   % IR noise intensity
Shzero = 1; Ehzero = 1; Ihzero = 1; Jhzero = 1; Qhzero = 1; Rhzero = 1; Srzero = 1;
Irzero = 1;
T = 60;

% --- LSCM patch settings ---
L = 7680;           % number of stochastic patches across [0,T]
Nc = 3;            % LGL collocation order per patch
dtpatch = T/L;

[xi, Dleg] = lglnodes(Nc);
Np1 = Nc+1;
nvar = 7;

Dphys = -Dleg * (2/dtpatch); 

beta1_values = [0.2, 0.4, 0.6];

colors = {'c', 'm', 'g'};
marker_colors = {'r', 'b', 'k'};

figure;
hold on;

legend_entries = {};

marker_interval = 400; 

for idx = 1:length(beta1_values)
    beta1 = beta1_values(idx);

    dWpatch = sqrt(dtpatch)*randn(1, L);

    drift = @(X) [ ...
        pih - beta1*X(7)*X(1) - beta2*X(3)*X(1) - mu1*X(1) + tau*alpha*X(4); ...
        beta1*X(7)*X(1) + beta2*X(3)*X(1) - mu1*X(2) - m*X(2); ...
        m*X(2) - mu1*X(3) - d*X(3) - theta*X(3); ...
        theta*gamma*X(3) - mu1*X(4) - d*X(4) - tau*X(4); ...
        theta*(1-gamma)*X(3) + tau*(1-alpha)*X(4) - mu1*X(5); ...
        pir - beta3*X(6)*X(7) - mu2*X(6); ...
        beta3*X(6)*X(7) - mu2*X(7) ];

    % Analytical Jacobian of drift w.r.t. X for the current beta1
    jacdrift = @(X) [ ...
      -beta1*X(7)-beta2*X(3)-mu1,  0,           -beta2*X(1),      tau*alpha,        0,    0,                -beta1*X(1); ...
       beta1*X(7)+beta2*X(3),     -mu1-m,        beta2*X(1),       0,               0,    0,                 beta1*X(1); ...
       0,                          m,           -mu1-d-theta,      0,               0,    0,                 0; ...
       0,                          0,            theta*gamma,     -mu1-d-tau,       0,    0,                 0; ...
       0,                          0,            theta*(1-gamma),  tau*(1-alpha),  -mu1,  0,                 0; ...
       0,                          0,            0,                0,              0,   -beta3*X(7)-mu2,   -beta3*X(6); ...
       0,                          0,            0,                0,              0,    beta3*X(7),        beta3*X(6)-mu2 ];

    Shem = zeros(1, L);
    Ehem = zeros(1, L);
    Ihem = zeros(1, L);
    Qhem = zeros(1, L);
    Rhem = zeros(1, L);
    Srem = zeros(1, L);
    Irem = zeros(1, L);

    Xcur = [Shzero; Ehzero; Ihzero; Qhzero; Rhzero; Srzero; Irzero];

    % --- Stochastic LSCM  ---
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
                idxk = (k-1)*nvar + (1:nvar);
                Jac(idxk, idxk) = Jac(idxk, idxk) - jacdrift(Xg(:,k));
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

        Xend_det = Xg(:, Np1);                       % spectral end-of-patch value
        Winc = dWpatch(j);                            
        Sigma = [sigma1; sigma2; sigma3; sigma4; sigma5; sigma6; sigma7];

        Xnext = Xend_det + Sigma .* Xcur * Winc;
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

    net = feedforwardnet([25, 25, 25]);
    net = configure(net, X', Y');
    
    [net, tr] = train(net, X', Y');

    % Extract MSE values for each dataset from the training record
    trainMSE = tr.perf(tr.best_epoch);      % MSE for training data at the best epoch
    valMSE = tr.vperf(tr.best_epoch);       % MSE for validation data at the best epoch
    testMSE = tr.tperf(tr.best_epoch);      % MSE for test data at the best epoch

    % Display the MSE values
    fprintf('beta1 = %.2f\n', beta1);
    fprintf('Training MSE: %e\n', trainMSE);
    fprintf('Validation MSE: %e\n', valMSE);
    fprintf('Test MSE: %e\n', testMSE);

    % Use the trained network to predict future states
    predicted_Y = net(X');

    % Plot Ih(t) for current beta1
    plot(time_days, Ihem, 'Color', colors{idx}, 'LineWidth', 1);
    legend_entries{end+1} = ['Ih(t) \beta_1 = ', num2str(beta1)];

    % Plot ANN-predicted points at specified intervals
    plot(time_days(2:marker_interval:end), predicted_Y(3, 1:marker_interval:end), 'o', ...
      'MarkerFaceColor', marker_colors{idx}, 'LineWidth', 1);
    legend_entries{end+1} = ['NN Ih(t) \beta_1 = ', num2str(beta1)];
end

xlabel('Time (days)', 'FontSize', 12);
ylabel('I_H(t)', 'FontSize', 12, 'Rotation', 0, 'HorizontalAlignment', 'right'); 

text(0.5, 1, ' Sto - I_H(t)', 'FontSize', 12, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', 'Units', 'normalized');

legend(legend_entries, 'Location', 'best');

ax = gca; 
ax.Box = 'on'; 
ax.TickDir = 'in'; 

hold off; 
