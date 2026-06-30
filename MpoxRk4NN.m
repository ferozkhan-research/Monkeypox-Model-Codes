

clear all; close all; clc;
rand('state',100);   
randn('state',100);  

%% Parameters
T0 = 0; 
T = 60; 
n = 256; 
dt = (T - T0)/n;

Pi_H = 4; Pi_R = 3;
beta1 = 0.2; beta2 = 0.2; beta3 = 0.4;
tau = 0.6; alpha = 0.1; gamma = 0.5;
mu1 = 1; mu2 = 1; m = 0.7; d = 0.01; theta = 0.01;

sigma = 0.0*ones(7,1);

%% Initialize state variables
S_H = zeros(1,n); E_H = zeros(1,n); I_H = zeros(1,n);
Q_H = zeros(1,n); R_H = zeros(1,n); S_R = zeros(1,n); I_R = zeros(1,n);

% Initial conditions
S_H(1)=1; E_H(1)=1; I_H(1)=1; 
Q_H(1)=1; R_H(1)=1; S_R(1)=1; I_R(1)=1;

%% Brownian increments
dW = sqrt(dt)*randn(7,n);

%% Deterministic dynamics
f = @(x)[...
    Pi_H - (beta1*x(7)+beta2*x(3))*x(1) + tau*alpha*x(4) - mu1*x(1);
    (beta1*x(7)+beta2*x(3))*x(1) - (mu1+m)*x(2);
    m*x(2) - (mu1+d+theta)*x(3);
    theta*gamma*x(3) - (mu1+d+tau)*x(4);
    theta*(1-gamma)*x(3) + tau*(1-alpha)*x(4) - mu1*x(5);
    Pi_R - beta3*x(6)*x(7) - mu2*x(6);
    beta3*x(6)*x(7) - mu2*x(7)];

%% Stochastic dynamics
g = @(x)[sigma(1)*x(1); sigma(2)*x(2); sigma(3)*x(3); ...
          sigma(4)*x(4); sigma(5)*x(5); sigma(6)*x(6); sigma(7)*x(7)];

%% Stochastic Euler–Maruyama (RK4-like substitute)
for i=2:n
    X = [S_H(i-1); E_H(i-1); I_H(i-1); Q_H(i-1); R_H(i-1); S_R(i-1); I_R(i-1)];
    X_new = X + dt*f(X) + g(X).*dW(:,i-1);
    S_H(i)=X_new(1); E_H(i)=X_new(2); I_H(i)=X_new(3);
    Q_H(i)=X_new(4); R_H(i)=X_new(5); S_R(i)=X_new(6); I_R(i)=X_new(7);
end

time = linspace(T0,T,n);

%% Prepare dataset (inputs: time, outputs: states)
inputs = time;
targets = [S_H; E_H; I_H; Q_H; R_H; S_R; I_R];
inputs = inputs/max(inputs);   % Normalize

%% Create RNN-like Feedforward Network
hiddenLayerSize = [20 20 20];
net = feedforwardnet(hiddenLayerSize,'trainlm');
net.trainParam.epochs = 500;
net.trainParam.goal = 1e-6;
net.trainParam.min_grad = 1e-7;

[net,tr] = train(net,inputs,targets);

%% Predictions
preds = net(inputs);

%% Error metrics
mse_err = mean((targets - preds).^2,2);
mae_err = mean(abs(targets - preds),2);

fprintf('MSE per variable:\n'); disp(mse_err');
fprintf('MAE per variable:\n'); disp(mae_err');

%% Plot all variables together
figure;
vars = {'S_H','E_H','I_H','Q_H','R_H','S_R','I_R'};
colors = lines(7);
hold on;

for k=1:7
    % True RK4 solution (solid line)
    plot(time,targets(k,:),'Color',colors(k,:),'LineWidth',1,'DisplayName',[vars{k} ' (RK4)']);
    % NN predictions (markers every 10 steps)
    plot(time(1:10:end),preds(k,1:10:end),'*','Color',colors(k,:),'MarkerSize',6,'DisplayName',[vars{k} ' (FFNN)']);
end
axis([0 T -0.3 4]);
xlabel('Time'); ylabel('Population');
title('RK4 vs FFNN-Det');
legend('show');
%grid on; 
box on;
