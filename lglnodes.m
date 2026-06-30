function [x, D] = lglnodes(N)
% LGLNODES  Legendre-Gauss-Lobatto nodes and spectral differentiation matrix
%   [x, D] = lglnodes(N) returns N+1 LGL collocation nodes x on [-1,1]
%   (descending: x(1)=+1 ... x(N+1)=-1) and the (N+1)x(N+1) Legendre
%   differentiation matrix D, such that D*f(x) approximates f'(x) to
%   spectral accuracy for smooth f.
%
%   Standard construction (Trefethen, "Spectral Methods in MATLAB", 2000;
%   Canuto et al., "Spectral Methods", 2006). MATLAB 2015-compatible
%   (no toolboxes, no implicit broadcasting).

    Np1 = N+1;
    x = cos(pi*(0:N)/N)';            % Chebyshev-Gauss-Lobatto initial guess
    P = zeros(Np1, Np1);
    xold = 2*ones(Np1,1);
    iter = 0;
    while max(abs(x-xold)) > eps
        iter = iter + 1;
        xold = x;
        P(:,1) = 1; P(:,2) = x;
        for k = 2:N
            P(:,k+1) = ((2*k-1)*x.*P(:,k) - (k-1)*P(:,k-1))/k;
        end
        x = xold - (x.*P(:,Np1) - P(:,N)) ./ (Np1*P(:,Np1));
        if iter > 200
            break;
        end
    end
    P(:,1) = 1; P(:,2) = x;
    for k = 2:N
        P(:,k+1) = ((2*k-1)*x.*P(:,k) - (k-1)*P(:,k-1))/k;
    end
    D = zeros(Np1,Np1);
    for i = 1:Np1
        for j = 1:Np1
            if i ~= j
                D(i,j) = P(i,Np1)/P(j,Np1) / (x(i)-x(j));
            end
        end
    end
    D(1,1) = N*Np1/4;
    D(Np1,Np1) = -N*Np1/4;
end
