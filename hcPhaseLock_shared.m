function hcPhaseLock_shared

N = 10;  % number of species per hypercycle
phi0 = 0.0005;

% Generate initial conditions
x1 = rand(N,1); x1 = x1 / sum(x1);
x2 = rand(N,1); x2 = x2 / sum(x2);

% Force x^0 = y^0 at initial time
x2(1) = x1(1);
x2 = x2 / sum(x2);

x0 = [x1; x2];
p = 1;
noise = 0;

% Create randomized hypercycle interaction matrices
A1 = zeros(N,N);
A2 = zeros(N,N);

for i = 2:N
    A1(i,i-1) = p * (1 + 2*noise*(rand-0.5));
    A2(i,i-1) = p * (1 + 2*noise*(rand-0.5));
end
A1(1,N) = p * (1 + 2*noise*(rand-0.5));
A2(1,N) = p * (1 + 2*noise*(rand-0.5));


% Solve ODEs
tspan = [0 1000];
[t,x] = ode45(@dynamics, tspan, x0);

% Plot populations
figure(1); clf;
subplot(2,1,1);
hold on;
for i = 1:N
    if i == 1
        plot(t, x(:,i), 'LineWidth', 2.5, 'Color', [0,0,0]); % orange for shared species
    else
        plot(t, x(:,i), 'LineWidth', 1);
    end
end
title('Hypercycle 1 Populations'); ylabel('Population');
hold off;

subplot(2,1,2);
hold on;
for i = 1:N
    if i == 1
        plot(t, x(:,N+i), 'LineWidth', 2.5, 'Color', [0,0,0]); % orange for shared species
    else
        plot(t, x(:,N+i), 'LineWidth', 1);
    end
end
title('Hypercycle 2 Populations'); ylabel('Population'); xlabel('Time');
hold off;

% Compute phases
u = exp(2i*pi*(0:N-1)/N); % u_a = exp(2pi i (a-1)/N)
x1_complex = x(:,1:N) * u.';
x2_complex = x(:,N+1:end) * u.';

theta1 = angle(x1_complex);
theta2 = angle(x2_complex);

% Plot phase difference
figure(2); clf;
plot(t, abs(wrapToPi(theta1 - theta2)), 'k');
xlabel('Time'); ylabel('|\theta_1 - \theta_2|');
title('Phase Difference');

% Plot population differences
figure(3); clf;
hold on;
for i = 1:N
    if i == 1
        plot(t, abs(x(:,i) - x(:,N+i)), 'LineWidth', 2.5, 'Color', [0.8500 0.3250 0.0980]); % orange for shared species
    else
        plot(t, abs(x(:,i) - x(:,N+i)), 'LineWidth', 1);
    end
end
xlabel('Time'); ylabel('|x1 - x2|');
title('Population Differences');
hold off;

    function dxdt = dynamics(t, y)
        x1 = y(1:N);
        y1 = y(N+1:end);

        % Enforce x^0 = y^0
        y1(1) = x1(1);

        f1 = A1 * x1;
        f2 = A2 * y1;

        phi1 = phi0 + x1' * f1;
        phi2 = phi0 + y1' * f2;

        dx1 = x1 .* (f1 - phi1);
        dy1 = y1 .* (f2 - phi2);

        % Enforce derivative consistency for species 1
        dy1(1) = dx1(1);

        dxdt = [dx1; dy1];
    end

end
