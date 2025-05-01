function hcPhaseLock

N_pop = 10;     % number of species per hypercycle
N_osc = 2;      % number of hypercycles
phi0 = 0.0005;
diffusion_strength = 10^(-3);  % strength of diffusion between cycles
sim_time = 400;

% Generate initial conditions for two hypercycles
x1 = rand(N_pop,1); x1 = x1 / sum(x1);
x2 = rand(N_pop,1); x2 = x2 / sum(x2);
x_init = [x1; x2];

% Define the p-lists for each hypercycle
p = 1;
p1_list = p * ones(N_pop,1);  % oscillator 1
p2_list = p * ones(N_pop,1);  % oscillator 2
p_lists = [p1_list, p2_list];  % [N_pop x N_osc]

% Adjacency matrix: fully connected except self-coupling
Adj = [0 1;
       1 0];

% Complex Fourier basis vector for projection
u = exp(2i*pi*(0:N_pop-1)/N_pop).'; % column vector

% Solve ODEs using the updated generalized run_simulation
[x, t, omega_vec] = hc_sim(N_pop, N_osc, phi0, diffusion_strength, Adj, p_lists, x_init, sim_time, u);

% Plot populations
figure(1); clf;
subplot(2,1,1);
plot(t, x(:,1:N_pop)); title('Hypercycle 1 Populations'); ylabel('Population');

subplot(2,1,2);
plot(t, x(:,N_pop+1:end)); title('Hypercycle 2 Populations'); ylabel('Population'); xlabel('Time');

% Compute complex projection
x1_complex = x(:,1:N_pop) * u;
x2_complex = x(:,N_pop+1:end) * u;

theta1 = angle(x1_complex);
theta2 = angle(x2_complex);

% Plot phase difference
figure(2); clf;
plot(t, abs(wrapToPi(theta1 - theta2)));
xlabel('Time'); ylabel('|\theta_1 - \theta_2|');
title('Phase Difference');

% Compute instantaneous frequencies
omega1 = gradient(unwrap(theta1), t);
omega2 = gradient(unwrap(theta2), t);

% Plot frequency difference
figure(3); clf;
plot(t, omega1 - omega2);
hold on;
yline(0, 'k:');  % Add dotted line at zero
xlabel('Time'); ylabel('\omega_1 - \omega_2');
title('Instantaneous Frequency Difference');

% Plot absolute differences between hypercycles
figure(4); clf;
for i = 1:N_pop
    plot(t, abs(x(:,i) - x(:,N_pop+i))); hold on;
end
xlabel('Time'); ylabel('|x1 - x2|'); title('Population Differences');

% Display the observed average frequencies
fprintf('Observed frequencies:\n');
disp(omega_vec);

end
