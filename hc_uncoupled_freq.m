function omega = hc_uncoupled_freq(N, phi0, p_list, x0, sim_time, u)
% Returns frequency of the uncoupled hypercycle
%
% Inputs:
%  - N: number of species per hypercycle
%  - phi0: baseline phi0
%  - p_list: vector of p-values
%  - x_0: initial condition vector
%  - sim_time: total simulation time
%  - u: complex vectors for projection

% Solve ODEs
tspan = [0 sim_time];
[t, x] = ode45(@dynamics, tspan, x0);

% Use complex-valued signal
z = x * u;

% Restrict to last portion of time
valid_idx = t > sim_time / 5;
t_trimmed = t(valid_idx);
z_trimmed = z(valid_idx);

% Compute unwrapped phase
theta = unwrap(angle(z_trimmed));

% Estimate frequency from slope of unwrapped phase
omega = (theta(end) - theta(1)) / (t_trimmed(end) - t_trimmed(1));


    function dxdt = dynamics(~, y)
        x = y(1:N);
        f = zeros(N,1);
        for a = 1:N
            am1 = mod(a-2, N) + 1;  % cyclic a-1
            f(a) = p_list(a) * x(am1);
        end
        phi = phi0 + x' * f;
        dxdt = x .* (f - phi);
    end

end
