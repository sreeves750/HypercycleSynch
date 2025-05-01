function hcRepPlot
    % Parameters
    N = 10;
    sim_time = 1000;
    phi0 = 0.0005;
    p = 1;

    % Define hypercycle matrix A
    A = zeros(N,N);
    for i = 2:N
        A(i,i-1) = p;
    end
    A(1,N) = p;

    % Initial condition
    x0 = rand(N,1);
    x0 = x0 / sum(x0);

    % Simulate dynamics using default solver settings
    tspan = [0 sim_time];
    [t, X] = ode45(@(t,x) replicator_dynamics(t,x,A,phi0), tspan, x0);

    % Complex basis vectors
    u = exp(2*pi*1i*(0:N-1)'/N);

    % Project onto complex representation
    z = X * u;

    r = abs(z);
    theta = unwrap(angle(z));

    % Plot
    figure(1);
    plot(t, theta, 'k', 'LineWidth', 1.5);
    xlabel('Time');
    ylabel('Phase');
    % title('Hypercycle Phase vs. Time');
    grid on;

    % Plot
    figure(2);
    plot(t, r, 'k', 'LineWidth', 1.5);
    xlabel('Time');
    ylabel('radius');
    % title('Representation magnitude vs. Time');
    grid on;

    function dx = replicator_dynamics(~, x, A, phi0)
        f = A * x;
        phi = phi0 + x' * f;
        dx = x .* (f - phi);
    end
end
