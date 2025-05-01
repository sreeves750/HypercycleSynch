function hcFreqWidth

N_pop = 8; % species per hypercycle
N_osc = 2;  % number of hypercycles
phi0 = 0.0005;
sim_time = 1000;
p_table = [9.355, 15.57, 23.29, 32.71, 43.88, 57.41, 73.56, 91.37, 115.45, 139.55, 174.46, 206.14, 258.31, 293.3, 371.39, 427.62];
p = p_table(N_pop-4);
D_values = logspace(-3*N_pop, -1, 12);
widths = NaN(size(D_values));
searches = 10;

u = exp(2*pi*1i*(0:N_pop-1)'/N_pop);

Adj = [0 1; 1 0];

p_list_base = p * ones(N_pop,1);

x1i = rand(N_pop,1); x1i = x1i / sum(x1i);
x2i = rand(N_pop,1); x2i = x2i / sum(x2i);
x_init = [x1i; x2i];

threshold = 0.001; % locking threshold

q = parallel.pool.DataQueue;
Nsteps = length(D_values);
afterEach(q, @(~) trackProgress(Nsteps));

parfor d_idx = 1:Nsteps
    diffusion_strength = D_values(d_idx);

    % Local copies inside parfor
    p_list_base_local = p_list_base;
    x_init_local = x_init;
    u_local = u;
    Adj_local = Adj;
    x1i_local = x1i;
    x2i_local = x2i;

    % Find right edge
    right_offset = NaN;
    offset_guess = 0;
    while offset_guess <= 10
        [locked, ~] = check_locking(offset_guess, diffusion_strength, p_list_base_local, x_init_local, N_pop, N_osc, phi0, sim_time, Adj_local, u_local, threshold);
        if locked
            offset_guess = offset_guess + 0.5;
        else
            break;
        end
    end
    if offset_guess > 10
        right_offset = NaN;
    else
        low = offset_guess - 0.5;
        high = offset_guess;
        for iter = 1:searches
            mid = (low + high)/2;
            [locked, ~] = check_locking(mid, diffusion_strength, p_list_base_local, x_init_local, N_pop, N_osc, phi0, sim_time, Adj_local, u_local, threshold);
            if locked
                low = mid;
            else
                high = mid;
            end
        end
        right_offset = (low + high)/2;
    end

    % Find left edge
    low = -1;
    high = 0;
    for iter = 1:searches
        mid = (low + high)/2;
        [locked, ~] = check_locking(mid, diffusion_strength, p_list_base_local, x_init_local, N_pop, N_osc, phi0, sim_time, Adj_local, u_local, threshold);
        if locked
            high = mid;
        else
            low = mid;
        end
    end
    left_offset = (low + high)/2;

    if ~isnan(left_offset) && ~isnan(right_offset)
        omega_left = hc_uncoupled_freq(N_pop, phi0, p_list_base_local * (1 + left_offset), x2i_local, sim_time, u_local);
        omega_right = hc_uncoupled_freq(N_pop, phi0, p_list_base_local * (1 + right_offset), x2i_local, sim_time, u_local);
        widths(d_idx) = omega_right - omega_left;

        if (widths(d_idx) < 0) 
            widths(d_idx) = NaN;
        end
    else
        widths(d_idx) = NaN;
    end

    send(q, 1);
end

% Plot results
figure(10); clf;
plot(D_values, widths, 'ko-', 'MarkerFaceColor', 'k');
set(gca, 'XScale', 'log');
xlabel('Diffusion Strength D');
ylabel('Width of Frequency Locking Region');
grid on;

hold on;

xdata = D_values(:);
ydata = widths(:);

valid = ~isnan(xdata) & ~isnan(ydata);
xdata = xdata(valid);
ydata = ydata(valid);

% Define fitting model: y = a * log(1 + b*x)
fitmodel = fittype('a*log(1 + x*exp(b))', 'independent', 'x', 'coefficients', {'a', 'b'});

opts = fitoptions('Method', 'NonlinearLeastSquares', ...
                  'StartPoint', [1/10, 10]);

[fitresult, gof] = fit(xdata, ydata, fitmodel, opts);

xfit = logspace(log10(min(xdata)), log10(max(xdata)), 500);
yfit = fitresult.a * log(1 + xfit * exp(fitresult.b));
plot(xfit, yfit, 'r-', 'LineWidth', 2);

legend('Data', sprintf('Fit: a=%.3g, b=%.3g', fitresult.a, fitresult.b), 'Location', 'best');

fprintf('Fit coefficients:\n');
fprintf('a = %.6f\n', fitresult.a);
fprintf('b = %.6f\n', fitresult.b);
fprintf('R^2 = %.6f\n', gof.rsquare);

% Theory line
a_theory = 1 / (0.759144 * N_pop + 3.296635);
b_theory = 4.568472 * N_pop - 19.006794;

x_theory = logspace(log10(min(xdata)), log10(max(xdata)), 500);
y_theory = a_theory * log(1 + x_theory * exp(b_theory));

plot(x_theory, y_theory, 'b--', 'LineWidth', 2, 'DisplayName', 'Theory');

legend('Data', ...
       sprintf('Fit: a=%.3g, b=%.3g', fitresult.a, fitresult.b), ...
       'Theory', ...
       'Location', 'best');


hold off;



end

function [locked, omega_diff] = check_locking(offset, diffusion_strength, p_list_base, x_init, N_pop, N_osc, phi0, sim_time, Adj, u, threshold)
    p_lists = [p_list_base, p_list_base * (1 + offset)];
    [~, ~, omega_vec] = hc_sim(N_pop, N_osc, phi0, diffusion_strength, Adj, p_lists, x_init, sim_time, u);

    if any(isnan(omega_vec))
        locked = false;
        omega_diff = NaN;
    else
        omega_diff = omega_vec(1) - omega_vec(2);
        locked = abs(omega_diff) < threshold;
    end
end


function trackProgress(Nsteps)
    persistent progress_internal
    if isempty(progress_internal)
        progress_internal = 0;
    end
    progress_internal = progress_internal + 1;
    if mod(progress_internal, 1) == 0 || progress_internal == Nsteps
        fprintf('Progress: %d of %d (%.1f%%)\n', progress_internal, Nsteps, 100*progress_internal/Nsteps);
    end
end
