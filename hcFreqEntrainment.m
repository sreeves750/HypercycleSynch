function hcFreqEntrainment

N = 10;
phi0 = 0.0005;
diffusion_strength = 10^(-7);
offset_range = linspace(-0.95, 0.95, 50);
sim_time = 1000;
p = 57.41;

u = exp(2*pi*1i*(0:N-1)'/N);

omega1_list = NaN(length(offset_range),1);
omega2_list = NaN(length(offset_range),1);
omega_diff_list = NaN(length(offset_range),1);
offset_actual = NaN(length(offset_range),1);

p_list = p * ones(N,1);

x1i = rand(N,1); x1i = x1i / sum(x1i);
x2i = rand(N,1); x2i = x2i / sum(x2i);

omega1_free = hc_uncoupled_freq(N, phi0, p_list, x1i, sim_time, u);
fprintf('omega1 = %0.10f\n', omega1_free);

q = parallel.pool.DataQueue;
progress = 0;
Nsteps = length(offset_range);
afterEach(q, @trackProgress);

parfor k = 1:length(offset_range)
    offset = offset_range(k);
    x1 = x1i;
    x2 = x2i;
    x0 = [x1; x2];
    p_lists = [p_list, p_list * (1 + offset)];

    Adj = [0 1; 1 0];

    omega2_free = hc_uncoupled_freq(N, phi0, p_lists(:,2), x2, sim_time, u);
    [~, ~, omega_vec] = hc_sim(N, 2, phi0, diffusion_strength, Adj, p_lists, x0, sim_time, u);
    omega1_obs = omega_vec(1);
    omega2_obs = omega_vec(2);

    omega1_tmp = NaN;
    omega2_tmp = NaN;
    omega_diff_tmp = NaN;
    offset_actual_tmp = NaN;

    if ~any(isnan([omega1_free, omega2_free, omega1_obs, omega2_obs]))
        offset_actual_tmp = omega2_free - omega1_free;
        omega1_tmp = omega1_obs;
        omega2_tmp = omega2_obs;
        omega_diff_tmp = omega1_obs - omega2_obs;
    end

    offset_actual(k) = offset_actual_tmp;
    omega1_list(k) = omega1_tmp;
    omega2_list(k) = omega2_tmp;
    omega_diff_list(k) = omega_diff_tmp;

    send(q, 1);
end

x_vals = offset_actual(:);
y1_vals = omega1_list(:);
y2_vals = omega2_list(:);
ydiff_vals = omega_diff_list(:);

valid = ~isnan(x_vals) & ~isnan(y1_vals) & ~isnan(y2_vals) & ~isnan(ydiff_vals);
x_vals = x_vals(valid);
y1_vals = y1_vals(valid);
y2_vals = y2_vals(valid);
ydiff_vals = ydiff_vals(valid);

figure(1); clf;
scatter(x_vals, y1_vals, 20, 'r', 'filled'); hold on;
scatter(x_vals, y2_vals, 20, 'b', 'filled');
yline(omega1_free, 'r:', 'LineWidth', 1.5);
fplot(@(x) x + omega1_free, [min(x_vals), max(x_vals)], 'b:', 'LineWidth', 1.5);
xlabel('Uncoupled frequency offset \omega_2 - \omega_1');
ylabel('Observed oscillator frequencies \omega_1, \omega_2');
title('Observed Frequencies vs Uncoupled Offset');
legend('\omega_1','\omega_2','Location','NorthWest');

figure(2); clf;
scatter(x_vals, -ydiff_vals, 20, 'r', 'filled'); hold on;
yline(0, 'k:', 'LineWidth', 1.5);
fplot(@(x) x, [min(x_vals), max(x_vals)], 'k:', 'LineWidth', 1.5);
xlabel('Uncoupled frequency offset \omega_2 - \omega_1');
ylabel('Frequency difference \omega_1 - \omega_2');
title('Frequency Difference vs Uncoupled Offset');

nbins = 50;
edges = linspace(min(x_vals), max(x_vals), nbins+1);
bin_centers = (edges(1:end-1) + edges(2:end)) / 2;

mean_y1 = NaN(1, nbins);
mean_y2 = NaN(1, nbins);
mean_ydiff = NaN(1, nbins);

for i = 1:nbins
    in_bin = x_vals >= edges(i) & x_vals < edges(i+1);
    if any(in_bin)
        mean_y1(i) = mean(y1_vals(in_bin));
        mean_y2(i) = mean(y2_vals(in_bin));
        mean_ydiff(i) = mean(ydiff_vals(in_bin));
    end
end

figure(3); clf;
plot(bin_centers, mean_y1, 'ro-', 'DisplayName', '\omega_1'); hold on;
plot(bin_centers, mean_y2, 'bo-', 'DisplayName', '\omega_2');
yline(omega1_free, 'r:', 'LineWidth', 1.5);
fplot(@(x) x + omega1_free, [min(x_vals), max(x_vals)], 'b:', 'LineWidth', 1.5);
xlabel('Uncoupled frequency offset \omega_2 - \omega_1');
ylabel('Observed oscillator frequencies \omega_1, \omega_2');
title('Observed Frequencies');
legend('\omega_1','\omega_2','Location','NorthWest');

figure(4); clf;
plot(bin_centers, -mean_ydiff, 'ro-'); hold on;
yline(0, 'k:', 'LineWidth', 1.5);
fplot(@(x) x, [min(x_vals), max(x_vals)], 'k:', 'LineWidth', 1.5);
xlabel('Uncoupled frequency offset \omega_2 - \omega_1');
ylabel('Frequency difference \omega_1 - \omega_2');
title('Frequency Difference vs Uncoupled Offset');

fprintf('Done\n');

    function trackProgress(~)
        progress = progress + 1;
        if mod(progress, 10) == 0 || progress == Nsteps
            fprintf('Progress: %d of %d (%.1f%%)\n', progress, Nsteps, 100*progress/Nsteps);
        end
    end

end
