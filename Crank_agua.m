% CN radial conduction + per-shell post-processing:
%   E_cum(i,n): cumulative energy vs initial (raw)
%   E_step(i,n): per-step energy gained (raw)
%   E_thresh(i): energy to reach 100C from 298K
%   Qevap_vec(i): latent energy to evaporate target water (per shell)
%   remainder_hist(i,n): energy diverted to evaporation at step n
%   E_real(i,n): adjusted ("real") energy built from per-step gains minus diversions
%   n_dry(i): first time index when shell i fully evaporates 
%   T_real(i,n): adjusted temperature derived from E_real


clear; clc; close all;

%% Physical & numerical parameters
R       = 0.031;          % radius (m)
alpha   = 1.714e-8;        % m^2/s
k       = 0.12;           % W/m/K
rho     = 840;           % kg/m^3  (WET bulk density: includes water)
rho_dry = 400;            % kg/m^3  (DRY bulk density: NO water)
cp      = 8333;           % J/kg/K  (single cp used in conduction/energy)
T0      = 298.0;          % K
T_boil  = 373.15;         % K
q_given = 22040;          % W/m^2

Nr   = 2000;              % Nr intervals => Nr+1 nodes
dr   = R / Nr;
r    = linspace(0, R, Nr+1)';   % nodes

dt       = 0.1;           % s
T_end    = 3600;           % s
Nsteps   = ceil(T_end/dt);

%% Evaporation target setup
evap_frac = 1.0;          % fraction (0..1) of the available water mass to evaporate
L_v       = 2.257e6;      % J/kg (latent heat at 100 C)

%% Geometry helpers (fixed: no ablation)
faces_from_nodes   = @(r_nodes, Rsurf) [0.0; 0.5*(r_nodes(2:end)+r_nodes(1:end-1)); Rsurf];
volumes_from_faces = @(rf, L) pi*(rf(2:end).^2 - rf(1:end-1).^2)*L;
Laxis = 1.0;                                  % axial length (m)

rf   = faces_from_nodes(r, R);
V    = volumes_from_faces(rf, Laxis);         % m^3 per shell (Nr+1 x 1)

% Heat capacity per shell (single cp, uses WET rho for conduction bookkeeping)
Csol = rho * cp * V;                          % J/K per shell

%% Build Crank–Nicolson system (constant geometry)
A = sparse(Nr+1, Nr+1);
B = sparse(Nr+1, Nr+1);

for i = 2:Nr
    ri = r(i);
    c1 = alpha*dt/(2*dr^2);
    c2 = alpha*dt/(4*dr*ri);

    A(i,i-1) = -c1 + c2;
    A(i,i)   =  1 + 2*c1;
    A(i,i+1) = -c1 - c2;

    B(i,i-1) =  c1 - c2;
    B(i,i)   =  1 - 2*c1;
    B(i,i+1) =  c1 + c2;
end

% Center Neumann (symmetry): T(1) - T(2) = 0
A(1,1) = 1; A(1,2) = -1;
B(1,1) = 1; B(1,2) = -1;

% Surface node + applied flux
i  = Nr + 1;
ri = r(i);
c1 = alpha*dt/(2*dr^2);
c2 = alpha*dt/(4*dr*ri);

A(i,i-1) = -2*c1;
A(i,i)   =  1 + 2*c1;
B(i,i-1) =  2*c1;
B(i,i)   =  1 - 2*c1;

% Constant flux -> RHS temperature source at surface
q_src  = 2*q_given/dr;        % consistent with -2*c1 stencil
deltaT = q_src*dt/(rho*cp);   % K per step

%% Solve CN for full horizon (no ablation)
T = T0*ones(Nr+1,1);
T_hist = nan(Nr+1, Nsteps);

for n = 1:Nsteps
    b = B*T;
    b(end) = b(end) + deltaT;     % apply flux source
    T = A \ b;
    T_hist(:,n) = T;
end

t_hist = (1:Nsteps)' * dt;

%% ENERGY MATRICES (raw)
% Cumulative energy vs initial
E_cum  = Csol .* (T_hist - T0);          % (Nr+1) x Nsteps

% Per-step raw energy gain
E_step = zeros(size(E_cum));
E_step(:,1)      = E_cum(:,1);
E_step(:,2:end)  = E_cum(:,2:end) - E_cum(:,1:end-1);

% Energy to reach 100C FROM 298K (per shell; time-invariant)
E_thresh = Csol * (T_boil - T0);         % (Nr+1) x 1

%% Water mass per shell from densities (WET - DRY)
% This assumes rho, rho_dry are bulk densities for the same geometry.
% Any positive difference is attributed to water mass.
rho_water_mass_per_vol = max(rho - rho_dry, 0);  % kg/m^3 of water
m_water_total = rho_water_mass_per_vol * V;      % kg water per shell
m_water_t     = evap_frac * m_water_total;       % target water mass to evaporate per shell (kg)

% Latent energy per shell
Qevap_vec = m_water_t * L_v;                     % J latent target per shell

%% PER-SHELL DIVERSION (step-based accumulation)
remainder_hist = zeros(size(E_cum));   % diverted at step n (J)
E_real         = zeros(size(E_cum));   % adjusted energy (built from steps)
Qrem           = Qevap_vec;            % remaining latent per shell (J)
n_dry          = nan(Nr+1,1);          % first time index when shell fully dry

for i = 1:Nr+1
    E_run   = 0;       % running raw cumulative energy vs initial up to step n
    cum_div = 0;       % running cumulative diversion up to step n
    for n = 1:Nsteps
        % 1) Update raw running energy by per-step gain
        Ei_step = E_step(i,n);     % raw energy gained this step
        E_run   = E_run + Ei_step; % equals E_cum(i,n)

        % 2) If water remains, compute excess above threshold AFTER past diversions
        if Qrem(i) > 0
            excess = E_run - E_thresh(i) - cum_div;
            if excess > 0
                div = min(excess, Qrem(i));           % divert up to remaining latent
                cum_div               = cum_div + div;
                Qrem(i)               = Qrem(i) - div;
                remainder_hist(i,n)   = div;
                if Qrem(i) <= 0 && isnan(n_dry(i))
                    n_dry(i) = n;                     % first dry step index
                end
            end
        end

        % 3) Build adjusted energy BY ADDING per-step gain minus this step's diversion
        if n == 1
            E_real(i,n) = Ei_step - remainder_hist(i,n);
        else
            E_real(i,n) = E_real(i,n-1) + (Ei_step - remainder_hist(i,n));
        end
    end
end

%% Adjusted temperature and animation
T_real = T0 + E_real ./ Csol;   % adjusted temperature profile over time

% Simple animation
frame_skip = 5;
figure('Name','T_real Animation','Color','w');
h = plot(r, T_real(:,1), 'LineWidth', 1.5);
xlabel('Radius (m)'); ylabel('Temperature (K)');
title(sprintf('Adjusted Temperature Profile (t = %.2f s)', t_hist(1)));
ylim([T0, max(T_real(:))+50]); grid on;
for n = 1:frame_skip:Nsteps
    set(h, 'YData', T_real(:,n));
    title(sprintf('Adjusted Temperature Profile (t = %.2f s)', t_hist(n)));
    drawnow;
end

%% (Optional) sanity checks/plots
E_total_raw  = sum(E_cum,1);
E_total_real = sum(E_real,1);
figure; plot(t_hist, E_total_raw, 'LineWidth',1.5); hold on;
plot(t_hist, E_total_real, '--', 'LineWidth',1.5);
legend('Raw cumulative energy','Adjusted (real) energy','Location','northwest');
grid on; xlabel('Time (s)'); ylabel('Energy (J)'); title('Total Energy vs Time');

figure; plot(r, Qrem, 'LineWidth',1.2);
grid on; xlabel('Radius (m)'); ylabel('Remaining latent per shell (J)');
title('Remaining Evaporation Energy After Post-Processing');

%% Report position(s) where T_real = 600 K at final time
%% Report radius/radii where T_real = 600 K at final time
T_target = 600;                  % K
tol      = 1e-6;                 % tolerance for "exact" equality
T_final  = T_real(:,end);        % adjusted temperatures at t_end

r_hits   = [];                   % point hits
ranges   = [];                   % [r_start r_end] for plateaus

% 1) Exact hits (including plateaus)
eq = abs(T_final - T_target) <= tol;
if any(eq)
    idx = find(eq);
    % group consecutive indices into plateaus
    d = diff(idx);
    breaks = [0; find(d>1); numel(idx)];
    for b = 1:numel(breaks)-1
        seg = idx(breaks(b)+1 : breaks(b+1));
        if numel(seg) == 1
            r_hits(end+1,1) = r(seg); %#ok<AGROW>
        else
            % plateau exactly at target: report radius range
            ranges(end+1,:) = [r(seg(1)) r(seg(end))]; %#ok<AGROW>
        end
    end
end

% 2) Crossings between nodes (linear interpolation)
for k = 1:numel(r)-1
    Ti = T_final(k);
    Tj = T_final(k+1);
    % skip if either endpoint already counted as exact hit
    if abs(Ti - T_target) <= tol || abs(Tj - T_target) <= tol
        continue
    end
    % detect a sign change around the target
    if (Ti - T_target) * (Tj - T_target) < 0
        % linear interpolation in radius
        alpha = (T_target - Ti) / (Tj - Ti);   % in (0,1)
        rcross = r(k) + alpha * (r(k+1) - r(k));
        r_hits(end+1,1) = rcross; %#ok<AGROW>
    end
end

% 3) Print results
if isempty(r_hits) && isempty(ranges)
    fprintf('At t = %.1f s, T_{real} does not reach %.1f K anywhere in [0, %.5f] m.\n', ...
        t_hist(end), T_target, r(end));
else
    fprintf('At t = %.1f s, T_{real} = %.1f K at:\n', t_hist(end), T_target);
    % point hits
    if ~isempty(r_hits)
        r_hits = unique(r_hits);  % de-dup and sort
        for m = 1:numel(r_hits)
            fprintf('  r ≈ %.6f m (point)\n', r_hits(m));
        end
    end
    % plateau ranges
    for m = 1:size(ranges,1)
        fprintf('  r ∈ [%.6f, %.6f] m (plateau)\n', ranges(m,1), ranges(m,2));
    end
end


