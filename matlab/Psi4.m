% Copyright 2026 Celephais contributors
%
% This file is part of Celephais and is licensed under the GNU General
% Public License, version 3 or (at your option) any later version.

clear;
% =====================================================================
%  Psi4.m
%  ---------------------------------------------------------------------
%  Co-precessing (quadrupole-aligned) waveforms from the full set of
%  numerical psi_4 modes of a precessing binary.
%
%  Prerequisites: Clone to matlab path https://github.com/Hao-Jui/gw-matlab
%
%  Method (QA "co-precessing" frame): O'Shaughnessy et al. arXiv:1109.5224;
%  Schmidt, Hannam & Husa arXiv:1207.3088; the frame Kawaguchi et al.
%  arXiv:1705.07459 (Eq. 5) adopt.  All inertial-frame modes h_lm are
%  available, so the orbital axis and co-precessing frame are obtained
%  exactly (no single-observer orbital-phase inversion needed).
%
%  Pipeline:
%    1. Load every psi_4^{lm} (l=2..lmax, m=-l..l) at the extraction radius.
%    2. r*psi_4^{lm} -> r*h_lm by Fixed-Frequency Integration (FFI, Reisswig
%       & Pollney arXiv:1006.1632), optionally extrapolated to infinity
%       (Nakano arXiv:1501.02890); per-mode cutoff w0=w0_factor*|m|*Omega_orb
%       (m=0 floored to |m|=1).
%    3. Radiation axis L^(t) = dominant eigenvector of the <L_a L_b> matrix
%       (O'Shaughnessy), l=2..lmax_axis.
%    4. Euler angles (Kawaguchi Eqs. 2-3,6): theta_L = acos(L^_z),
%       phi_L = Arg[L^_x + i L^_y] - pi/2, psi_L = -int phi_L' cos(theta_L) dt.
%    5. Co-precessing modes h^QA_{l,m'} = sum_m [D^l_{m,m'}]* h_lm,
%       alpha = Arg[L^_x+iL^_y], beta = theta_L, gamma = psi_L.
%
%  A single-spin simple-precession model maps the L^ cone about the fixed
%  axis J = L + S1 to the spin-orbit tilt theta_SL and supplies
%  the analytic theta_L/phi_L/psi_L overlays in Fig 2.
%
%  Outputs: Fig 1 (featured mode inertial vs QA, GW frequency)
%           Fig 2 (theta_L, phi_L, psi_L with analytic overlays and theta_SL inversion),
%                 and a console summary with the axis conditioning.
%
%  Units: code time unit == 1 M_sun
% =====================================================================
save_plot     = 0;                 % export the paper's precession-frame figure
save_waveform = 0;                 % write (t, Re/Im r*h^QA_22) to a .dat
plot_name     = "qa_waveform";

root          = fileparts(fileparts(mfilename('fullpath')));
data_dir      = fullfile(root, 'matlab/Dat/N120_res15');

l_unit  = 1.4769994423016508;   % M [code units] -> km (optional physical axis)
t_unit  = l_unit/2.99792458e5;
% ---------------------------------------------------------------------
%  User parameters
% ---------------------------------------------------------------------
lmax            = 4;        % highest l loaded AND summed into the observer strain (data has l<=4)
lmax_axis       = 4;        % highest l entering the <L_a L_b> radiation-axis matrix
extr_radius     = 600;      % extraction radius: 800 | 600 | 480 | 400 (also the areal radius)
extrapolate_to_infinity = 1;% apply arXiv:1501.02890 finite-radius correction to each mode
background_spin = 0.0;      % Kerr-background spin for the Nakano spin-coupling term; 0 = Schwarzschild.
                            % Nonzero auto-loads same-m neighbor modes for the coupling.
adm_mass        = 2.8;      % total mass M [code units]; retarded r*, Tukey Delta_t, (1-2M/r) tetrad factor
mass1           = 1.4;      % body-1 (spinning) mass [code units]
mass2           = 1.4;      % body-2 (companion) mass [code units]
spin1_chi       = 0.3;      % dimensionless spin |chi1| = |J1|/m1^2; S1 = chi*m1^2
spin1_tilt_deg  = 80.83;   % initial spin tilt [deg] from the pole; sets the fixed precession axis
                            % J_pred = L0 zhat + S1 Shat0 (+-x branch picked against the data below)
Omega_orb       = 0.0073140971866294440;   % initial ORBITAL angular frequency
w0_factor       = 0.8;      % FFI cutoff = w0_factor*|m|*Omega_orb (m=0 floored to |m|=1)
delta_t_factor  = 200;      % FFI Tukey cosine-ramp duration = factor*M
tcut            = 0;        % retarded-time window cut: tret<tcut zeroed [M]
psi4_convention = +1;       % +1: psi4 = +d^2 h/dt^2 ; -1: psi4 = -d^2 h/dt^2
use_load_cache  = 1;        % per-run-dir .mat cache of the parsed ASCII mode set (gw.load_mode_set;
                            % invalidated by any file's bytes/mtime); 0 = always parse
diag_ell        = 2;        % l of the featured mode shown inertial-vs-QA in Fig 1
diag_m          = 2;        % m of the featured mode

if abs(diag_m) > diag_ell || diag_ell < 2 || diag_ell > lmax
    error('Psi4:badDiag', 'require 2 <= diag_ell <= lmax and |diag_m| <= diag_ell.');
end
if lmax < 2 || lmax ~= round(lmax), error('Psi4:badLmax', 'lmax must be an integer >= 2.'); end
if lmax_axis < 2 || lmax_axis > lmax, error('Psi4:badLmaxAxis', 'lmax_axis must satisfy 2 <= lmax_axis <= lmax.'); end

omega_code_to_hz = 1 / (2*pi*t_unit);   % code angular frequency -> Hz

if ~(Omega_orb > 0), error('Psi4:badOmegaOrb', 'Omega_orb (initial orbital angular frequency) must be positive.'); end

% ---------------------------------------------------------------------
%  Load every mode (l=2..lmax, m=-l..l) at the extraction radius
%  (shared cached loader gw.load_mode_set; the mode-file column layout and
%  supported radii live in gw.load_psi4_mode_file; missing modes stay zero)
% ---------------------------------------------------------------------
mode_ell = []; mode_m = [];
for l = 2:lmax
    for m = -l:l
        mode_ell(end+1) = l;
        mode_m(end+1)   = m;
    end
end
n_modes = numel(mode_ell);

[~, run_label] = fileparts(char(data_dir));
[t, psi_finite, mode_present] = gw.load_mode_set(char(data_dir), mode_ell, mode_m, ...
    1:n_modes, extr_radius, use_load_cache, run_label);
if isempty(t), error('Psi4:noModes', 'No mode files were found in %s.', data_dir); end
N = numel(t);                  % psi_finite: N x n_modes finite-radius r*psi_4 (inertial)
if ~any(mode_present & mode_ell==2 & abs(mode_m)==2)
    error('Psi4:noDominant', 'The (2,+-2) modes are required and were not found.');
end

% ---------------------------------------------------------------------
%  Time bookkeeping: retarded time via the Schwarzschild tortoise r*
% ---------------------------------------------------------------------
gm      = adm_mass;
delta_t = delta_t_factor * gm;
if ~(delta_t > 0), error('Psi4:badDeltaT', 'Delta_t must be positive; set adm_mass and delta_t_factor > 0.'); end
if adm_mass > 0
    r_star = extr_radius + 2*adm_mass*log(extr_radius/(2*adm_mass) - 1);
else
    r_star = extr_radius;
end
tret       = t - r_star;
time_to_ms = t_unit * 1e3;   % 1 code unit (M_sun) -> ms
tplot      = tret * time_to_ms;
dt = median(diff(t));
if max(abs(diff(t) - dt)) > 1e-3*dt
    warning('Psi4:nonUniformTimeStep', 'non-uniform time step; FFI assumes uniform dt.');
end

% per-mode FFI cutoff w0 = w0_factor*|m|*Omega_orb, m=0 floored to |m|=1
mode_w0 = w0_factor * max(abs(mode_m), 1) * Omega_orb;

% ---------------------------------------------------------------------
%  r*psi_4^{lm} -> r*h_lm  (finite-radius extrapolation + FFI), all modes
%  via the shared gw.psi4_to_strain; a nonzero background_spin adds the
%  Nakano same-m coupling built internally from the loaded mode set
% ---------------------------------------------------------------------
h_lm = gw.psi4_to_strain(psi_finite, tret, mode_ell, mode_m, mode_w0, ...
           delta_t, tcut, psi4_convention, extrapolate_to_infinity, ...
           extr_radius, adm_mass, background_spin);   % inertial-frame r*h_lm

% window / marker start at the leading-ramp end, tcut + Delta_t (retarded)
i_on = find(tret >= tcut + delta_t, 1, 'first');
if isempty(i_on), i_on = 1; end

win = (1:N).' >= i_on;
% merger time: peak amplitude of the dominant (2,2) mode
[~, i_mrg] = max(abs(h_lm(:, find(mode_ell==2 & mode_m==2, 1))));

xspan = [tret(i_on), tret(i_mrg)+500]*t_unit*1e3;
% ---------------------------------------------------------------------
%  Radiation axis from the O'Shaughnessy <L_a L_b> matrix (l<=lmax_axis),
%  shared gw.radiation_axis (batched pageeig, auto-fallback to eig if the
%  toolbox lacks it).  The dominant eigenvector is a
%  well-defined axis only while lambda_1 is separated from lambda_2
%  (near-degenerate post-merger); axis_relgap flags it.
% ---------------------------------------------------------------------
[axis_vec, axis_relgap] = gw.radiation_axis(h_lm, mode_ell, mode_m, lmax_axis);

% Euler angles (Kawaguchi Eqs. 2-3,6); computed on the analysis window,
% held constant before turn-on so the unwrap/integral never see junk.
theta_L = acos(max(-1, min(1, axis_vec(:,3))));
argL = zeros(N,1);
argL(win) = unwrap(atan2(axis_vec(win,2), axis_vec(win,1)));
argL(~win) = argL(i_on);
phi_L = argL - pi/2;                            % paper's -pi/2 azimuth shift
dargL = zeros(N,1);
dargL(win) = gradient(argL(win), tret(win));
psi_L = zeros(N,1);
psi_L(win) = -cumtrapz(tret(win), dargL(win) .* cos(theta_L(win)));

% ---------------------------------------------------------------------
%  Co-precessing (QA) modes:  h^QA_{l,m'} = sum_m conj(D^l_{m,m'}) h_lm
%    conj(D^l_{m,m'}) = e^{+i m alpha} d^l_{m,m'}(beta) e^{+i m' gamma}
%  alpha = argL, beta = theta_L, gamma = psi_L.
% ---------------------------------------------------------------------
alpha = argL; beta = theta_L; gamma = psi_L;
h_qa = zeros(N, n_modes);
for l = 2:lmax
    cols = find(mode_ell == l);
    [ms_sorted, order] = sort(mode_m(cols)); cols = cols(order);
    Hl = h_lm(:, cols);                          % N x (2l+1), columns m=-l..l
    Cl = zeros(N, 2*l+1);
    for ap = 1:2*l+1
        mp = ms_sorted(ap);
        acc = zeros(N,1);
        for bp = 1:2*l+1
            m = ms_sorted(bp);
            d = gw.wigner_small_d(l, m, mp, beta);  % d^l_{m,mp}(beta), vector over t
            acc = acc + exp(1i*m*alpha) .* d .* Hl(:,bp);
        end
        Cl(:,ap) = exp(1i*mp*gamma) .* acc;
    end
    h_qa(:, cols) = Cl;
end

% round-trip closure (certifies the rotation math): rotate QA forward to inertial
h_rec = zeros(N, n_modes);
for l = 2:lmax
    cols = find(mode_ell == l);
    [ms_sorted, order] = sort(mode_m(cols)); cols = cols(order);
    Cl = h_qa(:, cols);
    Hl = zeros(N, 2*l+1);
    for am = 1:2*l+1
        m = ms_sorted(am);
        acc = zeros(N,1);
        for bp = 1:2*l+1
            mp = ms_sorted(bp);
            d = gw.wigner_small_d(l, m, mp, beta);
            acc = acc + d .* exp(-1i*mp*gamma) .* Cl(:,bp);
        end
        Hl(:,am) = exp(-1i*m*alpha) .* acc;
    end
    h_rec(:, cols) = Hl;
end
roundtrip_err = max(abs(h_rec(win,:) - h_lm(win,:)), [], 'all') / max(abs(h_lm(win,:)), [], 'all');

% featured mode (diag_ell,diag_m): inertial vs co-precessing
kd = find(mode_ell==diag_ell & mode_m==diag_m, 1);
if isempty(kd), error('Psi4:diagModeMissing', 'featured mode (%d,%+d) not loaded.', diag_ell, diag_m); end
h_inert_diag = h_lm(:,kd);
h_qa_diag    = h_qa(:,kd);
% orbital phase from the QA (2,2) mode (Phi_QA = half the (2,2) phase)
Phi_qa = 0.5 * unwrap(angle(h_qa(:, find(mode_ell==2 & mode_m==2,1))));

% instantaneous GW frequency of the featured QA mode
omega_qa = abs(gradient(unwrap(angle(h_qa_diag)), dt));
f_qa_hz  = omega_qa .* omega_code_to_hz;

% ---------------------------------------------------------------------
%  Diagnostics
% ---------------------------------------------------------------------
% de-precessing diagnostics are canonical to the dominant l=2 quadrupole (independent of diag_ell)
k22p = find(mode_ell==2&mode_m==2,1);  k22m = find(mode_ell==2&mode_m==-2,1);
k21p = find(mode_ell==2&mode_m==1,1);  k21m = find(mode_ell==2&mode_m==-1,1);
% QA power fraction in (2,+-2) over the window (should be ~1 if inspiral)
P_all = sum(abs(h_qa(win,:)).^2, 2);
P_22  = abs(h_qa(win,k22p)).^2 + abs(h_qa(win,k22m)).^2;
f22 = mean(P_22 ./ P_all, 'omitnan');
% equatorial-symmetry restoration in the QA frame: h^QA_{2,-2} vs (-1)^2 conj(h^QA_{2,2})
sym_res_inert = mean(abs(h_lm(win,k22m) - conj(h_lm(win,k22p))), 'omitnan') / ...
                mean(abs(h_lm(win,k22p)), 'omitnan');
sym_res_qa    = mean(abs(h_qa(win,k22m) - conj(h_qa(win,k22p))), 'omitnan') / ...
                mean(abs(h_qa(win,k22p)), 'omitnan');
% m=+-1 / (2,2) amplitude ratio, inertial vs QA (the de-precessing effect)
ratio_m1_inert = mean((abs(h_lm(win,k21p))+abs(h_lm(win,k21m)))./(2*abs(h_lm(win,k22p))), 'omitnan');
ratio_m1_qa    = mean((abs(h_qa(win,k21p))+abs(h_qa(win,k21m)))./(2*abs(h_qa(win,k22p))), 'omitnan');

% --- spin-orbit tilt inversion (single-spin simple precession) ---
% theta_JL(t) = cone half-angle of L-hat about the fixed axis Jhat; invert
% J = L + S1,
% theta_SL = theta_JL + asin[(L/S1) sin(theta_JL)].
% L = PN orbital angular momentum L(v), v=(M Omega_orb)^{1/3}, Omega_orb =
% half the QA (2,2) frequency; S1 = spin1_chi*mass1^2.  Jhat = fixed ID axis
% J_pred = L0 zhat + S1 Shat0; mean(L-hat) only picks the +-x azimuth sign.
w_insp = win & ((1:N).' <= i_mrg);
omega_22 = abs(gradient(unwrap(angle(h_qa(:,k22p))), dt));      % QA (2,2) GW frequency = 2*Omega_orb
Omega_orb_t = 0.5 * omega_22;
M_bin = mass1 + mass2;  nu = (mass1*mass2)/M_bin^2;
pn_L  = @(v) (nu*M_bin^2 ./ v) .* (1 + (1.5 + nu/6).*v.^2 + (27/8 - 19*nu/8 + nu^2/24).*v.^4);  % non-spinning PN L(v)
v_pn  = (M_bin * max(Omega_orb_t, eps)).^(1/3);
L_orb = pn_L(v_pn);
S1_mag = spin1_chi * mass1^2;
if ~(isfinite(S1_mag) && S1_mag > 0)
    error('Psi4:badSpin', 'The cone inversion requires a finite positive spin magnitude.');
end
if any(~isfinite(L_orb(w_insp)) | L_orb(w_insp) <= 0)
    error('Psi4:badOrbitalAngularMomentum', ...
          'The cone inversion requires finite positive PN orbital angular momentum.');
end
tilt0  = deg2rad(spin1_tilt_deg);
L0     = pn_L((M_bin * Omega_orb)^(1/3));
Jhat_mean = mean(axis_vec(w_insp,:), 1);  Jhat_mean = Jhat_mean / norm(Jhat_mean);
Jcand = [L0*[0 0 1] + S1_mag*[ sin(tilt0) 0 cos(tilt0)];
         L0*[0 0 1] + S1_mag*[-sin(tilt0) 0 cos(tilt0)]];
Jcand = Jcand ./ vecnorm(Jcand, 2, 2);
[~, jpick] = max(Jcand * Jhat_mean.');
Jhat = Jcand(jpick, :);
theta_JL = acos(max(-1, min(1, axis_vec * Jhat.')));           % L-hat cone half-angle about Jhat
asin_arg = (L_orb ./ max(S1_mag, eps)) .* sin(theta_JL);
theta_SL = nan(N,1);
good_tilt = (abs(asin_arg) <= 1) & (S1_mag > 0);
asin_good = asin(asin_arg(good_tilt));
theta_SL(good_tilt) = theta_JL(good_tilt) + asin_good;
good_insp = good_tilt & w_insp;
if ~any(good_insp)
    error('Psi4:noValidConeInverse', 'No inspiral samples admit a real cone inversion.');
end
theta_JL_roundtrip = atan2(S1_mag*sin(theta_SL(good_insp)), ...
                           L_orb(good_insp) + S1_mag*cos(theta_SL(good_insp)));
assert(max(abs(theta_JL_roundtrip - theta_JL(good_insp))) < 1e-10, ...
       'Psi4:coneRoundTrip', 'The cone inversion failed the forward-map check.');
tilt_med = median(theta_SL(w_insp), 'omitnan');
tilt_std = std(theta_SL(w_insp), 'omitnan');

% --- theta_JL uncertainty propagated to the tilt ---
% sigma_JL = hypot(statistical nutation scatter reduced by #orbits, systematic
% sub-window median drift), pushed through the inverse.  The inverse
% becomes ill conditioned near theta_SL* = acos(-S1/L).
iiw     = find(w_insp);
tJL_w   = theta_JL(iiw);
P_orb0  = 2*pi/Omega_orb;
res_nut = tJL_w - movmedian(tJL_w, max(3, round(P_orb0/dt)));
n_orb   = (tret(iiw(end)) - tret(iiw(1))) / P_orb0;
sig_stat = std(res_nut) / sqrt(max(n_orb, 1));
thirds  = round(linspace(1, numel(iiw)+1, 4));
med3    = arrayfun(@(k) median(tJL_w(thirds(k):thirds(k+1)-1)), 1:3);
sig_sys = std(med3);
sig_JL  = hypot(sig_stat, sig_sys);
tJL_med = median(tJL_w);
LoS     = median(L_orb(iiw)) / max(S1_mag, eps);
theta_JL_max_med = asin(min(1, 1/LoS));
tJL_med_clip = min(max(tJL_med, 0), theta_JL_max_med);
tJL_lo_med   = min(max(tJL_med - sig_JL, 0), theta_JL_max_med);
tJL_hi_med   = min(max(tJL_med + sig_JL, 0), theta_JL_max_med);
arg_ctr_med  = min(1, max(-1, LoS*sin(tJL_med_clip)));
arg_lo_med   = min(1, max(-1, LoS*sin(tJL_lo_med)));
arg_hi_med   = min(1, max(-1, LoS*sin(tJL_hi_med)));
tilt_ctr = tJL_med_clip + asin(arg_ctr_med);
tilt_lo  = tJL_lo_med + asin(arg_lo_med);
tilt_hi  = tJL_hi_med + asin(arg_hi_med);
tilt_star = acos(max(-1, min(1, -1/LoS)));
amp_ctr   = 1 + LoS*cos(tJL_med)/sqrt(max(1 - (LoS*sin(tJL_med))^2, eps));
% Pointwise 1-sigma envelopes.  Clamp theta_JL itself at the physical cone
% maximum before inversion.
theta_JL_max = asin(min(1, S1_mag ./ max(L_orb, eps)));
theta_JL_lo = min(max(theta_JL - sig_JL, 0), theta_JL_max);
theta_JL_hi = min(max(theta_JL + sig_JL, 0), theta_JL_max);
arg_lo = min(1, max(-1, (L_orb ./ max(S1_mag, eps)) .* sin(theta_JL_lo)));
arg_hi = min(1, max(-1, (L_orb ./ max(S1_mag, eps)) .* sin(theta_JL_hi)));
tSL_lo = theta_JL_lo + asin(arg_lo);
tSL_hi = theta_JL_hi + asin(arg_hi);
assert(all(tSL_lo(w_insp) <= tSL_hi(w_insp) + 100*eps), ...
       'Psi4:unorderedTiltEnvelope', 'A cone-inversion uncertainty envelope is unordered.');

% --- analytic simple-precession frame angles (zero free parameters) ---
% Lhat rides the cone about the fixed Jhat: half-angle tJL_a(t) from the
% J = L + S geometry, phase from the leading-order precession rate
% Omega_p = (2 + 1.5 m2/m1) J v^6/M^3.  Overlaid dashed on Fig 2.
Lt_a = L_orb;  Lt_a(~win) = L0;
vt_a = v_pn;   vt_a(~win) = (M_bin*Omega_orb)^(1/3);
Jmag_a  = sqrt(Lt_a.^2 + S1_mag^2 + 2*Lt_a.*S1_mag*cos(tilt0));
Om_p_a  = (2 + 1.5*mass2/mass1) .* Jmag_a .* vt_a.^6 / M_bin^3;
Phi_p_a = cumtrapz(tret, Om_p_a .* (tret >= 0));     % ID moment ~ tret = 0
tJL_a   = atan(S1_mag*sin(tilt0) ./ (Lt_a + S1_mag*cos(tilt0)));
e1_a = [0 0 1] - Jhat(3)*Jhat;  e1_a = e1_a/norm(e1_a);  % Phi_p = 0 <=> Lhat(0) = z
e2_a = cross(Jhat, e1_a);
Lhat_a = cos(tJL_a).*Jhat + sin(tJL_a).*(cos(Phi_p_a)*e1_a + sin(Phi_p_a)*e2_a);
theta_L_a = acos(max(-1, min(1, Lhat_a(:,3))));
argL_a = zeros(N,1); argL_a(win) = unwrap(atan2(Lhat_a(win,2), Lhat_a(win,1)));
phi_L_a = argL_a - pi/2;
darg_a = zeros(N,1); darg_a(win) = gradient(argL_a(win), tret(win));
psi_L_a = zeros(N,1); psi_L_a(win) = -cumtrapz(tret(win), darg_a(win).*cos(theta_L_a(win)));

fprintf('--- psi4 -> QA co-precessing (%d modes, l<=%d; axis l<=%d, r_ex=%g) ---\n', ...
        nnz(mode_present), lmax, lmax_axis, extr_radius);
fprintf('  N=%d  dt=%.4f  T=%.1f  Delta_t=%.1f  window-start t_ret=%.1f M (%.2f ms)\n', ...
        N, dt, t(end)-t(1), delta_t, tret(i_on), tret(i_on) * time_to_ms);
fprintf('  extrapolate_to_infinity=%d  M=%.6g  a=%.6g\n', ...
        extrapolate_to_infinity, adm_mass, background_spin);
fprintf('  ROTATION round-trip closure (rotate o de-rotate = id): %.2e (should be ~1e-12)\n', roundtrip_err);
fprintf('  QA (2,+-2) power fraction over window          = %.4f\n', f22);
fprintf('  |h_2,+-1|/|h_2,2|  inertial -> QA              = %.3e -> %.3e\n', ratio_m1_inert, ratio_m1_qa);
fprintf('  equatorial-sym residual |h_2,-2 - conj h_2,2|  inertial -> QA = %.3e -> %.3e\n', ...
        sym_res_inert, sym_res_qa);
fprintf('  max theta_L over window = %.3f rad (%.1f deg);  psi_L range = [%.2f, %.2f]\n', ...
        max(theta_L(win)), rad2deg(max(theta_L(win))), min(psi_L(win)), max(psi_L(win)));
fprintf('  axis conditioning (lambda1-lambda2)/lambda1: median=%.2f min=%.2e over window\n', ...
        median(axis_relgap(win)), min(axis_relgap(win)));
fprintf('    (small values => near-degenerate <LL>; QA axis unreliable there, typically post-merger)\n');
fprintf('  CONE-INFERRED spin-orbit tilt theta_SL = %.1f +- %.1f deg\n', ...
        rad2deg(tilt_med), rad2deg(tilt_std));
fprintf('    [theta_JL cone median=%.2f deg, L/S1=%.1f, chi1=%.3f, m1=%.3g; single-spin]\n', ...
        rad2deg(median(theta_JL(w_insp),'omitnan')), median(L_orb(w_insp),'omitnan')/max(S1_mag,eps), spin1_chi, mass1);
fprintf('    [fixed ID axis: J_pred polar=%.2f deg azim=%.0f deg (tilt0=%g deg, %+dx branch); mean-Lhat axis polar=%.2f deg azim=%.0f deg used only for the sign pick]\n', ...
        rad2deg(acos(Jhat(3))), rad2deg(atan2(Jhat(2),Jhat(1))), spin1_tilt_deg, 3-2*jpick, ...
        rad2deg(acos(Jhat_mean(3))), rad2deg(atan2(Jhat_mean(2),Jhat_mean(1))));
fprintf('  theta_JL = %.2f +- %.2f deg  (stat %.3f over %.1f orbits, sys %.2f from sub-window drift)\n', ...
        rad2deg(tJL_med), rad2deg(sig_JL), rad2deg(sig_stat), n_orb, rad2deg(sig_sys));
fprintf('  propagated tilt (d theta_SL/d theta_JL = %.0fx): %.1f deg\n', ...
        amp_ctr, rad2deg(tilt_ctr));
fprintf('    1-sigma interval [%.1f, %.1f] deg; forward-map turning point theta_SL* = %.1f deg\n', ...
        rad2deg(tilt_lo), rad2deg(tilt_hi), rad2deg(tilt_star));

% ---------------------------------------------------------------------
%  Figure 1 -- strain (inertial vs co-precessing) and the QA GW frequency
% ---------------------------------------------------------------------
fig1 = figure(1); clf(fig1);
set(fig1, 'defaultTextInterpreter','latex', 'defaultLegendInterpreter','latex', ...
          'defaultAxesTickLabelInterpreter','latex');
set(fig1, 'Units','inches', 'Position',[1 1 7.5 5.5], 'Color','w');
tiledlayout(fig1, 2, 1, 'TileSpacing','compact', 'Padding','compact');

ax1 = nexttile; hold(ax1,'on'); box(ax1,'on'); grid(ax1,'on');
plot(ax1, tplot, real(h_inert_diag), '-', 'LineWidth', 0.9, 'Color', [0.70 0.70 0.72]);
plot(ax1, tplot, real(h_qa_diag),    '-', 'LineWidth', 1.4, 'Color', [0.00 0.45 0.70]);
xline(ax1, tret(i_on) * time_to_ms, '--', 'turn-on', 'Interpreter','latex', 'LabelVerticalAlignment','bottom', 'Color',[0.4 0.4 0.4]);
legend(ax1, {sprintf('$\\mathrm{Re}\\,(r\\,h^{\\rm NR}_{%d,%+d})$', diag_ell, diag_m), ...
             sprintf('$\\mathrm{Re}\\,(r\\,h^{\\rm QA}_{%d,%+d})$', diag_ell, diag_m)}, ...
       'Interpreter','latex','Location','northwest','NumColumns',2);
ylabel(ax1, sprintf('$r\\,h_{%d,%+d}/M$', diag_ell, diag_m), 'Interpreter','latex');
set(ax1, 'TickLabelInterpreter','latex','FontSize',15,'LabelFontSizeMultiplier',1.1);
xticklabels(ax1, []);
title(ax1, sprintf('QA co-precessing frame ($\\ell\\le%d$ axis), $r_{\\rm ex}=%g$', lmax_axis, extr_radius), ...
      'Interpreter','latex');

ax3 = nexttile; hold(ax3,'on'); box(ax3,'on'); grid(ax3,'on');
plot(ax3, tplot, f_qa_hz,  '-', 'LineWidth', 1.4, 'Color', [0.00 0.45 0.70]);
xline(ax3, tret(i_on) * time_to_ms, '--', 'Color',[0.4 0.4 0.4], 'HandleVisibility','off');
legend(ax3, {sprintf('$f_{\\rm GW}$ QA $(%d,%+d)$', diag_ell, diag_m)}, ...
       'Interpreter','latex','Location','northwest');
ylabel(ax3, '$f_{\rm GW}\ [{\rm Hz}]$', 'Interpreter','latex');
ylim(ax3, [0, max(3*f_qa_hz(i_mrg), 1)]);
xlabel(ax3, '$t_{\rm ret}\ [{\rm ms}]$', 'Interpreter','latex');
set(ax3, 'TickLabelInterpreter','latex','FontSize',15,'LabelFontSizeMultiplier',1.1);
linkaxes([ax1 ax3], 'x'); xlim(ax1, xspan);

% ---------------------------------------------------------------------
%  Figure 2 -- precession-frame diagnostics
% ---------------------------------------------------------------------
fig2 = figure(2); clf(fig2);
set(fig2, 'defaultTextInterpreter','latex', 'defaultLegendInterpreter','latex', ...
          'defaultAxesTickLabelInterpreter','latex');
set(fig2, 'Units','inches', 'Position',[1 1 7.5 8], 'Color','w');
tl2 = tiledlayout(fig2, 3, 1, 'TileSpacing','compact', 'Padding','compact');
tlAB = tiledlayout(tl2, 2, 1, 'TileSpacing','none', 'Padding','tight');   % join top+middle (no gap)
tlAB.Layout.Tile = 1; tlAB.Layout.TileSpan = [2 1];

axA = nexttile(tlAB); box(axA,'on'); grid(axA,'on');
yyaxis(axA, 'left'); hold(axA,'on');
h_theta = plot(axA, tplot(win), rad2deg(theta_L(win)), '-', 'LineWidth',1.4, 'Color',[0.00 0.45 0.70]);
h_theta_a = plot(axA, tplot(win), rad2deg(theta_L_a(win)), '--', 'LineWidth',1.0, 'Color',[0.00 0.45 0.70]);
ylabel(axA, '$\theta_L$ [deg]', 'Interpreter','latex');
axA.YAxis(1).Color = [0.00 0.45 0.70];
yyaxis(axA, 'right'); hold(axA,'on');
h_phi = plot(axA, tplot(win), rad2deg(phi_L(win)), '-', 'LineWidth',1.2, 'Color',[0.84 0.37 0.00]);
h_phi_a = plot(axA, tplot(win), rad2deg(phi_L_a(win)), '--', 'LineWidth',1.0, 'Color',[0.84 0.37 0.00]);
ylabel(axA, '$\varphi_L$ [deg]', 'Interpreter','latex');
axA.YAxis(2).Color = [0.84 0.37 0.00];
legend(axA, [h_theta h_theta_a h_phi h_phi_a], ...
       {'$\theta_L$','$\theta_L$ analytic','$\varphi_L$','$\varphi_L$ analytic'}, ...
       'Interpreter','latex','Location','northwest','NumColumns',2);
title(axA, 'orbital-axis direction (O''Shaughnessy $\langle L_aL_b\rangle$)', 'Interpreter','latex');
set(axA, 'TickLabelInterpreter','latex','FontSize',15,'LabelFontSizeMultiplier',1.1); xticklabels(axA,[]);
axA.Toolbar.Visible = 'off';

axB = nexttile(tlAB); hold(axB,'on'); box(axB,'on'); grid(axB,'on');
plot(axB, tplot(w_insp), rad2deg(psi_L(w_insp)), '-', 'LineWidth',1.4, 'Color',[0.00 0.62 0.45]);
plot(axB, tplot(w_insp), rad2deg(psi_L_a(w_insp)), '--', 'LineWidth',1.0, 'Color',[0.00 0.62 0.45]);
legend(axB, {'$\psi_L$ (minimal rotation)','$\psi_L$ analytic'}, ...
       'Interpreter','latex','Location','southwest','NumColumns',2);
ylabel(axB, '$\psi_L$ [deg]', 'Interpreter','latex');
set(axB, 'TickLabelInterpreter','latex','FontSize',15,'LabelFontSizeMultiplier',1.1); xticklabels(axB,[]);
axB.Toolbar.Visible = 'off';

% Cone-inferred spin-orbit tilt and propagated uncertainty.
axC = nexttile(tl2, 3); hold(axC,'on'); box(axC,'on'); grid(axC,'on');
fill(axC, [tplot(iiw); flipud(tplot(iiw))], ...
     rad2deg([tSL_lo(iiw); flipud(tSL_hi(iiw))]), ...
     [0.64 0.08 0.18], 'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility','off');
h_tilt = plot(axC, tplot(w_insp), rad2deg(theta_SL(w_insp)), '-', ...
              'LineWidth',1.4, 'Color',[0.64 0.08 0.18]);
h_input = yline(axC, rad2deg(tilt0), ':', 'LineWidth',1.2, 'Color',[0.20 0.20 0.20]);
legend(axC, [h_tilt h_input], ...
       {'$\theta_{\rm SL}$','input $\iota$'}, ...
       'Interpreter','latex','Location','best','NumColumns',2);
ylabel(axC, '$\theta_{\rm SL}$ [deg]', 'Interpreter','latex');
title(axC, sprintf('cone-inferred spin--orbit tilt ($\\chi_1$=%.2f)', spin1_chi), 'Interpreter','latex');
set(axC, 'TickLabelInterpreter','latex','FontSize',15,'LabelFontSizeMultiplier',1.1);
axC.Toolbar.Visible = 'off';
xlabel(axC, '$t_{\rm ret}\ [{\rm ms}]$', 'Interpreter','latex');
linkaxes([axA axB axC], 'x'); xlim(axA, [tret(i_on), tret(i_mrg)]*t_unit*1e3);

% ---------------------------------------------------------------------
%  Optional outputs
% ---------------------------------------------------------------------
if save_waveform == 1
    outdat = fullfile(data_dir, 'strain_qa22.dat');
    hdr = sprintf(['# t[0]  Re(r*h^QA_22)[1]  Im(r*h^QA_22)[2]', ...
                   '   (l<=%d, r_ex=%g, extrap=%d, M=%g, a=%g)\n'], ...
                  lmax, extr_radius, extrapolate_to_infinity, adm_mass, background_spin);
    hqa22 = h_qa(:, find(mode_ell==2&mode_m==2,1));
    fid = fopen(outdat,'w'); fprintf(fid, '%s', hdr);
    fprintf(fid, '% .8e  % .8e  % .8e\n', [t, real(hqa22), imag(hqa22)].');
    fclose(fid);
    fprintf('  wrote %s\n', outdat);
end
if save_plot == 1
    outdir = fullfile(root, 'code_paper', 'plot');
    if exist(outdir,'dir') ~= 7, mkdir(outdir); end
    outfile = fullfile(outdir, plot_name + "_frame.pdf");
    assert(~isempty(findall(fig2, 'Type','axes')), 'Psi4:emptyFigure', ...
           'The precession-frame figure contains no axes.');
    drawnow;
    exportgraphics(tl2, outfile, 'ContentType','vector', 'BackgroundColor','white');
    outfile_info = dir(outfile);
    assert(~isempty(outfile_info) && outfile_info.bytes > 10000, 'Psi4:blankExport', ...
           'The exported precession-frame PDF appears to be blank.');
    fprintf('  wrote %s (%d bytes)\n', outfile, outfile_info.bytes);
end
