function matlab_spoq_parity_run
% Small MATLAB parity experiments for the minimal SPOQ solver.
%
% This script mirrors the Python outer loop:
%   grad = gradlplq(...)
%   A = condlplq(..., ro=0)
%   u = x - gamma * grad ./ A
%   x_next = proxPPXAplus(D, A/gamma, u, y, xi, J, prec)
%
% Outputs are written as CSV files under:
%   spoq_parity_outputs/matlab/<case_name>/

base_dir = fileparts(mfilename('fullpath'));
tool_dir = fullfile(base_dir, 'SPOQ-Sparse-Restoration-Toolbox-v1', 'SPOQ-Sparse-Restoration-Toolbox-v1.0', 'Tools');
addpath(tool_dir);

out_root = fullfile(base_dir, 'spoq_parity_outputs', 'matlab');
if ~exist(out_root, 'dir')
    mkdir(out_root);
end

params.alpha = 1e-3;
params.beta = 5e-2;
params.eta = 2e-1;
params.p = 1.0;
params.q = 2.0;
gamma = 1.0;
max_iter = 12;
J = 20000;
prec = 1e-16;

run_case('identity_2d', eye(2), [0.45; 0.05], [0.90; 0.70], 0.30, gamma, max_iter, J, prec, params, out_root);
run_case('nonidentity_2d', [1.0, 2.0; 0.0, 1.0], [0.55; 0.10], [0.85; 0.40], 0.28, gamma, max_iter, J, prec, params, out_root);

disp(['Wrote MATLAB parity traces to: ', out_root]);
end


function run_case(case_name, D, y, x0, xi, gamma, max_iter, J, prec, params, out_root)
case_dir = fullfile(out_root, case_name);
if ~exist(case_dir, 'dir')
    mkdir(case_dir);
end

x = x0(:);
n = length(x);
x_trace = zeros(max_iter + 1, n);
psi_trace = zeros(max_iter + 1, 1);
step_norms = zeros(max_iter + 1, 1);
feasibility = zeros(max_iter + 1, 1);
min_component = zeros(max_iter + 1, 1);

x_trace(1, :) = x';
psi_trace(1) = spoq_penalty_exact(x, params.alpha, params.beta, params.eta, params.p, params.q);
feasibility(1) = norm(D * x - y, 2) - xi;
min_component(1) = min(x);

for k = 1:max_iter
    grad = gradlplq(x, params.alpha, params.beta, params.eta, params.p, params.q);
    A = condlplq(x, params.alpha, params.beta, params.eta, params.p, params.q, 0);
    B = A ./ gamma;
    u = x - gamma * grad ./ A;
    x_next = proxPPXAplus(D, B, u, y, xi, J, prec);

    x_trace(k + 1, :) = x_next';
    psi_trace(k + 1) = spoq_penalty_exact(x_next, params.alpha, params.beta, params.eta, params.p, params.q);
    step_norms(k + 1) = norm(x_next - x, 2);
    feasibility(k + 1) = norm(D * x_next - y, 2) - xi;
    min_component(k + 1) = min(x_next);
    x = x_next;
end

writematrix(x_trace, fullfile(case_dir, 'x_trace.csv'));
writematrix(psi_trace, fullfile(case_dir, 'psi_trace.csv'));
writematrix(step_norms, fullfile(case_dir, 'step_norms.csv'));
writematrix(feasibility, fullfile(case_dir, 'feasibility_residual.csv'));
writematrix(min_component, fullfile(case_dir, 'min_component.csv'));
end


function value = spoq_penalty_exact(x, alpha, beta, eta, p, q)
lp_power = sum((x.^2 + alpha.^2).^(p/2) - alpha.^p);
if lp_power < 0
    lp_power = 0;
end
lq = (eta.^q + sum(abs(x).^q)).^(1/q);
value = log(((lp_power + beta.^p).^(1/p)) / lq);
end
