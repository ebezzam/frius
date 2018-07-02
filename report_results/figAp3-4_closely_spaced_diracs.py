import numpy as np

# plot settings
import plot_settings
import matplotlib.pyplot as plt
ALPHA = 0.6

import sys
sys.path.append('..')
from frius import sample_ideal_project, estimate_fourier_coeff, compute_ann_filt, estimate_time_param, estimate_amplitudes, compute_srr_db_points, compute_srr_db

"""
Resolving two diracs that are close in noiseless situation.
"""


def compute_condition(t, dt):
    """
    Inputs are normalized times.

    `t` is the position of one dirac and `dt` is the distance between the two
    diracs.
    """

    u0 = np.exp(-1j*2*np.pi*t)
    u1 = u0*np.exp(-1j*2*np.pi*dt)

    # eigenvalues from quadratic formula
    a = 1
    b = -1*(1+u1)
    c = -1*(u0-u1)
    lamb_max = (-b + np.sqrt(b*b-4*a*c))/(2*a)
    lamb_min = (-b - np.sqrt(b*b-4*a*c))/(2*a)

    return np.abs(lamb_max)/np.abs(lamb_min)

period = 1
offset = 0  # place at center to avoid "circular" distance
n_vals = 100
min_sep = 1e-7   # fraction of period
max_sep = 1e-2      # fraction of period

# sweep
sep_vals = np.logspace(np.log10(min_sep), np.log10(max_sep), base=10.0, 
    num=n_vals)
sep_est = np.zeros(len(sep_vals))
srr_dev = np.zeros(len(sep_vals))
srr = np.zeros(len(sep_vals))
cond = np.zeros(len(sep_vals))
ck_hat = np.zeros((len(sep_vals), 2))
tk_hat = np.zeros((len(sep_vals), 2))
for i, sep in enumerate(sep_vals):

    tk = np.array([offset, offset+sep*period])
    K = len(tk)
    ck = np.ones(K)

    # sample
    y_samp, t_samp, fs_ind = sample_ideal_project(ck, tk, period)

    # recovery
    fs_coeff_hat = estimate_fourier_coeff(y_samp, t_samp, fs_ind=fs_ind)
    ann_filt = compute_ann_filt(fs_coeff_hat, K)
    _tk_hat = estimate_time_param(ann_filt, period)
    _ck_hat, cond[i] = estimate_amplitudes(fs_coeff_hat, fs_ind/period, 
        _tk_hat, period, return_cond=True)

    # evaluate
    sep_est[i] = abs(_tk_hat[0]-_tk_hat[1])
    srr_dev[i] = compute_srr_db_points(tk[1]+0.5, max(_tk_hat)+0.5)
    y_samp_est = sample_ideal_project(_ck_hat, _tk_hat, period)[0]
    srr[i] = compute_srr_db(y_samp, y_samp_est)
    ord_tk = np.argsort(_tk_hat)
    tk_hat[i] = _tk_hat[ord_tk]
    ck_hat[i] = _ck_hat[ord_tk]

# ground truth condition
cond_true = [compute_condition(offset/period, dt) for dt in sep_vals]


"""
Visualize
"""
plt.figure()
plt.loglog(sep_vals, cond, 'b--', label="Simulated", alpha=ALPHA)
plt.loglog(sep_vals, cond_true, 'g-', label="Theoretical", alpha=ALPHA)
plt.ylabel('Vandermonde condition')
plt.xlabel("Normalized separation")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("_figAp3a.pdf", format='pdf', dpi=300)

plt.figure()
plt.loglog(sep_vals, sep_est, alpha=ALPHA)
plt.ylabel("Estimated seperation")
plt.xlabel("Normalized separation")
plt.grid()
plt.tight_layout()

plt.figure()
plt.semilogx(sep_vals, srr_dev, alpha=ALPHA)
plt.ylabel("$SRR_{dev}$ [dB]")
plt.xlabel("Normalized separation")
plt.grid()
plt.tight_layout()

plt.figure()
plt.semilogx(sep_vals, srr, alpha=ALPHA)
plt.ylabel("SRR [dB]")
plt.xlabel("Normalized separation")
plt.grid()
plt.tight_layout()
plt.savefig("_figAp4a.pdf", format='pdf', dpi=300)

plt.figure()
plt.semilogx(sep_vals, ck_hat[:,0], 'g-', label="Estimated $c_0$", alpha=ALPHA)
plt.semilogx(sep_vals, ck_hat[:,1], 'b--', label="Estimated $c_1$", alpha=ALPHA)
plt.ylabel("Recovered amplitudes")
plt.xlabel("Normalized separation")
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig("_figAp4b.pdf", format='pdf', dpi=300)

tk = sep_vals*period
plt.figure()
plt.loglog(sep_vals, tk_hat[:,1], 'b--', label="Estimated $ t_1$", alpha=ALPHA)
plt.loglog(sep_vals, tk, 'g-', label="True $t_1$", alpha=ALPHA)
plt.xlabel("Normalized separation")
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig("_figAp3b.pdf", format='pdf', dpi=300)

plt.show()



    