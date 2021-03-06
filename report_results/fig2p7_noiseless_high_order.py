import numpy as np
import os
import plot_settings
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..",))

from frius import create_pulse_param, compute_ann_filt, \
    estimate_time_param, estimate_amplitudes, evaluate_recovered_param, \
    compute_srr_db_points, compute_srr_db
from frius import distance2time, total_freq_response, sample_rf, sample_iq

"""
Critial sampling with higher pulse order to show "breakdown" of algorithm due
to high order.
"""

# constants
clk_verasonics = 62.5e6
center_freq = clk_verasonics/12
samp_freq = clk_verasonics/3
bw = 2/3
bwr = -6
n_cycles = 2.5
speed_sound = 1540
seed = 0

# user parameters
n_diracs = 30     # report Figure done with 15
depth = 5e-2  # in meters
viz = True

"""
Create FRI parameters
"""
period = distance2time(depth, speed_sound)
ck, tk = create_pulse_param(n_diracs, period=period, seed=seed)

"""
Critical sampling
"""
M = n_diracs
n_samples = 2*M+1
samp_bw = n_samples/period
Ts = 1/samp_bw

y_samp, t_samp = sample_iq(ck, tk, period, samp_bw, 
    center_freq, bw, n_cycles, bwr)

"""
Estimate FS coefficients of sum of diracs
"""
freqs_fft = np.fft.fftfreq(n_samples, Ts)
increasing_order = np.argsort(freqs_fft)
freqs_fft = freqs_fft[increasing_order]
Y = (np.fft.fft(y_samp))[increasing_order] / n_samples

# equalize
freqs = freqs_fft+center_freq
H_tot = total_freq_response(freqs, center_freq, bw, n_cycles, bwr)
fs_coeff_hat = Y / H_tot

"""
FRI recovery
"""
ann_filt = compute_ann_filt(fs_coeff_hat, n_diracs)
tk_hat = estimate_time_param(ann_filt, period)
ck_hat = estimate_amplitudes(fs_coeff_hat, freqs, tk_hat, period)

"""
Evaluate
"""
evaluate_recovered_param(ck, tk, ck_hat, tk_hat)

tk_err = compute_srr_db_points(tk, tk_hat)
print("Locations SRR : %f dB" % tk_err)

"""
Visualize recovery
"""
y_rf, t_rf = sample_rf(ck, tk, period, samp_freq,
                       center_freq, bw, n_cycles, bwr)
y_rf_resynth, t_rf = sample_rf(ck_hat, tk_hat, period, samp_freq,
                               center_freq, bw, n_cycles, bwr)
err_sig = compute_srr_db(y_rf, y_rf_resynth)
print("Resynthesized error : %f dB" % err_sig)

if viz:

    time_scal = 1e5

    """rf data + pulse locations"""
    plt.figure()
    plt.plot(time_scal*t_rf, y_rf, label="RF data", alpha=0.65)
    baseline = plt.stem(time_scal*tk, ck, 'g', markerfmt='go',label="Parameters")[2]
    plt.setp(baseline, color='g')
    baseline.set_xdata([0, time_scal*period])
    plt.xlim([0,time_scal*period])
    plt.grid()
    plt.xlabel("Time [%s seconds]" % str(1/time_scal))
    plt.tight_layout()
    plt.legend(loc='lower right')
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])

    """rf data + iq data"""
    norm_fact_rf = np.max(abs(y_rf))
    norm_fact_iq = np.max(abs(y_samp))

    plt.figure()
    plt.plot(time_scal*t_rf, y_rf/norm_fact_rf, label="RF data", alpha=0.65)
    plt.plot(time_scal*t_samp, np.real(y_samp)/norm_fact_iq, 'go-', label="Measured (I)",
             alpha=0.7)
    plt.plot(time_scal*t_samp, np.imag(y_samp)/norm_fact_iq, 'm^-', label="Measured (Q)",
             alpha=0.7)
    plt.xlabel("Time [%s seconds]" % str(1/time_scal))
    plt.grid()
    plt.tight_layout()
    plt.legend(loc='lower right')
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])

    """reconstruction"""
    plt.figure()
    baseline = plt.stem(time_scal*tk, ck, 'g', markerfmt='go', label="True")[2]
    plt.setp(baseline, color='g')
    baseline.set_xdata([0, time_scal*period])
    baseline = plt.stem(time_scal*tk_hat, ck_hat, 'r', markerfmt='r^',
                        label="Estimate")[2]
    plt.setp(baseline, color='r')
    baseline.set_xdata(([0, time_scal*period]))
    plt.xlabel("Time [%s seconds]" % str(1/time_scal))
    plt.xlim([0,time_scal*period])
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])
    fp = os.path.join(os.path.dirname(__file__), "figures", "_fig2p7a.pdf")
    plt.savefig(fp, dpi=300)

    """ resynthesized signal"""
    plt.figure()
    plt.plot(time_scal*t_rf, y_rf, label="True", alpha=0.65)
    plt.plot(time_scal*t_rf, y_rf_resynth, label="Estimate", alpha=0.65)
    plt.xlim([0,time_scal*period])
    plt.grid()
    plt.xlabel("Time [%s seconds]" % str(1/time_scal))
    plt.tight_layout()
    plt.legend(loc='lower right')
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])
    fp = os.path.join(os.path.dirname(__file__), "figures", "_fig2p7b.pdf")
    plt.savefig(fp, dpi=300)


plt.show()
