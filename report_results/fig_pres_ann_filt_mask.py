import numpy as np
import plot_settings
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from frius import create_pulse_param, compute_ann_filt, \
    estimate_time_param, estimate_amplitudes, evaluate_recovered_param
from frius import distance2time, total_freq_response, sample_rf, sample_iq

"""
Example of critial sampling.
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
n_diracs = 5
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


"""
Plot measured data, alongside typical RF data
"""
y_rf, t_rf = sample_rf(ck, tk, period, samp_freq, 
    center_freq, bw, n_cycles, bwr)

# compute annihilating filter in time domain
z = np.exp(-1j*2*np.pi*t_rf/period)
ann_filt_mask = np.polyval(ann_filt, z)
ann_filt_mask /= max(abs(ann_filt_mask))


if viz:

    """rf data + pulse locations"""
    time_scal = 1e5
    plt.figure()
    plt.plot(time_scal*t_rf, y_rf/max(y_rf), label="RF signal", alpha=0.65)
    plt.plot(time_scal*t_rf, np.abs(ann_filt_mask), 'r', alpha=0.65, label="Mask")
    baseline = plt.stem(time_scal*tk, ck/max(ck), 'g', markerfmt='go',label="Parameters")[2]
    plt.setp(baseline, color='g')
    baseline.set_xdata([0, time_scal*period])
    plt.xlim([0,time_scal*period])
    plt.grid()
    plt.xlabel("Time [%s seconds]" % str(1/time_scal))
    plt.tight_layout()
    plt.legend(loc='lower right')
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])
    # ax.axes.xaxis.set_ticklabels([])
    plt.savefig("ann_filt_mask.pdf", dpi=300)

    """rf data + iq data"""
    norm_fact_rf = np.max(abs(y_rf))
    norm_fact_iq = np.max(abs(y_samp))

    plt.figure()
    plt.plot(time_scal*t_rf, y_rf/norm_fact_rf, label="RF signal", alpha=0.65)
    plt.plot(time_scal*t_samp, np.real(y_samp)/norm_fact_iq, 'go-', 
        label="Measured (I)", alpha=0.7)
    plt.plot(time_scal*t_samp, np.imag(y_samp)/norm_fact_iq, 'm^-', 
        label="Measured (Q)", alpha=0.7)
    plt.xlabel("Time [%s seconds]" % str(1/time_scal))
    plt.grid()
    plt.tight_layout()
    plt.legend(loc='lower right')
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])
    # ax.axes.xaxis.set_ticklabels([])
    # plt.savefig("noiseless_iq_data.pdf", dpi=300)


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
    # ax.axes.xaxis.set_ticklabels([])
    # plt.savefig("noiseless_reconstruction.pdf", dpi=300)



print("Sampling rate (FRI) : %f" % samp_bw)
print("Sampling rate reduction from RF : %f" % (samp_freq/samp_bw))
print("Sampling rate reduction from IQ : %f" % (bw*center_freq/samp_bw))


plt.show()