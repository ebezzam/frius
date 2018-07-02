import numpy as np
import warnings

import plot_settings
import matplotlib.pyplot as plt
ALPHA = 0.7
LEGEND_FONT = 15

import sys
sys.path.append('..')
from frius import create_pulse_param, sample_ideal_project, estimate_fourier_coeff, compute_ann_filt, estimate_time_param, estimate_amplitudes, compute_srr_db_points, cadzow_denoising, gen_fri


"""
Place two Diracs but try to estimate one, showing induced model order error
for which oversampling and denoising is necessary.
"""

def visualize(ck, tk, ck_hat, tk_hat, y_samp, t_samp, ck_hat_gen=None, 
    tk_hat_gen=None):

    BW = 1 / t_samp[1]
    plt.figure()
    baseline = plt.stem(tk, ck, 'g', markerfmt='go', label="True", alpha=ALPHA)[2]
    plt.setp(baseline, color='g')
    baseline.set_xdata([0, period])
    plt.scatter(t_samp, y_samp/BW, label="Samples", alpha=ALPHA)
    plt.plot(t_samp, y_samp/BW, alpha=ALPHA)
    if ck_hat_gen is None:
        baseline = plt.stem(tk_hat, ck_hat, 'm', markerfmt='m^', label="Est.", 
            alpha=ALPHA)[2]
        plt.setp(baseline, color='m')
        baseline.set_xdata([0, period])
    else:
        baseline = plt.stem(tk_hat, ck_hat, 'm', markerfmt='m^', 
            label="Est. (standard)", alpha=ALPHA)[2]
        plt.setp(baseline, color='m')
        baseline.set_xdata([0, period])
        baseline = plt.stem(tk_hat_gen, ck_hat_gen, 'r', markerfmt='rv', 
            label="Est. (general)", alpha=ALPHA)[2]
        plt.setp(baseline, color='r')
        baseline.set_xdata([0, period])
    plt.xlim([0,period])
    plt.grid()
    plt.xlabel("Time [seconds]")
    plt.tight_layout()
    plt.legend(loc='upper right', fontsize=LEGEND_FONT)
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])


if __name__ == '__main__': 

    # create signal
    n_diracs = 2
    period = 1
    seed = 3
    K_rec = 1
    ck, tk = create_pulse_param(n_diracs, period, seed, unit_amp=True, pos=True)

    """ Critial sampling for 1 diracs, BUT not for two (assuming only 1) """
    y_samp, t_samp, fs_ind = sample_ideal_project(ck, tk, period, 
        K=K_rec)

    # recover with standard FRI
    fs_coeff = estimate_fourier_coeff(y_samp, t_samp, fs_ind)
    ann_filt = compute_ann_filt(fs_coeff, K_rec)

    # compute FRI parameters
    tk_hat = estimate_time_param(ann_filt, period)
    ck_hat = estimate_amplitudes(fs_coeff, fs_ind/period, tk_hat, period)

    # evaluate
    tk_err = compute_srr_db_points(tk, tk_hat)
    print("Critical sampling : %f dB" %tk_err)

    # visualize
    visualize(ck, tk, ck_hat, tk_hat, y_samp, t_samp)
    plt.savefig("_fig2p15a.pdf", format='pdf', dpi=300)

    """ Oversample with cadzow denoising + genfri """
    stop_cri = 'max_iter'; max_ini=7
    cadzow_iter = 20
    oversampling_time = 15
    n_samples = oversampling_time*(2*K_rec + 1)

    # sample
    y_samp, t_samp, fs_ind = sample_ideal_project(ck, tk, period, n_samples=n_samples,
        K=K_rec)

    # standard FRI with cadzow denoising
    fs_coeff = estimate_fourier_coeff(y_samp, t_samp, fs_ind)
    fs_coeff = cadzow_denoising(fs_coeff, K_rec, n_iter=cadzow_iter)
    ann_filt = compute_ann_filt(fs_coeff, K_rec)
    tk_hat = estimate_time_param(ann_filt, period)
    ck_hat = estimate_amplitudes(fs_coeff, fs_ind/period, tk_hat, period)

    # GenFRI, first build forward mapping
    freqs_grid, t_samp_grid = np.meshgrid(fs_ind/period, t_samp)
    G = np.exp(2j*np.pi*freqs_grid*t_samp_grid)
    warnings.filterwarnings("ignore")
    fs_coeff_gen, min_error, c_opt, ini = gen_fri(G, y_samp, K_rec, max_ini=max_ini, 
        stop_cri=stop_cri, seed=seed)
    warnings.filterwarnings("default")
    tk_hat_gen = estimate_time_param(c_opt, period)
    ck_hat_gen = estimate_amplitudes(fs_coeff_gen, fs_ind/period, tk_hat_gen, period)

    # evaluate
    tk_err = compute_srr_db_points(tk, tk_hat)
    print("%d oversampling in time with cadzow + tls: %f dB" % (oversampling_time, tk_err))

    tk_err_gen = compute_srr_db_points(tk, tk_hat_gen)
    print("%d oversampling in time with genfri : %f dB" % (oversampling_time, tk_err_gen))

    # visualize
    visualize(ck, tk, ck_hat, tk_hat, y_samp, t_samp, ck_hat_gen, tk_hat_gen)
    plt.savefig("_fig2p15b.pdf", format='pdf', dpi=300)


    plt.show()