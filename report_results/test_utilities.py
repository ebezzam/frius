import numpy as np

import sys
sys.path.append('..')

from frius import create_pulse_param, compute_ann_filt, estimate_time_param, \
    estimate_fourier_coeff, estimate_amplitudes, compute_srr_db_points, \
    compute_srr_db, cadzow_denoising, gen_fri
from frius import sample_iq, sample_rf, add_noise, total_freq_response, estimate_2d_loc


def process_fig2p6(n_diracs, seed, period, H_tot, freqs, 
    center_freq, bw, n_cycles, bwr, samp_freq):

    ck, tk = create_pulse_param(n_diracs, period=period, seed=seed)

    # critical sampling parameters
    samp_bw = (2*n_diracs+1)/period

    # sample
    y_samp, t_samp = sample_iq(ck, tk, period, samp_bw, 
        center_freq, bw, n_cycles, bwr)

    # estimate FS coeff
    fs_coeff_hat = estimate_fourier_coeff(y_samp, t_samp, H=H_tot)

    # FRI recovery
    ann_filt = compute_ann_filt(fs_coeff_hat, n_diracs)
    tk_hat = estimate_time_param(ann_filt, period)
    ck_hat = estimate_amplitudes(fs_coeff_hat, freqs, tk_hat, period)

    # compute errors
    tk_err = compute_srr_db_points(tk, tk_hat)

    y_rf, t_rf = sample_rf(ck, tk, period, samp_freq, 
        center_freq, bw, n_cycles, bwr)
    y_rf_resynth, t_rf = sample_rf(ck_hat, tk_hat, period, samp_freq, 
        center_freq, bw, n_cycles, bwr)
    resynth_err = compute_srr_db(y_rf, y_rf_resynth)

    return tk_err, resynth_err


def process_noisy_samples(n_diracs, seed, period, H_tot, freqs, 
    center_freq, bw, n_cycles, bwr, samp_freq,
    snr_db, cadzow_iter, oversample_fact):
    """
    Fig. 2.8-2.9
    """

    ck, tk = create_pulse_param(n_diracs, period=period, seed=seed)

    # oversample
    samp_bw = (2*oversample_fact*n_diracs+1)/period
    y_samp, t_samp = sample_iq(ck, tk, period, samp_bw, 
        center_freq, bw, n_cycles, bwr)
    y_noisy = add_noise(y_samp, snr_db, seed=seed)

    # estimate fourier coefficients
    fs_coeff_hat = estimate_fourier_coeff(y_noisy, t_samp, H=H_tot)

    # denoising + recovery
    fs_coeff_clean = cadzow_denoising(fs_coeff_hat, n_diracs, n_iter=cadzow_iter)
    ann_filt = compute_ann_filt(fs_coeff_clean, n_diracs)
    tk_hat = estimate_time_param(ann_filt, period)
    ck_hat = estimate_amplitudes(fs_coeff_clean, freqs, tk_hat, period)

    # compute errors
    _tk_err = compute_srr_db_points(tk, tk_hat)

    y_rf, t_rf = sample_rf(ck, tk, period, samp_freq, 
        center_freq, bw, n_cycles, bwr)
    y_rf_resynth, t_rf = sample_rf(ck_hat, tk_hat, period, samp_freq, 
            center_freq, bw, n_cycles, bwr)
    _sig_err = compute_srr_db(y_rf, y_rf_resynth)

    return _tk_err, _sig_err


def process_fig2p10(n_diracs, period, snr_db,
    center_freq, bw, n_cycles, bwr, samp_freq,
    cadzow_iter, oversample_fact, 
    viz=False, seed=0):
    """
    Fig. 2.8-2.9
    """

    # create FRI parameters
    ck, tk = create_pulse_param(n_diracs, period=period, seed=seed)

    # set oversampling
    M = oversample_fact*n_diracs
    n_samples = 2*M+1
    samp_bw = n_samples/period
    Ts = 1/samp_bw

    # oversample
    y_samp, t_samp = sample_iq(ck, tk, period, samp_bw, 
        center_freq, bw, n_cycles, bwr)
    y_noisy = add_noise(y_samp, snr_db, seed=seed)

    # estimate fourier coefficients
    freqs_fft = np.fft.fftfreq(n_samples, Ts)
    increasing_order = np.argsort(freqs_fft)
    freqs_fft = freqs_fft[increasing_order]
    freqs = freqs_fft+center_freq
    H_tot = total_freq_response(freqs, center_freq, bw, n_cycles, bwr)
    fs_coeff_hat = estimate_fourier_coeff(y_noisy, t_samp, H=H_tot)

    # denoising + recovery
    fs_coeff_clean = cadzow_denoising(fs_coeff_hat, n_diracs, n_iter=cadzow_iter)
    ann_filt = compute_ann_filt(fs_coeff_clean, n_diracs)
    tk_hat = estimate_time_param(ann_filt, period)
    ck_hat = estimate_amplitudes(fs_coeff_clean, freqs, tk_hat, period)

    # compute errors
    tk_err = compute_srr_db_points(tk, tk_hat)

    y_rf, t_rf = sample_rf(ck, tk, period, samp_freq, 
        center_freq, bw, n_cycles, bwr)
    y_rf_resynth, t_rf = sample_rf(ck_hat, tk_hat, period, samp_freq, 
            center_freq, bw, n_cycles, bwr)
    sig_err = compute_srr_db(y_rf, y_rf_resynth)

    print()
    print("%d Diracs, %.02fx oversampling:"%(n_diracs,oversample_fact))
    print("Locations SRR : %.02f dB" % tk_err)
    print("Resynthesized error : %.02fdB" % sig_err)
    

    """visualize"""
    if viz:

        import matplotlib.pyplot as plt
        time_scal = 1e5

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

        # resynthesized signal
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


def process_noisy_samples_gen(n_diracs, seed, period, G, freqs, 
    center_freq, bw, n_cycles, bwr, samp_freq,
    snr_db, max_ini, oversample_fact):
    """
    Fig. 2.11-2.12
    """

    stop_cri = 'max_iter'    # 'mse' or 'max_iter'

    ck, tk = create_pulse_param(n_diracs, period=period, seed=seed)

    # oversample
    samp_bw = (2*oversample_fact*n_diracs+1)/period
    y_samp, t_samp = sample_iq(ck, tk, period, samp_bw, 
        center_freq, bw, n_cycles, bwr)
    y_noisy = add_noise(y_samp, snr_db, seed=seed)

    # denoising + recovery
    fs_coeff_gen, min_error, c_opt, ini = gen_fri(G, y_noisy, n_diracs, 
        max_ini=max_ini, stop_cri=stop_cri, seed=seed)
    tk_hat = estimate_time_param(c_opt, period)
    ck_hat = estimate_amplitudes(fs_coeff_gen, freqs, tk_hat, period)

    # compute errors
    _tk_err = compute_srr_db_points(tk, tk_hat)

    y_rf, t_rf = sample_rf(ck, tk, period, samp_freq, 
        center_freq, bw, n_cycles, bwr)
    y_rf_resynth, t_rf = sample_rf(ck_hat, tk_hat, period, samp_freq, 
            center_freq, bw, n_cycles, bwr)
    _sig_err = compute_srr_db(y_rf, y_rf_resynth)

    return _tk_err, _sig_err


def process_fig2p14(n_diracs, seed, period, H_clean, freqs, idft_trunc, samp_bw,
    center_freq, bw, n_cycles, bwr, samp_freq,
    snr_db, max_ini, cadzow_iter, oversample_fact):

    ck, tk = create_pulse_param(n_diracs, period=period, seed=seed)
    y_samp, t_samp = sample_iq(ck, tk, period, samp_bw, 
        center_freq, bw, n_cycles, bwr)

    # add noise
    H_tot = add_noise(H_clean, snr_db, seed=seed)

    """ Cadzow + TLS """
    # estimate FS coefficients of sum of diracs
    fs_coeff_hat = estimate_fourier_coeff(y_samp, t_samp, H=H_tot)

    # denoise and recover parameters
    fs_coeff_clean = cadzow_denoising(fs_coeff_hat, n_diracs, n_iter=cadzow_iter)
    ann_filt = compute_ann_filt(fs_coeff_clean, n_diracs)
    tk_hat = estimate_time_param(ann_filt, period)
    ck_hat = estimate_amplitudes(fs_coeff_clean, freqs, tk_hat, period)

    """ GenFRI, denoise and recover parameters SIMULATANEOUSLY """
    G = idft_trunc*H_tot
    fs_coeff_gen, min_error, c_opt, ini = gen_fri(G, y_samp, n_diracs, 
        stop_cri='max_iter', max_ini=max_ini, seed=seed)
    tk_hat_gen = estimate_time_param(c_opt, period)
    ck_hat_gen = estimate_amplitudes(fs_coeff_gen, freqs, tk_hat_gen, period)

    """
    Evaluate 
    """
    _tk_err = compute_srr_db_points(tk, tk_hat)
    _tk_err_gen = compute_srr_db_points(tk, tk_hat_gen)

    y_rf, t_rf = sample_rf(ck, tk, period, samp_freq, 
        center_freq, bw, n_cycles, bwr)
    y_rf_resynth, t_rf = sample_rf(ck_hat, tk_hat, period, samp_freq, 
            center_freq, bw, n_cycles, bwr)
    _sig_err = compute_srr_db(y_rf, y_rf_resynth)

    y_rf_resynth_gen, t_rf = sample_rf(ck_hat_gen, tk_hat_gen, period, 
        samp_freq, center_freq, bw, n_cycles, bwr)
    _sig_err_gen = compute_srr_db(y_rf, y_rf_resynth_gen)

    return _tk_err, _sig_err, _tk_err_gen, _sig_err_gen


def evaluate_tof_sorting(array_pos, x_coord_true, z_coord_true, echo_sort_est, 
    speed_sound=1540, verbose=True):
    """
    return True if correct
    """

    # solve for 2D location using quadratic formula
    x_coord_est, z_coord_est = estimate_2d_loc(array_pos, echo_sort_est, 
        speed_sound, avg=True)

    """ Evaluate """
    true_coord = np.concatenate((x_coord_true[:,np.newaxis], 
        z_coord_true[:,np.newaxis]), axis=1) 
    est_coord = np.concatenate((x_coord_est[:,np.newaxis], 
        z_coord_est[:,np.newaxis]), axis=1) 
    srr_loc = compute_srr_db_points(true_coord, est_coord)

    if verbose:
        print() 
        print("SRR on 2D locations : %f dB" % srr_loc)

    if srr_loc < 200:
        return False
    else:
        return True

