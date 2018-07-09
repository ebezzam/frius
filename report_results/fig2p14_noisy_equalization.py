import numpy as np
import os, datetime, time, warnings
from joblib import Parallel, delayed

import plot_settings
import matplotlib.pyplot as plt
ALPHA = 0.7
MARKER_SIZE = 10

from test_utilities import process_fig2p14

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..",))
from frius import total_freq_response, distance2time

from frius import create_pulse_param, sample_iq, add_noise, estimate_fourier_coeff, cadzow_denoising, compute_ann_filt, estimate_time_param, estimate_amplitudes, gen_fri, compute_srr_db_points, sample_rf, compute_srr_db

"""
For K = 10 pulses vary the SNR of pulse shape estimate.

Compare Cadzow vs. GenFRI
"""

if __name__ == '__main__': 

    # load data
    results_dir = 'noisy_equalization_10_06_15_17h39'

    # test parameters
    n_diracs = 10
    max_ini = 5                 # for GenFRI
    cadzow_iter = 20            # for Cadzow + TLS, even number
    snr_vals = np.arange(35, -1, -5)
    oversample_fact = 10
    n_trials = 50
    n_jobs = 5
    
    # load results if available, otherwise run test
    try:
        npzfile = np.load(os.path.join(os.path.dirname(__file__), results_dir, "results.npz"))
        n_diracs = npzfile['n_diracs']
        tk_err = npzfile['tk_err']
        sig_err = npzfile['sig_err']
        sig_err_gen = npzfile['sig_err_gen']
        tk_err_gen = npzfile['tk_err_gen']
        snr_vals = npzfile['snr_vals']
        print("Loading data from %s..." % results_dir)
        run_sweep = False
    except:
        run_sweep = True
        print("No data available. Running test...")
        print()

    if run_sweep:

        depth = 5e-2  # in meters

        # constants
        clk_verasonics = 62.5e6
        center_freq = clk_verasonics/12
        samp_freq = clk_verasonics/3
        bw = 2/3
        bwr = -6
        n_cycles = 2.5
        speed_sound = 1540
        period = distance2time(depth, speed_sound)

        # set sampling
        M = oversample_fact*n_diracs
        n_samples = 2*M+1
        samp_bw = n_samples/period
        Ts = 1/samp_bw
        t_samp = np.arange(n_samples)*Ts

        # frequency coefficients to estimate
        freqs_fft = np.fft.fftfreq(n_samples, Ts)
        increasing_order = np.argsort(freqs_fft)
        freqs_fft = freqs_fft[increasing_order]

        # pulse shape
        freqs = freqs_fft+center_freq
        H_clean = total_freq_response(freqs, center_freq, bw, n_cycles, bwr)

        # forward mapping frequencies
        freqs_grid, t_samp_grid = np.meshgrid(freqs_fft, t_samp)
        idft_trunc = np.exp(2j*np.pi*freqs_grid*t_samp_grid)

        # sweep
        start_time = time.time()

        tk_err = np.zeros((len(snr_vals), n_trials))
        sig_err = np.zeros((len(snr_vals), n_trials))
        tk_err_gen = np.zeros((len(snr_vals), n_trials))
        sig_err_gen = np.zeros((len(snr_vals), n_trials))

        warnings.filterwarnings("ignore")
        for i, snr_db in enumerate(snr_vals):

            start_snr = time.time()

            print("SNR : %f" % snr_db, end=", ")

            res = Parallel(n_jobs=n_jobs)(
                delayed(process_fig2p14)(n_diracs, seed, period, 
                    H_clean, freqs, idft_trunc, samp_bw,
                    center_freq, bw, n_cycles, bwr, samp_freq,
                    snr_db, max_ini, cadzow_iter, oversample_fact)
                for seed in range(n_trials))
            tk_err[i,:] = np.array([tup[0] for tup in res])
            sig_err[i,:] = np.array([tup[1] for tup in res])
            tk_err_gen[i,:] = np.array([tup[2] for tup in res])
            sig_err_gen[i,:] = np.array([tup[3] for tup in res])

            snr_time = time.time()-start_snr
            print("%f sec" % snr_time)

        warnings.filterwarnings("default")
        tot_time = time.time() - start_time
        print("TOTAL TIME : %f min" % (tot_time/60))
        print()

        """
        Save results
        """
        time_stamp = datetime.datetime.now().strftime("%m_%d_%Hh%M")
        results_dir = os.path.join(os.path.dirname(__file__),"noisy_equalization_%d_%s" % (n_diracs, time_stamp))
        os.makedirs(results_dir)
        np.savez(os.path.join(results_dir, "results"), n_trials=n_trials, 
            n_diracs=n_diracs, snr_vals=snr_vals, oversample_fact=oversample_fact,
            sig_err=sig_err, tk_err=tk_err, sig_err_gen=sig_err_gen, 
            tk_err_gen=tk_err_gen)

        print("Results saved to %s" % results_dir)
        

    """ Visualize """
    avg_loc = np.mean(tk_err, axis=1)
    avg_sig = np.mean(sig_err, axis=1)
    avg_loc_gen = np.mean(tk_err_gen, axis=1)
    avg_sig_gen = np.mean(sig_err_gen, axis=1)

    plt.figure()
    plt.plot(snr_vals, avg_loc, 'b-', marker='^', markersize=MARKER_SIZE,
        alpha=ALPHA, label="standard")
    plt.plot(snr_vals, avg_loc_gen, 'g-', marker='o', markersize=MARKER_SIZE,
        alpha=ALPHA, label="general")
    plt.xlabel("SNR [dB]")
    plt.ylabel("$ SRR_{dev}$ [dB]")
    plt.ylim([min(np.floor(np.min(avg_loc)/5)*5,np.floor(np.min(avg_loc_gen)/5)*5)-5,\
        max(np.round(np.max(avg_loc)/5)*5, np.round(np.max(avg_loc_gen)/5)*5)+5])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    fp = os.path.join(os.path.dirname(__file__), "figures", "_fig2p14a.pdf")
    plt.savefig(fp, dpi=300)

    plt.figure()
    plt.plot(snr_vals, avg_sig, 'b-', marker='^', markersize=MARKER_SIZE,
        alpha=ALPHA, label="standard")
    plt.plot(snr_vals, avg_sig_gen, 'g-', marker='o', markersize=MARKER_SIZE,
        alpha=ALPHA, label="general")
    plt.xlabel("SNR [dB]")
    plt.ylabel("$ SRR$ [dB]")
    plt.ylim([-5,max(np.round(np.max(avg_sig)/5)*5, np.round(np.max(avg_sig_gen)/5)*5)+5])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    fp = os.path.join(os.path.dirname(__file__), "figures", "_fig2p14b.pdf")
    plt.savefig(fp, dpi=300)

    plt.show()
