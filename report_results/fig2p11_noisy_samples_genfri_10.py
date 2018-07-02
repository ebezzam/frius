import numpy as np
import os, datetime, time, warnings
from joblib import Parallel, delayed

import plot_settings
import matplotlib.pyplot as plt
ALPHA = 0.7
MARKER_SIZE = 10
LINESTYLE = '-'

from test_utilities import process_noisy_samples_gen

import sys
sys.path.append('..')
from frius import total_freq_response, distance2time

"""
For K = 10 pulses vary the SNR and oversampling rate. 

Apply GenFRI.
"""

if __name__ == '__main__': 

    # load data
    results_dir = 'noisy_samples_gen_10_06_14_19h25'

    # parameters for test (if not loading file)
    n_diracs = 10
    max_ini = 1
    n_jobs = 5
    n_trials = 200
    snr_vals = np.arange(50, 1, -5)
    oversampling_vals = np.arange(10, 1, -2)

    # load file available, otherwise run test
    try:
        npzfile = np.load(os.path.join(results_dir, "results.npz"))
        n_trials = npzfile['n_trials']
        n_diracs = npzfile['n_diracs']
        snr_vals = npzfile['snr_vals']
        oversampling_vals = npzfile['oversampling_vals']
        sig_err = npzfile['sig_err']
        tk_err = npzfile['tk_err']
        print("Loading data from %s..." % results_dir)
        run_sweep = False
    except:
        run_sweep = True
        print("No data available. Running test...")
        print()

    if run_sweep:

        # constants (typical US values)
        clk_verasonics = 62.5e6
        center_freq = clk_verasonics/12
        samp_freq = clk_verasonics/3
        bw = 2/3
        bwr = -6
        n_cycles = 2.5
        speed_sound = 1540
        depth = 5e-2  # in meters
        period = distance2time(depth, speed_sound)

        # sweep
        start_time = time.time()

        tk_err = np.zeros((len(oversampling_vals), len(snr_vals), n_trials))
        sig_err = np.zeros((len(oversampling_vals), len(snr_vals), n_trials))

        warnings.filterwarnings("ignore")
        for k, oversample_fact in enumerate(oversampling_vals):

            print("Oversampling factor : %f" % oversample_fact)

            # oversampling
            M = oversample_fact*n_diracs
            n_samples = 2*M+1
            samp_bw = n_samples/period
            Ts = 1/samp_bw
            t_samp = np.arange(n_samples)*Ts

            # frequencies within samples bandwidth (baseband)
            freqs_fft = np.fft.fftfreq(n_samples, Ts)
            increasing_order = np.argsort(freqs_fft)
            freqs_fft = freqs_fft[increasing_order]

            # pulse
            freqs = freqs_fft+center_freq
            H_tot = total_freq_response(freqs, center_freq, bw, n_cycles, bwr)

            # forward mapping
            freqs_grid, t_samp_grid = np.meshgrid(freqs_fft, t_samp)
            idft_trunc = np.exp(2j*np.pi*freqs_grid*t_samp_grid)
            G = idft_trunc*H_tot

            for i, snr_db in enumerate(snr_vals):

                print("SNR : %f" % snr_db)

                res = Parallel(n_jobs=n_jobs)(
                    delayed(process_noisy_samples_gen)(n_diracs, seed, period, 
                        G, freqs, center_freq, bw, n_cycles, bwr, samp_freq,
                        snr_db, max_ini, oversample_fact)  
                    for seed in range(n_trials))
                tk_err[k,i,:] = np.array([tup[0] for tup in res])
                sig_err[k,i,:] = np.array([tup[1] for tup in res])

        warnings.filterwarnings("default")
        tot_time = time.time() - start_time
        print("TOTAL TIME : %f min" % (tot_time/60))
        print()

        """
        Save results
        """
        time_stamp = datetime.datetime.now().strftime("%m_%d_%Hh%M")
        results_dir = "noisy_samples_gen_%d_%s" % (n_diracs, time_stamp)
        os.makedirs(results_dir)

        np.savez(os.path.join(results_dir, "results"), n_trials=n_trials, 
            n_diracs=n_diracs, snr_vals=snr_vals, oversampling_vals=oversampling_vals,
            sig_err=sig_err, tk_err=tk_err)

    """ Visualize """
    avg_loc_score = np.mean(tk_err, axis=2)
    avg_sig_score = np.mean(sig_err, axis=2)

    plt.figure()
    markers = ['>', '^', 'v', '<', 'o']
    for k, oversample_fact in enumerate(oversampling_vals):
        plt.plot(snr_vals, avg_loc_score[k], marker=markers[k], 
            markersize=MARKER_SIZE, linestyle=LINESTYLE,
            alpha=ALPHA, label="%dx"%oversample_fact)
    plt.xlabel("SNR [dB]")
    plt.ylabel("$ SRR_{dev}$ [dB]")
    plt.ylim([min(np.floor(np.min(avg_loc_score)/5)*5,0),np.round(np.max(avg_loc_score)/5)*5+5])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "_fig2p11a.pdf"), 
        format='pdf', dpi=300)

    plt.figure()
    for k, oversample_fact in enumerate(oversampling_vals):
        plt.plot(snr_vals, avg_sig_score[k], marker=markers[k], 
            markersize=MARKER_SIZE, linestyle=LINESTYLE,
            alpha=ALPHA, label="%dx"%oversample_fact)
    plt.xlabel("SNR [dB]")
    plt.ylabel("$ SRR$ [dB]")
    plt.ylim([min(np.floor(np.min(avg_sig_score)/5)*5,0), np.round(np.max(avg_sig_score)/5)*5+5])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "_fig2p11b.pdf"), 
        format='pdf', dpi=300)
    print("Results saved to %s" % results_dir)

    plt.show()