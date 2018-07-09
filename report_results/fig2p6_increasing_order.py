import numpy as np
import os, datetime, time
import plot_settings
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from test_utilities import process_fig2p6

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..",))
from frius import total_freq_response, distance2time


"""
Increase number of pulses in noiseless situation. Compute SRR of resynthesized
signal and on time locations.

Set `results_dir` to path in order to load already computed data. Otherwise,
perform test by setting `results_dir` to e.g. `None`.

Parameters for test can be set in the main function.
"""


if __name__ == '__main__': 

    # load data
    results_dir = 'noiseless_increasing_order06_13_11h05'

    # parameters for test (if not loading file)
    max_n_diracs = 100
    step_size = 5
    n_diracs_vals = np.arange(start=max_n_diracs, stop=1, step=-step_size)
    n_diracs_vals = np.insert(n_diracs_vals, 0, [500, 400, 300, 200])
    n_trials = 50
    n_jobs = 5

    # load file available, otherwise run test
    try:
        npzfile = np.load(os.path.join(os.path.dirname(__file__), results_dir, "results.npz"))
        n_diracs_vals = npzfile["n_diracs_vals"]
        err_sig = npzfile["err_sig"]
        err_loc = npzfile["err_loc"]
        print("Loading data from %s..." % results_dir)
        run_sweep = False
    except:
        run_sweep = True
        print("No data available. Running test...")
        print()

    if run_sweep:

        # constants (typical US values)
        clk_verasonics = 62.5e6
        n_cycles = 2.5
        center_freq = clk_verasonics/12
        samp_freq = clk_verasonics/3
        speed_sound = 1540
        bw = 2/3
        bwr = -6
        depth = 5e-2  # in meters
        period = distance2time(depth, speed_sound)

        # sweep
        err_loc = np.zeros((len(n_diracs_vals), n_trials))
        err_sig = np.zeros((len(n_diracs_vals), n_trials))
        print("Number of pulses to sweep over : ", end="")
        print(n_diracs_vals)
        start_sweep = time.time()
        for i, K in enumerate(n_diracs_vals):

            print()
            print("Num. of pulses : %d" % K)
            n_diracs_time = time.time()

            # critical sampling parameters
            M = K
            n_samples = 2*M+1
            samp_bw = n_samples/period
            Ts = 1/samp_bw

            # frequencies within samples bandwidth (baseband)
            freqs_fft = np.fft.fftfreq(n_samples, Ts)
            increasing_order = np.argsort(freqs_fft)
            freqs_fft = freqs_fft[increasing_order]

            # pulse for equalizing
            freqs = freqs_fft+center_freq
            H_tot = total_freq_response(freqs, center_freq, bw, n_cycles, bwr)

            # random trials
            res = Parallel(n_jobs=n_jobs)(
                delayed(process_fig2p6)(K, j, period, H_tot, freqs, 
                    center_freq, bw, n_cycles, bwr, samp_freq) 
                for j in range(n_trials))
            res = np.array(res)

            err_loc[i,:] = res[:,0]
            err_sig[i,:] = res[:,1]

            avg_time = (time.time() - n_diracs_time)/n_trials
            print("Average reconstruction time for %d dirac(s) : %f sec" % (K, avg_time))

        """ Save """
        time_stamp = datetime.datetime.now().strftime("%m_%d_%Hh%M")
        results_dir = os.path.join(os.path.dirname(__file__), "noiseless_increasing_order%s" % time_stamp)
        os.makedirs(results_dir)
        np.savez(os.path.join(results_dir, "results"), n_diracs_vals=n_diracs_vals,
            err_sig=err_sig, err_loc=err_loc)
        print("Results saved to %s" % results_dir)

        print()
        print("TOTAL SIMULATION TIME : %f min" % ((time.time()-start_sweep)/60.) )


    """ Visualize """
    loc_err_per_ndiracs = np.mean(err_loc, axis=1)
    sig_err_per_ndiracs = np.mean(err_sig, axis=1)
    loc_std = np.std(err_loc, axis=1)
    sig_std = np.std(err_sig, axis=1)

    f, (ax1, ax2) = plt.subplots(2,1, sharex=True)

    ax1.errorbar(n_diracs_vals, sig_err_per_ndiracs, sig_std, ecolor='r', marker='o')
    ax1.set_ylabel("SRR [dB]")
    ax1.axes.set_yticks(np.arange(0,max(sig_err_per_ndiracs+sig_std),50))
    ax1.set_xscale("log", nonposx='clip')
    ax1.set_ylim([0, max(sig_err_per_ndiracs+sig_std)])
    ax1.grid()

    ax2.errorbar(n_diracs_vals, loc_err_per_ndiracs, loc_std, ecolor='r', marker='o')
    ax2.set_ylabel("$t_k$ [dB]")
    ax2.grid()
    ax2.axes.set_yticks(np.arange(0,max(loc_err_per_ndiracs+loc_std),50))
    ax2.set_xscale("log", nonposx='clip')
    ax2.set_xlim([min(n_diracs_vals), max(n_diracs_vals)])
    ax2.set_ylim([0, max(loc_err_per_ndiracs+loc_std)])
    ax2.set_xlabel("Num. of pulses [log scale]")
    f.tight_layout()

    fp = os.path.join(os.path.dirname(__file__), "figures", "_fig2p6.pdf")
    plt.savefig(fp, dpi=300)

    plt.show()

