import numpy as np
import time, datetime, os, warnings
from joblib import Parallel, delayed

import plot_settings
import matplotlib.pyplot as plt
ALPHA = 0.65
MARKER_SIZE = 10

from test_utilities import evaluate_tof_sorting

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..",))
from frius import Probe, Medium
from frius import tof_sort_pruning

"""
Test basic TOF sorting algorithm that includes all pruning for varying 
number of channels. 
"""

def process(seed, n_points):

    rng = np.random.RandomState(seed)

    # randomly place points
    medium = Medium(width=width, 
                    depth=max_depth,
                    min_depth=min_depth,
                    speed_sound=speed_sound,
                    n_points=n_points, seed=seed)
    x_coord_true = medium.x_coord
    z_coord_true = medium.z_coord

    # receive with each probe
    _n_comb = np.zeros(len(nrx_vals))
    _proc_time = np.zeros(len(nrx_vals))
    for nrx_idx, nrx in enumerate(nrx_vals):

        array_pos = probe_rx[nrx].array_pos

        # simulate recording
        recordings = probe_rx[nrx].record(medium, max_depth)

        # ground truth
        echo_sort_true = probe_rx[nrx].round_trip
        
        # shuffle and sort
        echo_unsorted = echo_sort_true.copy()
        for idx in range(nrx):
            rng.shuffle(echo_unsorted[idx,:])

        start_time = time.time()
        echo_sort_est, _n_comb[nrx_idx] = tof_sort_pruning(echo_unsorted, 
            array_pos, speed_sound,
            max_tof=True, pos_parab=True, remove=True)
        res = evaluate_tof_sorting(array_pos, x_coord_true, z_coord_true, echo_sort_est, 
            verbose=False)
        if not res:
            print("Num. points : %d, seed %d, NRX: %d" % (n_points, seed, nrx))
        _proc_time[nrx_idx] = (time.time()-start_time)

    return _n_comb, _proc_time


if __name__ == '__main__': 

    # results file
    results_dir = 'tof_increasing_nrx_06_17_18h26'

    # test parameters (if not loading file)
    n_jobs = 5
    n_points_vals = np.arange(30, 0, -5)
    nrx_vals = np.arange(6, 2, -1)
    n_trials = 100

    # load file if available, otherwise run test
    try:
        npzfile = np.load(os.path.join(os.path.dirname(__file__), results_dir, "results.npz"))
        n_points_vals = npzfile["n_points_vals"]
        nrx_vals = npzfile["nrx_vals"]
        n_comb = npzfile["n_comb"]
        proc_time = npzfile["proc_time"]
        print("Loading data from %s..." % results_dir)
        run_sweep = False
    except:
        run_sweep = True
        print("No data available. Running test...")
        print()

    if run_sweep:

        # test parameters
        min_depth = 0.01
        max_depth = 0.06
        n_elements_tx = 128

        # constants, probe parameters
        clk_verasonics = 62.5e6
        speed_sound = 1540
        center_freq = clk_verasonics/12
        samp_freq = clk_verasonics/3

        """ create probe """
        lamb = speed_sound/center_freq
        pitch_tx = lamb/2

        # make probe for reception
        probe_rx = dict()
        for nrx in nrx_vals:
            pitch_rx = (n_elements_tx-1)//(nrx-1) * pitch_tx
            probe_rx[nrx] = Probe(n_elements=nrx,
                            pitch=pitch_rx,
                            samp_freq=samp_freq,
                            dtype=np.float32)
            probe_rx[nrx].set_response(center_freq, 
                            n_cycles=2.5,
                            bandwidth=2/3, 
                            bwr=-6)

        width = (n_elements_tx-1)*pitch_tx

        """ sweep """
        n_comb = np.zeros((n_trials, len(n_points_vals), len(nrx_vals)))
        proc_time = np.zeros((n_trials, len(n_points_vals), len(nrx_vals)))

        start_test = time.time()

        warnings.filterwarnings("ignore")
        for seed in range(n_trials):

            start_trial = time.time()

            res = Parallel(n_jobs=n_jobs)(delayed(process)(seed, n_points) for n_points in n_points_vals)
            n_comb[seed] = np.array([tup[0] for tup in res])
            proc_time[seed] = np.array([tup[1] for tup in res])

            trial_time = time.time() - start_trial
            print("TRIAL (%d/%d) TIME : %f" % (seed+1, n_trials, trial_time))

        warnings.filterwarnings("default")
        print("TOTAL TIME : %f" % (time.time() - start_test))

        """
        Save results
        """
        time_stamp = datetime.datetime.now().strftime("%m_%d_%Hh%M")
        results_dir = os.path.join(os.path.dirname(__file__), "tof_increasing_nrx_%s" % (time_stamp))
        os.makedirs(results_dir)
        np.savez(os.path.join(results_dir, "results"), 
            n_trials=n_trials, n_points_vals=n_points_vals, nrx_vals=nrx_vals,
            n_comb=n_comb, proc_time=proc_time)

        print("Results saved to %s" % results_dir)

    """
    Visualize
    """
    avg_num_comb = np.mean(n_comb, axis=0)
    avg_proc_time = np.mean(proc_time, axis=0)

    plt.figure()
    markers = ['^', '<', '>', 'v', 'o']
    for nrx_idx, nrx in enumerate(nrx_vals):
        plt.plot(n_points_vals, avg_num_comb[:,nrx_idx], label="%d"%nrx, alpha=ALPHA, marker=markers[nrx_idx], markersize=MARKER_SIZE)
    plt.grid()
    plt.legend()
    plt.xlabel("Num. reflectors")
    plt.ylabel("Num. combinations")
    plt.tight_layout()
    fp = os.path.join(os.path.dirname(__file__), "figures", "_fig3p4a.pdf")
    plt.savefig(fp, dpi=300)

    plt.figure()
    for nrx_idx, nrx in enumerate(nrx_vals):
        plt.plot(n_points_vals, avg_proc_time[:,nrx_idx], label="%d"%nrx, alpha=ALPHA, marker=markers[nrx_idx], markersize=MARKER_SIZE)
    plt.grid()
    plt.legend()
    plt.xlabel("Num. reflectors")
    plt.ylabel("Processing time [s]")
    plt.tight_layout()
    fp = os.path.join(os.path.dirname(__file__), "figures", "_fig3p4b.pdf")
    plt.savefig(fp, dpi=300)

    plt.show()
