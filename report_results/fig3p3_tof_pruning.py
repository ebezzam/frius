import numpy as np
import time, datetime, os, warnings
from joblib import Parallel, delayed

import plot_settings
import matplotlib.pyplot as plt
LEGEND_FONTSIZE = 15
ALPHA = 0.65
MARKER_SIZE = 10

from test_utilities import evaluate_tof_sorting

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..",))
from frius import Probe, Medium, estimate_2d_loc
from frius import compute_srr_db_points
from frius import tof_sort_pruning


"""
Test basic TOF sorting algorithm that includes pruning approaches. 
"""

def process(seed, n_points):

    rng = np.random.RandomState(seed)

    # randomly place points
    medium = Medium(width=width, 
                    depth=max_depth,
                    min_depth=min_depth,
                    speed_sound=speed_sound,
                    n_points=n_points, seed=seed)

    # simulate recording
    recordings = probe_rx.record(medium, max_depth)

    # ground truth
    echo_sort_true = probe_rx.round_trip
    x_coord_true = probe_rx.medium.x_coord
    z_coord_true = probe_rx.medium.z_coord

    # shuffle and sort
    echo_unsorted = echo_sort_true.copy()
    for idx in range(n_elements_rx):
        rng.shuffle(echo_unsorted[idx,:])

    # no pruning (estimate time for each test from `max tof` pruning)
    _n_comb = 0
    for K in range(n_points):
        _n_comb += (K+1)*(K+1)
    start_time = time.time()
    res = evaluate_tof_sorting(array_pos, x_coord_true, z_coord_true, 
        echo_sort_true, verbose=False)
    _proc_time = (time.time()-start_time)

    # max tof
    start_time = time.time()
    echo_sort_est, _n_comb_maxtof = tof_sort_pruning(echo_unsorted,
        array_pos, speed_sound,
        max_tof=True, pos_parab=False)
    _proc_time += (time.time()-start_time)/_n_comb_maxtof*_n_comb # estimate of no pruning!
    res = evaluate_tof_sorting(array_pos, x_coord_true, z_coord_true, 
        echo_sort_est, verbose=False)
    if not res:
        print("Num. points : %d, seed %d" % (n_points, seed))
    _proc_time_maxtof = (time.time()-start_time)

    # parab
    start_time = time.time()
    echo_sort_est, _n_comb_parab = tof_sort_pruning(echo_unsorted, 
        array_pos, speed_sound,
        max_tof=False, pos_parab=True)
    res = evaluate_tof_sorting(array_pos, x_coord_true, z_coord_true, 
        echo_sort_est, verbose=False)
    if not res:
        print("Num. points : %d, seed %d" % (n_points, seed))
    _proc_time_parab = (time.time()-start_time)

    # both pruning
    start_time = time.time()
    echo_sort_est, _n_comb_both = tof_sort_pruning(echo_unsorted, 
        array_pos, speed_sound,
        max_tof=True, pos_parab=True)
    res = evaluate_tof_sorting(array_pos, x_coord_true, z_coord_true, 
        echo_sort_est, verbose=False)
    if not res:
        print("Num. points : %d, seed %d" % (n_points, seed))
    _proc_time_both = (time.time()-start_time)

    return _proc_time, _proc_time_maxtof, _proc_time_parab, _proc_time_both, \
        _n_comb, _n_comb_maxtof, _n_comb_parab, _n_comb_both

if __name__ == '__main__': 

    # results file
    results_dir = 'tof_pruning_06_17_17h09'

    # test parameters (if not loading file)
    n_elements_rx = 3
    n_jobs = 5
    n_points_vals = np.arange(60, 0, -10)
    n_trials = 100

    # load file if available, otherwise run test
    try:
        npzfile = np.load(os.path.join(os.path.dirname(__file__), results_dir, "results.npz"))
        n_points_vals = npzfile["n_points_vals"]
        n_comb = npzfile["n_comb"]
        n_comb_maxtof = npzfile["n_comb_maxtof"]
        n_comb_parab = npzfile["n_comb_parab"]
        n_comb_both = npzfile["n_comb_both"]
        proc_time = npzfile["proc_time"]
        proc_time_maxtof = npzfile["proc_time_maxtof"]
        proc_time_parab = npzfile["proc_time_parab"]
        proc_time_both = npzfile["proc_time_both"]
        print("Loading data from %s..." % results_dir)
        run_sweep = False
    except:
        run_sweep = True
        print("No data available. Running test...")
        print()

    if run_sweep:

        # constants, probe parameters
        clk_verasonics = 62.5e6
        speed_sound = 1540
        center_freq = clk_verasonics/12
        samp_freq = clk_verasonics/3
        min_depth = 0.01
        max_depth = 0.06
        n_elements_tx = 128

        """ create probe """
        lamb = speed_sound/center_freq
        pitch_tx = lamb/2
        pitch_rx = (n_elements_tx-1)//(n_elements_rx-1) * pitch_tx

        # make probe for reception
        probe_rx = Probe(n_elements=n_elements_rx,
                        pitch=pitch_rx,
                        samp_freq=samp_freq,
                        dtype=np.float32)
        probe_rx.set_response(center_freq, 
                            n_cycles=2.5,
                            bandwidth=2/3, 
                            bwr=-6)

        width = probe_rx.array_pos[-1]-probe_rx.array_pos[0]
        array_pos = probe_rx.array_pos

        """ sweep """
        n_comb = np.zeros((len(n_points_vals), n_trials))
        n_comb_maxtof = np.zeros((len(n_points_vals), n_trials))
        n_comb_parab = np.zeros((len(n_points_vals), n_trials))
        n_comb_both = np.zeros((len(n_points_vals), n_trials))

        proc_time = np.zeros((len(n_points_vals), n_trials))
        proc_time_maxtof = np.zeros((len(n_points_vals), n_trials))
        proc_time_parab = np.zeros((len(n_points_vals), n_trials))
        proc_time_both = np.zeros((len(n_points_vals), n_trials))

        start_test = time.time()

        warnings.filterwarnings("ignore")
        for seed in range(n_trials):

            start_trial = time.time()

            res = Parallel(n_jobs=n_jobs)(delayed(process)(seed, n_points) for n_points in n_points_vals)
            proc_time[:,seed] = np.array([tup[0] for tup in res])
            proc_time_maxtof[:,seed] = np.array([tup[1] for tup in res])
            proc_time_parab[:,seed] = np.array([tup[2] for tup in res])
            proc_time_both[:,seed] = np.array([tup[3] for tup in res])

            n_comb[:,seed] = np.array([tup[4] for tup in res])
            n_comb_maxtof[:,seed] = np.array([tup[5] for tup in res])
            n_comb_parab[:,seed] = np.array([tup[6] for tup in res])
            n_comb_both[:,seed] = np.array([tup[7] for tup in res])

            trial_time = time.time() - start_trial
            print("TRIAL (%d/%d) TIME : %f" % (seed+1, n_trials, trial_time))

        warnings.filterwarnings("default")
        print("TOTAL TIME : %f" % (time.time() - start_test))

        """
        Save results
        """
        time_stamp = datetime.datetime.now().strftime("%m_%d_%Hh%M")
        results_dir = os.path.join(os.path.dirname(__file__), "tof_pruning_%s" % (time_stamp))
        os.makedirs(results_dir)
        np.savez(os.path.join(results_dir, "results"), 
            n_trials=n_trials, n_points_vals=n_points_vals, 
            n_comb=n_comb, n_comb_maxtof=n_comb_maxtof,
            n_comb_parab=n_comb_parab, n_comb_both=n_comb_both,
            proc_time=proc_time, proc_time_maxtof=proc_time_maxtof,
            proc_time_parab=proc_time_parab, proc_time_both=proc_time_both)

        print("Results saved to %s" % results_dir)


    """
    Visualize 
    """
    avg_num_comb = np.mean(n_comb, axis=1)
    avg_num_comb_max_tof = np.mean(n_comb_maxtof, axis=1)
    avg_num_comb_parab = np.mean(n_comb_parab, axis=1)
    avg_num_comb_both = np.mean(n_comb_both, axis=1)

    avg_proc = np.mean(proc_time, axis=1)
    avg_proc_max_tof = np.mean(proc_time_maxtof, axis=1)
    avg_proc_parab = np.mean(proc_time_parab, axis=1)
    avg_proc_both = np.mean(proc_time_both, axis=1)

    plt.figure()
    plt.plot(n_points_vals, avg_num_comb, label="Removing used", alpha=ALPHA, marker='<', markersize=MARKER_SIZE)
    plt.plot(n_points_vals, avg_num_comb_max_tof, label="Max TOF", alpha=ALPHA, marker='^', markersize=MARKER_SIZE)
    plt.plot(n_points_vals, avg_num_comb_parab, label="Pos parab", alpha=ALPHA, marker='v', markersize=MARKER_SIZE)
    plt.plot(n_points_vals, avg_num_comb_both, label="All", alpha=ALPHA, marker='>', markersize=MARKER_SIZE)
    plt.grid()
    plt.legend()
    plt.xlabel("Num. reflectors")
    plt.ylabel("Num. combinations")
    plt.tight_layout()
    fp = os.path.join(os.path.dirname(__file__), "figures", "_fig3p3a.pdf")
    plt.savefig(fp, dpi=300)

    plt.figure()
    plt.plot(n_points_vals, avg_proc, label="Removing used", alpha=ALPHA, marker='<', markersize=MARKER_SIZE)
    plt.plot(n_points_vals, avg_proc_max_tof, label="Max TOF", alpha=ALPHA, marker='^', markersize=MARKER_SIZE)
    plt.plot(n_points_vals, avg_proc_parab, label="Pos parab", alpha=ALPHA, marker='v', markersize=MARKER_SIZE)
    plt.plot(n_points_vals, avg_proc_both, label="All", alpha=ALPHA, marker='>', markersize=MARKER_SIZE)
    plt.grid()
    plt.legend()
    plt.xlabel("Num. reflectors")
    plt.ylabel("Processing time [s]")
    plt.tight_layout()
    fp = os.path.join(os.path.dirname(__file__), "figures", "_fig3p3b.pdf")
    plt.savefig(fp, dpi=300)

    plt.show()
