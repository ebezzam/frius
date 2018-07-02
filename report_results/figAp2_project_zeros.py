import numpy as np
import os, datetime, time
import plot_settings
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
ALPHA = 0.6

import numpy.polynomial.polynomial as poly
import mpmath
mpmath.mp.dps = 100

import sys
sys.path.append('..')
from frius import distance

"""
Observe effect of projecting zeros to unit circle.
"""

def process(seed, K):
    """
    K is model order / number of zeros
    """

    print(K, end=" ")

    # create the dirac locations with many, many points
    rng = np.random.RandomState(seed)
    tk = np.sort(rng.rand(K)*period)

    # true zeros
    uk = np.exp(-1j*2*np.pi*tk/period)
    coef_poly = poly.polyfromroots(uk)   # more accurate than np.poly
    
    # estimate zeros
    uk_hat = np.roots(np.flipud(coef_poly))

    # place on unit circle?
    uk_hat_unit = uk_hat / np.abs(uk_hat)

    # compute error
    min_dev_norm = distance(uk, uk_hat)[0]
    _err_roots = 20*np.log10(np.linalg.norm(uk)/min_dev_norm)

    min_dev_norm = distance(uk, uk_hat_unit)[0]
    _err_unit = 20*np.log10(np.linalg.norm(uk)/min_dev_norm)

    return _err_roots, _err_unit


if __name__ == '__main__': 

    # load data
    results_dir = 'project_zeros_06_13_22h28'

    # test parameters if no results file
    n_diracs_max = 200    # same number of zeros
    step_size = 15
    depth_cm = 15
    n_trials = 30
    n_jobs = 10
    prob_vals = np.array([500, 450, 400, 350, 300, 250])

    try:
        npzfile = np.load(os.path.join(results_dir, "results.npz"))
        err_roots = npzfile['err_roots']
        err_unit = npzfile['err_unit']
        n_trials = npzfile['n_trials']
        n_diracs_vals = npzfile['n_diracs_vals']
        print("Loading data from %s..." % results_dir)
        run_sweep = False
    except:
        run_sweep = True
        print("No data available. Running test...")
        print()


    if run_sweep:

        # physical constants
        speed_sound = 1540
        depth = depth_cm/100
        period = 2*depth/speed_sound

        n_diracs_vals = np.arange(n_diracs_max, 1, -step_size)
        n_diracs_vals = np.unique(np.concatenate((n_diracs_vals, prob_vals))).astype(int)

        err_roots = np.zeros((n_trials, len(n_diracs_vals)))
        err_unit = np.zeros((n_trials, len(n_diracs_vals)))

        for i in range(n_trials):

            print("TRIAL %d/%d" % (i+1, n_trials))

            start_trial = time.time()

            res = Parallel(n_jobs=n_jobs)(delayed(process)(i, K) for K in n_diracs_vals)
            err_roots[i,:] = np.array([tup[0] for tup in res])
            err_unit[i,:] = np.array([tup[1] for tup in res])

            trial_time = time.time() - start_trial
            print("time : %f min" % (trial_time/60))


        """
        Save results
        """
        time_stamp = datetime.datetime.now().strftime("%m_%d_%Hh%M")
        results_dir = "project_zeros_%s" % time_stamp
        os.makedirs(results_dir)
        np.savez(os.path.join(results_dir, "results"), n_trials=n_trials, 
            n_diracs_vals=n_diracs_vals, err_roots=err_roots, err_unit=err_unit)

        print("Results saved to %s" % results_dir)


    """ Visualize """
    avg_roots = np.mean(err_roots,axis=0)
    avg_unit = np.mean(err_unit,axis=0)

    std_roots = np.std(err_roots,axis=0)
    std_unit = np.std(err_unit,axis=0)

    plt.figure()
    plt.errorbar(n_diracs_vals, avg_roots, std_roots, fmt='b-', ecolor='b', 
        marker='o', label="original", alpha=ALPHA)
    plt.errorbar(n_diracs_vals, avg_unit, std_unit, fmt='g-', ecolor='g', 
        marker='^', label="unit circle", alpha=ALPHA)
    plt.grid()
    plt.ylabel("$ SRR_{dev}$ [dB]")
    plt.xlabel("Num. zeros (%d trials)" % n_trials)
    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.savefig(os.path.join(results_dir, "_figAp2.pdf"), format='pdf', dpi=300)

    plt.show()