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
Compare different functions for computing the zeros/roots of a complex
polynomial.
"""

def compare_mpc(x, y):
    if abs(x.real) < abs(y.real):
        return x.real-y.real
    else:
        return y.real-x.real

def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K:
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K

def process(seed, K):
    """
    K is model order / number of zeros
    """

    # create the dirac locations with many, many points
    rng = np.random.RandomState(seed)
    tk = np.sort(rng.rand(K)*period)

    # true zeros
    uk = np.exp(-1j*2*np.pi*tk/period)
    coef_poly = poly.polyfromroots(uk)   # more accurate than np.poly
    
    # estimate zeros
    uk_hat = np.roots(np.flipud(coef_poly))
    uk_hat_poly = poly.polyroots(coef_poly)
    uk_hat_mpmath = mpmath.polyroots(np.flipud(coef_poly), maxsteps=100, 
        cleanup=True, error=False, extraprec=50)

    # compute error
    min_dev_norm = distance(uk, uk_hat)[0]
    _err_roots = 20*np.log10(np.linalg.norm(uk)/min_dev_norm)

    min_dev_norm = distance(uk, uk_hat_poly)[0]
    _err_poly = 20*np.log10(np.linalg.norm(uk)/min_dev_norm)

    # for mpmath, need to compute error with its precision
    uk = np.sort(uk)
    uk_mpmath = [mpmath.mpc(z) for z in uk]
    uk_hat_mpmath = sorted(uk_hat_mpmath, key=cmp_to_key(compare_mpc))
    dev = [uk_mpmath[k] - uk_hat_mpmath[k] for k in range(len(uk_mpmath))]
    _err_mpmath = 20*mpmath.log(mpmath.norm(uk_mpmath) / mpmath.norm(dev), 
            b=10)

    return _err_roots, _err_poly, _err_mpmath


if __name__ == '__main__': 

    # load data
    results_dir = 'zero_finding_06_13_21h37'

    # test parameters if no results file
    n_diracs_max = 100    # same number of zeros
    step_size = 5
    n_trials = 30
    n_jobs = 10
    depth_cm = 15
    prob_vals = []   # extra values to test

    try:
        npzfile = np.load(os.path.join(results_dir, "results.npz"))
        err_roots = npzfile['err_roots']
        err_poly = npzfile['err_poly']
        err_mpmath = npzfile['err_mpmath']
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

        # values to sweep
        n_diracs_vals = np.arange(n_diracs_max, 1, -step_size)
        n_diracs_vals = np.unique(np.concatenate((n_diracs_vals, prob_vals))).astype(int)

        err_roots = np.zeros((n_trials, len(n_diracs_vals)))
        err_poly = np.zeros((n_trials, len(n_diracs_vals)))
        err_mpmath = np.zeros((n_trials, len(n_diracs_vals)))

        for i in range(n_trials):

            print("TRIAL %d/%d" % (i+1, n_trials))

            start_trial = time.time()

            res = Parallel(n_jobs=n_jobs)(delayed(process)(i, K) for K in n_diracs_vals)
            err_roots[i,:] = np.array([tup[0] for tup in res])
            err_poly[i,:] = np.array([tup[1] for tup in res])
            err_mpmath[i,:] = np.array([tup[2] for tup in res])

            trial_time = time.time() - start_trial
            print("time : %f min" % (trial_time/60))


        """
        Save results
        """
        time_stamp = datetime.datetime.now().strftime("%m_%d_%Hh%M")
        results_dir = "zero_finding_%s" % time_stamp
        os.makedirs(results_dir)
        np.savez(os.path.join(results_dir, "results"), n_trials=n_trials, 
            n_diracs_vals=n_diracs_vals, err_roots=err_roots, err_poly=err_poly,
            err_mpmath=err_mpmath)

        print("Results saved to %s" % results_dir)


    """ Visualize """
    avg_roots = np.mean(err_roots,axis=0)
    avg_poly = np.mean(err_poly,axis=0)
    avg_mpmath = np.nanmean(err_mpmath,axis=0)

    std_roots = np.std(err_roots,axis=0)
    std_poly = np.std(err_poly,axis=0)
    std_mpmath = np.nanstd(err_mpmath,axis=0)

    plt.figure()
    plt.errorbar(n_diracs_vals, avg_roots, std_roots, fmt='b-', ecolor='b', 
        marker='o', label="np.roots", alpha=ALPHA)
    plt.errorbar(n_diracs_vals, avg_poly, std_poly, fmt='g-', ecolor='g', 
        marker='^', label="polyroots", alpha=ALPHA)
    plt.errorbar(n_diracs_vals, avg_mpmath, std_mpmath, fmt='m-', ecolor='m', 
        marker='v', label="mpmath", alpha=ALPHA)
    plt.grid()
    plt.ylabel("$ SRR_{dev}$ [dB]")
    plt.xlabel("Num. zeros (%d trials)" % n_trials)
    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.savefig(os.path.join(results_dir, "_figAp1.pdf"), format='pdf', dpi=300)

    plt.show()