import numpy as np
import operator, itertools
from scipy.linalg import svd
from scipy.interpolate import lagrange
import math

from .us_utils import estimate_2d_loc

def gram_from_edm(D):

    d1 =  D[:,0]
    ones_vec = np.ones(len(d1))
    return 0.5*(np.outer(ones_vec, d1)+np.outer(d1, ones_vec)-D)

def edm(X):
    """
    :param X: DxM matrix containing the locations where M is the
    number of points and D is the number of dimensions.
    """

    G = np.dot(X.T, X)
    diag_G = np.diag(G)
    ones_n = np.ones(X.shape[1])
    return np.outer(ones_n, diag_G) - 2*G + np.outer(diag_G, ones_n)

def gram(X):
    return np.dot(X.T, X)


def tof_sort_plain(echo_unsorted, array_pos, speed_sound, ref=0):
    """
    Basic TOF sorting without any pruning or post-processing.
    """

    # form EDM of known sensor positions
    nrx = len(array_pos)
    array_pos = np.vstack((array_pos, np.zeros(nrx)))
    dim = array_pos.shape[0]
    D = edm(array_pos)

    # initialize augmented matrix
    D_aug = np.zeros((nrx+1, nrx+1))
    D_aug[:nrx,:nrx] = D

    # select reference channel
    T1 = echo_unsorted[ref,:]
    other_channels = np.delete(np.arange(nrx), ref)

    # go through all combinations
    echo_sort_est = np.zeros(echo_unsorted.shape)
    for point_idx, t1 in enumerate(T1):

        # create all combos
        cand_per_chan_lst = []
        for k in other_channels:
            cand_per_chan_lst.append(list(echo_unsorted[k]))
        cand = list(itertools.product(*cand_per_chan_lst))

        # gram test on all combos
        score = np.inf
        est_match = np.zeros(nrx)
        for vec in cand:

            round_times = np.insert(np.array(vec), ref, t1)

            # remove estimate of tx dist
            tx_dist = estimate_2d_loc(array_pos[0], round_times[:,np.newaxis], speed_sound, avg=True)[1][0]
            if math.isnan(tx_dist):
                continue
            dk = speed_sound*round_times - tx_dist

            # augment EDM
            dk = dk*dk
            D_aug[-1,:nrx] = dk
            D_aug[:nrx,-1] = dk
            G = gram_from_edm(D_aug)
            s = svd(G, full_matrices=False, compute_uv=False)
            if s[dim] < score:
                score = s[dim]
                est_match = round_times

        echo_sort_est[:,point_idx] = est_match

    return echo_sort_est


def tof_sort_pruning(echo_unsorted, array_pos, speed_sound, ref=0, 
    max_tof=True, pos_parab=True, remove=True):
    """
    Basic TOF sorting with optional max TOF and positive curve pruning.
    """

    # form EDM of known sensor positions
    nrx = len(array_pos)
    array_pos = np.vstack((array_pos, np.zeros(nrx)))
    dim = array_pos.shape[0]
    D = edm(array_pos)
    pulse_cone = np.sqrt(D)/speed_sound

    # initialize augmented matrix
    D_aug = np.zeros((nrx+1, nrx+1))
    D_aug[:nrx,:nrx] = D

    # select reference channel
    T1 = echo_unsorted[ref,:]
    n_echoes = len(T1)
    other_channels = np.delete(np.arange(nrx), ref)

    # keep track of used echoes
    if remove:
        remaining_idx = dict()
        for k in range(nrx):
            remaining_idx[k] = list(np.arange(n_echoes))

    # go through all combinations
    echo_sort_est = np.zeros(echo_unsorted.shape)
    n_cand = 0
    for point_idx, t1 in enumerate(T1):

        # disregard already use echoes
        if remove:
            remaining_echos = []
            for k in range(nrx):
                remaining_echos.append(echo_unsorted[k][remaining_idx[k]])
        else:
            remaining_echos = echo_unsorted

        # max TOF diff pruning
        if max_tof:
            cand_per_chan, n_poss_pulse = pulse_cone_pruning(ref, t1, 
                remaining_echos, pulse_cone)
        else:
            cand_per_chan = remaining_echos

        # positive parabola
        if pos_parab:
            cand_per_chan_prun, n_poss_parab = parabola_pruning(ref, 
                array_pos[0], t1, cand_per_chan)
        else:
            cand_per_chan_prun = cand_per_chan

        # create all combos
        cand_per_chan_lst = []
        for k in other_channels:
            cand_per_chan_lst.append(list(cand_per_chan_prun[k]))
        cand = list(itertools.product(*cand_per_chan_lst))

        # gram test on all combos
        n_cand += len(cand)
        score = np.inf
        est_match = np.zeros(nrx)
        for vec in cand:

            round_times = np.insert(np.array(vec), ref, t1)

            # remove estimate of tx dist
            tx_dist = estimate_2d_loc(array_pos[0], round_times[:,np.newaxis], speed_sound, avg=True)[1][0]
            if math.isnan(tx_dist):
                continue
            dk = speed_sound*round_times - tx_dist

            # augment EDM
            dk = dk*dk
            D_aug[-1,:nrx] = dk
            D_aug[:nrx,-1] = dk
            G = gram_from_edm(D_aug)
            s = svd(G, full_matrices=False, compute_uv=False)
            if s[dim] < score:
                score = s[dim]
                est_match = round_times

        echo_sort_est[:,point_idx] = est_match

        # remove estimate from candidates for next echo
        if remove:
            for chan in other_channels:
                used_idx = np.argmin(abs(echo_unsorted[chan,:]-est_match[chan]))
                try:
                    remaining_idx[chan].remove(used_idx)
                except:
                    print("For channel %d, echo %d is used again." % (chan, used_idx))

    return echo_sort_est, n_cand
    

def tof_sort_greedy(echo_unsorted, array_pos, speed_sound, ref=0, verbose=False):

    """
    Test combinations by checking Gram matrix rank (d+1 singular value).

    Need at least (d+1) array elements.
    """

    nrx = len(array_pos)
    array_pos = np.vstack((array_pos, np.zeros(nrx)))
    dim = array_pos.shape[0]

    D = edm(array_pos)
    pulse_cone = np.sqrt(D)/speed_sound
    
    T1 = echo_unsorted[ref,:]
    other_channels = np.delete(np.arange(nrx), ref)
    three_chan_combos = list(itertools.combinations(np.arange(nrx), r=3))
    three_chan_combos = [list(combo) for combo in three_chan_combos]
    D4 = np.zeros((4, 4))

    sorted_echo_est_dict = dict()
    sorted_echo_chan_dict = dict()
    
    D_aug = np.zeros((nrx+1, nrx+1))
    D_aug[:nrx,:nrx] = D
    sorted_echo_est = np.zeros(echo_unsorted.shape)

    n_echoes = len(T1)
    total_comb = (n_echoes)**(nrx-1)
    pulse_pruning = np.zeros(n_echoes)
    parab_pruning = np.zeros(n_echoes)

    remaining_idx = dict()
    for k in range(nrx):
        remaining_idx[k] = list(np.arange(n_echoes))

    for point_idx, t1 in enumerate(T1):

        # disregard already use echoes
        remaining_echos = []
        for k in range(nrx):
            remaining_echos.append(echo_unsorted[k][remaining_idx[k]])

        # pulse cone pruning
        # cand_per_chan, n_poss_pulse = pulse_cone_pruning(ref, t1, 
        #         echo_unsorted, pulse_cone)
        cand_per_chan, n_poss_pulse = pulse_cone_pruning(ref, t1, 
                remaining_echos, pulse_cone)
        if verbose:
            print()
            print("NUM CAND %d" % point_idx)
            print("Removing prev echoes : %d" % n_poss_pulse)
        if n_poss_pulse == 0:
            cand_per_chan, n_poss_pulse = pulse_cone_pruning(ref, t1, 
                echo_unsorted, pulse_cone)
            if verbose:
                print("Keeping prev echoes : %d" % n_poss_pulse)

        if verbose:
            print("Pruning from pulse cone : %d/%d" % (n_poss_pulse, total_comb))
        pulse_pruning[point_idx] = n_poss_pulse

        # parabola pruning
        cand_per_chan_prun, n_poss_parab = parabola_pruning(ref, array_pos[0], t1, cand_per_chan)

        if verbose:
            print("Pruning from parabola cone : %d/%d" % (n_poss_parab, n_poss_pulse))
        parab_pruning[point_idx] = n_poss_parab

        # create all comboes
        cand_per_chan_lst = []
        for k in other_channels:
            cand_per_chan_lst.append(list(cand_per_chan_prun[k]))
        cand = list(itertools.product(*cand_per_chan_lst))

        # gram test on all combos
        score = np.inf
        est_match = np.zeros(nrx)
        for vec in cand:

            round_times = np.insert(np.array(vec), ref, t1)

            # remove estimate of tx dist, take average of all pairs?
            tx_dist = estimate_2d_loc(array_pos[0], round_times[:,np.newaxis], 
                speed_sound, avg=True)[1][0]
            if math.isnan(tx_dist):  # TODO : something more elegant here or inside `estimate_2d_loc`
                print("remove this dirty check")
                continue

            dk = speed_sound*round_times - tx_dist
            dk = dk*dk
            D_aug[-1,:nrx] = dk
            D_aug[:nrx,-1] = dk
            G = gram_from_edm(D_aug)

            # TODO check if G has neg entries --> not valid!

            s = svd(G, full_matrices=False, compute_uv=False)

            if s[dim] < score:
                score = s[dim]
                est_match = round_times

        sorted_echo_est[:,point_idx] = est_match

        # pick best combo of 3, echo might not be present in all channels
        sorted_echo_est_dict[point_idx] = est_match
        sorted_echo_chan_dict[point_idx] = list(np.arange(nrx))
        for combo in three_chan_combos:

            D4[:3,:3] = edm(array_pos[:, combo])
            round_times = est_match[combo]

            # remove estimate of tx dist
            tx_dist = estimate_2d_loc(array_pos[0][combo], round_times[:,np.newaxis], 
                speed_sound, avg=True)[1][0]
            if math.isnan(tx_dist):  # TODO : something more elegant here or inside `estimate_2d_loc`
                print("remove this dirty check 2")
                continue

            dk = speed_sound*round_times - tx_dist
            dk = dk*dk

            D4[-1,:3] = dk
            D4[:3,-1] = dk
            G = gram_from_edm(D4)
            s = svd(G, full_matrices=False, compute_uv=False)
            if s[dim] < score:
                score = s[dim]
                sorted_echo_est_dict[point_idx] = round_times
                sorted_echo_chan_dict[point_idx] = combo

        # remove estimate from candidates for next echo
        for k, chan in enumerate(sorted_echo_chan_dict[point_idx]):
            used_idx = np.argmin(abs(echo_unsorted[chan,:]-
                sorted_echo_est_dict[point_idx][k]))
            try:
                remaining_idx[chan].remove(used_idx)
            except:
                print("For channel %d, echo %d is used again." % (k, used_idx))
        # print(sorted_echo_chan_dict[point_idx])
        # for k in other_channels:
        #     print(k, end=" ")
        #     print(len(remaining_idx[k]))

    if verbose:
        print()
        print("Avg pulse pruning : %d/%d" % (np.mean(pulse_pruning), total_comb))
        print("Avg parab pruning : %d/%d" % (np.mean(parab_pruning), np.mean(pulse_pruning)))

    return sorted_echo_est, sorted_echo_est_dict, sorted_echo_chan_dict


def tof_sort_nde(echo_unsorted, array_pos, speed_sound, n_points=None, verbose=False):
    """
    Motivated by the scenario in which echoes do not appear across all channels.

    Need to make every channel a reference.
    """

    if n_points is None:
        n_points = echo_unsorted.shape[1]
    else:
        n_points = min(n_points, echo_unsorted.shape[1])

    # create known EDM
    D = edm(array_pos)
    pulse_cone = np.sqrt(D)/speed_sound
    dim = array_pos.shape[0]
    nrx = array_pos.shape[1]

    # initialize for rank test
    D_aug = np.zeros((nrx+1, nrx+1))
    D_aug[:nrx,:nrx] = D

    # initialize for final pruning where we take best combo of three (if better than all channels)
    three_chan_combos = list(itertools.combinations(np.arange(nrx), r=3))
    three_chan_combos = [list(combo) for combo in three_chan_combos]
    D4 = np.zeros((4, 4))

    # make each channel reference (since we can't assume echo falls in each channel)
    sorted_tof_est = []
    sorted_tof_chan = []
    sorted_tof_scores = []
    for ref in range(nrx):

        if verbose:
            print()
            print("-"*80)
            print("CHANNEL %d/%d AS REFERENCE" % (ref+1, nrx))


        # initialize data struct keeping track of used echoes
        remaining_idx = dict()
        for k in range(nrx):
            remaining_idx[k] = list(np.arange(echo_unsorted.shape[1]))
        other_channels = np.delete(np.arange(nrx), ref)

        # disregard already use echoes
        T1 = echo_unsorted[ref][remaining_idx[ref]]

        # check best combo with given reference
        sorted_tof_est_ref = dict()
        sorted_tof_chan_ref = dict()
        sorted_tof_score_ref = dict()
        for point_idx, t1 in enumerate(T1):

            if verbose:
                print("Trying TOF %d/%d..." % (point_idx+1, len(T1)))

            # disregard already use echoes
            remaining_echos = []
            for k in range(nrx):
                remaining_echos.append(echo_unsorted[k][remaining_idx[k]])

            # pulse cone pruning
            cand_per_chan, n_poss_pulse = pulse_cone_pruning(ref, t1, 
                remaining_echos, pulse_cone)
            if verbose:
                print("%d candidates after pulse cone pruning..." % n_poss_pulse)

            # parabola pruning
            cand_per_chan_prun, n_poss_parab = parabola_pruning(ref, array_pos[0], t1, cand_per_chan)
            if verbose:
                print("%d candidates after parabola pruning..." % n_poss_parab)

            # create all comboes
            cand_per_chan_lst = []
            for k in other_channels:
                cand_per_chan_lst.append(list(cand_per_chan_prun[k]))
            cand = list(itertools.product(*cand_per_chan_lst))

            # gram test on all combos
            cand_per_chan_lst = []
            for k in other_channels:
                cand_per_chan_lst.append(list(cand_per_chan_prun[k]))
            cand = list(itertools.product(*cand_per_chan_lst))
            if len(cand)==0:
                continue

            # gram test on all combos
            score = np.inf
            est_match = np.zeros(nrx)
            for vec in cand:

                round_times = np.insert(np.array(vec), ref, t1)

                # remove estimate of tx dist, take average of all pairs
                tx_dist = estimate_2d_loc(array_pos[0], round_times[:,np.newaxis], 
                    speed_sound, avg=True)[1][0]
                if math.isnan(tx_dist):
                    continue
                dk = speed_sound*round_times - tx_dist

                # augment EDM
                dk = dk*dk
                D_aug[-1,:nrx] = dk
                D_aug[:nrx,-1] = dk
                G = gram_from_edm(D_aug)
                s = svd(G, full_matrices=False, compute_uv=False)
                if s[dim] < score:
                    score = s[dim]
                    est_match = round_times

            # pick best combo of 3 if yields better score as echo might not be present in all channels
            sorted_tof_est_ref[point_idx] = est_match
            sorted_tof_chan_ref[point_idx] = list(np.arange(nrx))
            sorted_tof_score_ref[point_idx] = score

            for combo in three_chan_combos:

                D4[:3,:3] = edm(array_pos[:, combo])
                round_times = est_match[combo]

                # remove estimate of tx dist, take average of all pairs
                tx_dist = estimate_2d_loc(array_pos[0][combo], round_times[:,np.newaxis], 
                    speed_sound, avg=True)[1][0]
                if math.isnan(tx_dist):
                    continue
                dk = speed_sound*round_times - tx_dist

                # augment EDM
                dk = dk*dk
                D4[-1,:3] = dk
                D4[:3,-1] = dk
                G = gram_from_edm(D4)
                s = svd(G, full_matrices=False, compute_uv=False)
                if s[dim] < sorted_tof_score_ref[point_idx]:
                    sorted_tof_score_ref[point_idx] = s[dim]
                    sorted_tof_est_ref[point_idx] = round_times
                    sorted_tof_chan_ref[point_idx] = combo

            # remove estimate from candidates for next echo
            for k, chan in enumerate(sorted_tof_chan_ref[point_idx]):
                used_idx = np.argmin(abs(echo_unsorted[chan,:]-
                    sorted_tof_est_ref[point_idx][k]))
                try:
                    remaining_idx[chan].remove(used_idx)
                except:
                    if verbose:
                        print("For channel %d, echo %d is used again." % (k, used_idx))

        sorted_tof_est.append(sorted_tof_est_ref)
        sorted_tof_chan.append(sorted_tof_chan_ref)
        sorted_tof_scores.append(sorted_tof_score_ref)

    return sorted_tof_est, sorted_tof_chan, sorted_tof_scores


def pulse_cone_pruning(ref, t1, unsorted_echoes, pulse_cone, verbose=False):
    """
    Use pulse prunning as presented in:

    https://github.com/AdriBesson/ICASSP2018-pulse-stream-publication/blob/master/icassp_2018.pdf

    We can use the EDM matrix! For a particular pulse echo at t_i we know that
    the pulse in t_j has to be within +/- d_ij/c where d_ij is the distance 
    between sensor i and sensor j and c is the speed of sound.

    """

    n_elements_rx = len(pulse_cone)
    other_channels = np.delete(np.arange(n_elements_rx), ref)

    # for the echo value `t1` find the corresponding echoes in adjacent channels
    cand_per_chan = dict()
    n_poss = 1
    total_comb = 1
    for k in other_channels:

        diff = abs(unsorted_echoes[k] - t1)
        mask = (diff <= pulse_cone[ref,k])
        cand_per_chan[k] = unsorted_echoes[k][mask]

        n_poss *= len(cand_per_chan[k])
        total_comb *= len(unsorted_echoes[k])

    if verbose:
        print("Pulse pruned combinations : %d/%d" % (n_poss, total_comb))

    return cand_per_chan, n_poss


def parabola_pruning(ref, array_pos, t1, cand_per_chan, verbose=False):
    """
    Prune candidates for each channel by using prior information on the
    shape of the measurements, i.e. a parabola of the form:

    ax^2 + bx + c

    where `a` should be positive.

    With the reference, we take every three-point combination to make an
    interpolation function using Lagrange polynomials. If the interpolation
    function has a negative `a` coefficient we can drop that three point 
    combination from our candidates for the EDM rank test.
    """

    # determine all possible three-point combos
    n_elements_rx = len(array_pos)
    other_channels = np.delete(np.arange(n_elements_rx), ref)
    parab_interp_combos = list(itertools.combinations(other_channels, r=2))

    # initialize each channel with previous candidates
    cand_per_chan_prun = dict()
    max_echoes = 0
    for k in other_channels:
        cand_per_chan_prun[k] = cand_per_chan[k].copy()
        max_echoes = max(max_echoes, len(cand_per_chan_prun[k]))

    # prune candidates by picking those interpolation functions with positive x^2 coefficient
    n_interp_lagrange = 1
    for inter_comb in parab_interp_combos:

        cands_list = [cand_per_chan_prun[inter_comb[0]], cand_per_chan_prun[inter_comb[1]], np.array([t1])]
        parab_comb = list(itertools.product(*cands_list))

        # compute the parabolas
        x = [array_pos[inter_comb[0]], array_pos[inter_comb[1]], array_pos[ref]]
        parab_lagr = np.zeros(len(parab_comb))
        for k, y in enumerate(parab_comb):
            parab_lagr[k] =(y[0]/(x[0]-x[1])/(x[0]-x[2]) + \
               y[1]/(x[1]-x[0])/(x[1]-x[2]) + \
               y[2]/(x[2]-x[0])/(x[2]-x[1]))
        n_interp_lagrange += len(parab_lagr)

        # only take echoes which gave positive parabolas
        valid = [k for k, comb in enumerate(parab_comb) if parab_lagr[k] > 0]
        if not valid:
            continue
        s1, s2 = zip(*[comb[:2] for k, comb in enumerate(parab_comb) if parab_lagr[k] > 0])
        cand_per_chan_prun[inter_comb[0]] = np.intersect1d(cand_per_chan_prun[inter_comb[0]], np.array(s1))
        cand_per_chan_prun[inter_comb[1]] = np.intersect1d(cand_per_chan_prun[inter_comb[1]], np.array(s2))

    n_poss = 1
    prev_poss = 1
    for k in other_channels:
        cand_per_chan_prun[k] = np.unique(np.array(cand_per_chan_prun[k]))
        n_poss *= len(cand_per_chan_prun[k])
        prev_poss *= len(cand_per_chan[k])

    if verbose:
        print("Pruned combinations : %d/%d" % (n_poss, prev_poss))

    return cand_per_chan_prun, n_poss