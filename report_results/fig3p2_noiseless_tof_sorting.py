import numpy as np
import h5py, os, time

import plot_settings
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from frius import Probe, Medium, estimate_2d_loc
from frius import compute_srr_db_points
from frius import tof_sort_plain

"""
Test basic TOF sorting algorithm that does not include pruning / removing 
used echoes / trying all channels as reference. 
"""

# test parameters
seed = 1
min_depth = 0.01
max_depth = 0.06
n_points = 20
n_elements_tx = 128
n_elements_rx = 3

# constants, probe parameters
rng = np.random.RandomState(seed)
clk_verasonics = 62.5e6
speed_sound = 1540
center_freq = clk_verasonics/12
samp_freq = clk_verasonics/3


""" Simulate measured TOFs (take ground truth and shuffle) """
lamb = speed_sound/center_freq
pitch_tx = lamb/2
pitch_rx = (n_elements_tx-1)//(n_elements_rx-1) * pitch_tx

# make probe for reception
probe_rx = Probe(n_elements=n_elements_rx,
                pitch=pitch_rx,
                samp_freq=samp_freq,
                oversample_fact=50, 
                dtype=np.float32)
probe_rx.set_response(center_freq, 
                    n_cycles=2.5,
                    bandwidth=2/3, 
                    bwr=-6)

# place points randomly in medium
width = probe_rx.array_pos[-1]-probe_rx.array_pos[0]
medium = Medium(width=width, 
                depth=max_depth,
                min_depth=min_depth,
                speed_sound=speed_sound,
                n_points=n_points, seed=seed)

# simulate recording
recordings = probe_rx.record(medium, max_depth, viz=True)
plt.savefig("_fig3p2b.pdf", format='pdf', dpi=300)
probe_rx.visualize_medium_rec()
plt.savefig("_fig3p2a.pdf", format='pdf', dpi=300)

# ground truth
echo_sort_true = probe_rx.round_trip
x_coord_true = probe_rx.medium.x_coord
z_coord_true = probe_rx.medium.z_coord
array_pos = probe_rx.array_pos

# shuffle and sort
echo_unsorted = echo_sort_true.copy()
for k in range(n_elements_rx):
    rng.shuffle(echo_unsorted[k,:])

echo_sort_est = tof_sort_plain(echo_unsorted, array_pos, speed_sound)

# solve for 2D location using quadratic formula
x_coord_est, z_coord_est = estimate_2d_loc(array_pos, echo_sort_est, 
    speed_sound, avg=True)

""" Evaluate """
true_coord = np.concatenate((x_coord_true[:,np.newaxis], 
    z_coord_true[:,np.newaxis]), axis=1) 
est_coord = np.concatenate((x_coord_est[:,np.newaxis], 
    z_coord_est[:,np.newaxis]), axis=1) 
srr_loc = compute_srr_db_points(true_coord, est_coord)
print("SRR on 2D locations : %f dB" % srr_loc)

plt.show()