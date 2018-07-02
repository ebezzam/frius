import numpy as np
import h5py
import time
import os

import plot_settings
import matplotlib.pyplot as plt
ALPHA = 0.65

import sys
sys.path.append('..')
from frius import distance2time, time2distance, gen_echoes, estimate_2d_loc, das_beamform, image_bf_data
from frius import recover_parameters, compute_srr_db_points
from frius import tof_sort_pruning


"""
Proposed method for localization at the rate of innovation
"""

# algorithm parameters
oversample_fact = 5
n_elem_rx = 3
snr_db = 8
cadzow_iter = 20
K = 15
min_depth = None   # 0 for 7 dB  

"""
Load data
"""
raw_file = 'raw_data_15points_1pw_1p0cycles_1-05_13h08.h5'

seed = 0
hf = h5py.File(os.path.join('..', 'data', raw_file), 'r')

# extract all fields
rf_data = np.squeeze(np.array(hf['rf_data']))
time_vec = np.squeeze(np.array(hf['time_vec']))
true_positions = np.squeeze(np.array(hf['true_positions']))
center_freq = float(np.squeeze(np.array(hf['center_freq'])))
samp_freq = float(np.squeeze(np.array(hf['samp_freq'])))
samp_time = 1./samp_freq
speed_sound = float(np.squeeze(np.array(hf['speed_sound'])))
bandwidth = float(np.squeeze(np.array(hf['bandwidth'])))
n_cycles = float(np.squeeze(np.array(hf['num_cycles'])))
pitch = float(np.squeeze(np.array(hf['pitch'])))
n_elem_tx = int(np.squeeze(np.array(hf['n_elements'])))

probe_geometry = pitch*np.arange(n_elem_tx)
probe_geometry -= np.mean(probe_geometry)

x_true = true_positions[0,:]
z_true = true_positions[2,:]
true_tofs = gen_echoes(x_true, z_true, probe_geometry, speed_sound)
n_points = len(x_true)


""" Crop, add noise, and visualize signal """

if min_depth is None:
    min_depth = np.squeeze(np.array(hf['min_depth']))  # recover points after this

# prepare data
min_time = distance2time(min_depth, speed_sound)
first_sample = int(np.floor(min_time/samp_time))
t0 = samp_time*first_sample
rf_data_trunc = rf_data[:,first_sample:]
time_vec_trunc = time_vec[first_sample:]-t0

# adjust length for odd number of samples / FS coefficients
n_samples = rf_data_trunc.shape[1]
if not n_samples%2:   # need odd number of samples/fourier coefficients
    n_samples = n_samples-1
    rf_data_trunc = rf_data_trunc[:,:n_samples]
    time_vec_trunc = time_vec_trunc[:n_samples]

# add noise according to specified SNR
signal_std = np.linalg.norm(rf_data_trunc[0,:])
rng = np.random.RandomState(seed)
white_noise = rng.randn(rf_data_trunc.shape[0], rf_data_trunc.shape[1])
for k in range(n_elem_tx):
    white_noise_std = np.linalg.norm(white_noise[k,:])
    fact = signal_std * 10**(-snr_db/20) / white_noise_std
    white_noise[k,:] *= fact
noisy_rf = rf_data_trunc + white_noise

true_snr = 20*np.log10(np.linalg.norm(rf_data_trunc)/np.linalg.norm(white_noise))
print("SNR : %f dB" % true_snr)

# plot one channel
channel_idx = n_elem_tx//2
plt.figure()
plt.plot(time_vec_trunc+t0, noisy_rf[channel_idx,:], alpha=ALPHA, label="Noisy")
plt.plot(time_vec_trunc+t0, rf_data_trunc[channel_idx,:], alpha=ALPHA, label="Original")
plt.grid()
plt.xlim([t0, max(time_vec_trunc)+t0])
plt.xlabel("Time [s]")
plt.tight_layout()
plt.legend()
ax = plt.gca()
ax.axes.yaxis.set_ticklabels([])
plt.savefig("_fig4p2a.pdf", dpi=300)

"""
Subsample array at receive
"""
sub_idx_chan = np.linspace(0, n_elem_tx-1, n_elem_rx).astype(np.int)
print()
print("Used channels : ", end="")
print(sub_idx_chan)
print()
unsorted_echoes = np.zeros((n_elem_rx, n_points))


""" FRI to recover TOFs """

for k, rx_idx in enumerate(sub_idx_chan):

    print("CHANNEL %d/%d" % (rx_idx+1, n_elem_tx))

    # recover parameters
    _ck_hat, _tk_hat, duration = recover_parameters(
            noisy_rf[rx_idx,:], time_vec_trunc,
            K=K, oversample_freq=oversample_fact,
            center_freq=center_freq, bandwidth=bandwidth, 
            num_cycles=n_cycles)

    srr_dev = compute_srr_db_points(true_tofs[rx_idx,:], _tk_hat+t0)
    print("SRR_dev : %f" % srr_dev)

    unsorted_echoes[k,:] = _tk_hat+t0

"""
Sort echoes
"""
echo_sort_est = tof_sort_pruning(unsorted_echoes, 
    probe_geometry[sub_idx_chan], speed_sound)[0]

"""
Solve for 2D positions, intersection of parabolas
"""
x_est, z_est = estimate_2d_loc(probe_geometry[sub_idx_chan],
    echo_sort_est, speed_sound, avg=True)

true_coord = np.concatenate((x_true[:,np.newaxis], 
    z_true[:,np.newaxis]), axis=1) 
est_coord = np.concatenate((x_est[:,np.newaxis], 
    z_est[:,np.newaxis]), axis=1) 
srr_loc = compute_srr_db_points(true_coord, est_coord)
print("SRR_dev on 2D locations : %f" % srr_loc)


"""
Beamform image
"""
depth = time2distance(time_vec[-1], speed_sound)
bf_data = das_beamform(noisy_rf, samp_freq, pitch, probe_geometry, 
    center_freq, speed_sound, depth=depth, t_offset=t0)[0]

"""
Visualize
"""
scal_fact = 1e2
image_bf_data(bf_data, probe_geometry, max_depth=depth, scal_fact=scal_fact, dynamic_range=60, min_depth=min_depth)
plt.scatter(probe_geometry[sub_idx_chan]*scal_fact, 
    np.ones(len(sub_idx_chan))*min_depth*scal_fact, 
    c='r', marker= 's', label="RX lateral pos.", s=70)
plt.scatter(x_est*scal_fact, z_est*scal_fact, label="Estimate", 
    s=100, facecolors='none', edgecolors='r')
plt.xlabel("Lateral [cm]")
plt.ylabel("Axial [cm]")
plt.legend(loc=4, fontsize=15)
plt.tight_layout()
plt.savefig("_fig4p2b.png", dpi=300)

n_samples_fri = 2*K*oversample_fact+1
print("Num of samples (RF) : %d" % n_samples)
print("Num of samples (FRI) : %d" % n_samples_fri)
print("Sampling rate (FRI) : %f" % ((2*K*oversample_fact+1)/duration))
print("Sampling rate reduction : %f" % (n_samples/n_samples_fri))

print("Data rate reduction : %f" % (n_elem_tx*n_samples/n_elem_rx/n_samples_fri))
print("Compression total : %f" % ((n_elem_tx*n_samples)/(n_elem_rx*2*K)))

plt.show()

