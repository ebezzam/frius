import numpy as np
import h5py, os

import plot_settings
import matplotlib.pyplot as plt
ALPHA = 0.7

import sys
sys.path.append('..')
from frius import das_beamform, image_bf_data, time2distance, distance2time

"""
Visualize RF data for multichannel FRI.
"""

min_depth = 0.007
max_depth = 0.057


""" Load data """
raw_file = 'raw_data_20points_SNR50dB_1pw_1p0cycles_1-05_12h15.h5'
n_points = 20
fp = os.path.join(os.path.dirname(__file__), '..', 'data', raw_file)
f = h5py.File(fp, 'r')

# extract all fields
rf_data = np.squeeze(np.array(f['rf_data']))
time_vec = np.squeeze(np.array(f['time_vec']))
true_positions = np.squeeze(np.array(f['true_positions']))
center_freq = float(np.squeeze(np.array(f['center_freq'])))
samp_freq = float(np.squeeze(np.array(f['samp_freq'])))
samp_time = 1./samp_freq
speed_sound = float(np.squeeze(np.array(f['speed_sound'])))
bandwidth = float(np.squeeze(np.array(f['bandwidth'])))
num_cycles = float(np.squeeze(np.array(f['num_cycles'])))
pitch = float(np.squeeze(np.array(f['pitch'])))
n_elements = int(np.squeeze(np.array(f['n_elements'])))

period = time_vec[-1]+samp_time
probe_geometry = pitch*np.arange(n_elements)
probe_geometry -= np.mean(probe_geometry)

# extract locations of strong reflectors and corresponding echo locations across channels
x_true = true_positions[0,:][-n_points:]
z_true = true_positions[2,:][-n_points:]
print("%d strong reflectors" % n_points)
print("%d total reflectors" % true_positions.shape[1])


""" Crop and beamform """ 

# prepare data
min_time = distance2time(min_depth, speed_sound)
max_time = distance2time(max_depth, speed_sound)
first_sample = int(np.floor(min_time/samp_time))
last_sample = int(np.ceil(max_time/samp_time))
t0 = samp_time*first_sample
rf_data_trunc = rf_data[:,first_sample:last_sample]
time_vec_trunc = time_vec[first_sample:last_sample]-t0
duration = time_vec_trunc[-1] - time_vec_trunc[0] + samp_time
n_samples = rf_data_trunc.shape[1]

# check that beamformed image is still alright
scal_fact = 1e2
bf_data = das_beamform(rf_data_trunc, samp_freq, pitch, probe_geometry, 
    center_freq, speed_sound, depth=max_depth, t_offset=t0)[0]
image_bf_data(bf_data, probe_geometry, max_depth=max_depth, 
    min_depth=time2distance(t0, speed_sound), scal_fact = 1e2)
plt.ylabel('Axial [cm]')
plt.xlabel('Lateral [cm]')
plt.tight_layout()
fp = os.path.join(os.path.dirname(__file__), "figures", "_fig2p17a.png")
plt.savefig(fp, dpi=300)

# plot one channel
channel_idx = n_elements//2
plt.figure()
plt.plot(time_vec_trunc+t0, rf_data_trunc[channel_idx,:], alpha=ALPHA)
plt.xlim([min_time,max_time])
plt.grid()
plt.xlabel("Time [s]")
plt.tight_layout()
ax = plt.gca()
ax.axes.yaxis.set_ticklabels([])
fp = os.path.join(os.path.dirname(__file__), "figures", "_fig2p17b.pdf")
plt.savefig(fp, dpi=300)

plt.show()
