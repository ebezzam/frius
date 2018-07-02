import numpy as np
import os

# plotting settings
import plot_settings
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from frius import time2distance, das_beamform, image_bf_data


"""
User parameters
"""
min_depth = 0.01575
max_depth = 0.075

"""
Probe + raw date
"""
# probe/medium parameters
samp_freq = 50e6
center_freq = 5e6
dx = 0.9375*1e-3
speed_sound = 6300
bw_pulse = 1.0
pulse_dur = 500e-9
n_cycles = 0.5

# signed 16 bit integer [-128,127]
ndt_rawdata = np.genfromtxt(os.path.join('..', 'data', 'ndt_rawdata.csv'), delimiter=',')
n_samples = len(ndt_rawdata)
time_vec = np.arange(n_samples)/samp_freq
depth = time2distance(time_vec[-1], speed_sound)
n_elem_tx = ndt_rawdata.shape[1]

probe_geometry = np.arange(n_elem_tx)*dx


"""
DAS beamform 
"""
res = das_beamform(ndt_rawdata.T, samp_freq, 
    dx, probe_geometry, center_freq, speed_sound, depth)[0]
scal_fact = 1e2
image_bf_data(res, probe_geometry, depth, dynamic_range=40, scal_fact=scal_fact)
plt.xlabel("Lateral [cm]")
plt.ylabel("Axial [cm]")
plt.ylim([depth*scal_fact, 0])
plt.tight_layout()

plt.savefig("_fig4p4a.png", dpi=300)


"""
Single RF signal
"""
chan_idx = 0
scal_fact = 1e6
plt.figure()
plt.plot(scal_fact*time_vec, ndt_rawdata[:,chan_idx])
plt.grid()
plt.xlabel("Time [microseconds]")
ax = plt.gca()
ax.axes.yaxis.set_ticklabels([])
plt.tight_layout()
plt.savefig("_fig4p4b.pdf", dpi=300)


plt.show()