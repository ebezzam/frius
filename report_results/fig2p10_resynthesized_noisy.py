import os
import plot_settings
import matplotlib.pyplot as plt

from test_utilities import process_fig2p10

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..",))
from frius import distance2time

"""
Example of noisy samples for 20 dB.

For 10 Diracs oversample by factor of 8; for 20 Diracs oversample by a factor
of 4.

Over 100 samples seems to result in poor reconstruction...
"""

# user parameters
viz = True
snr_db = 20
cadzow_iter = 20    # even number

# constants
clk_verasonics = 62.5e6
center_freq = clk_verasonics/12
samp_freq = clk_verasonics/3
bw = 2/3
bwr = -6
n_cycles = 2.5
speed_sound = 1540
depth = 5e-2  # in meters
period = distance2time(depth, speed_sound)

# 10 diracs, 8x oversampled
process_fig2p10(n_diracs=10, period=period, snr_db=snr_db,
    center_freq=center_freq, bw=bw, n_cycles=n_cycles, bwr=bwr, 
    samp_freq=samp_freq, cadzow_iter=cadzow_iter, 
    oversample_fact=8, viz=viz)
fp = os.path.join(os.path.dirname(__file__), "figures", "_fig2p10a.pdf")
plt.savefig(fp, dpi=300)

# 10 diracs, 8x oversampled
process_fig2p10(n_diracs=20, period=period, snr_db=snr_db,
    center_freq=center_freq, bw=bw, n_cycles=n_cycles, bwr=bwr, 
    samp_freq=samp_freq, cadzow_iter=cadzow_iter, 
    oversample_fact=4, viz=viz)
fp = os.path.join(os.path.dirname(__file__), "figures", "_fig2p10b.pdf")
plt.savefig(fp, dpi=300)


plt.show()
