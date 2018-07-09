import numpy as np
import os

# plotting settings
import plot_settings
import matplotlib.pyplot as plt
ALPHA = 0.8

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..",))
from frius import total_freq_response


# probe/medium parameters
samp_freq = 50e6
center_freq = 5e6
dx = 0.9375*1e-3
speed_sound = 6300
bw_pulse = 1.0
pulse_dur = 500e-9
n_cycles = 0.5

# higher sampling rate
duration = 3e-6
samp_freq_hi = 5*samp_freq
samp_time_hi = 1 / samp_freq_hi
n_samples = int(duration//samp_time_hi)
freqs = np.fft.fftfreq(n_samples, samp_time_hi)

H_tot = total_freq_response(freqs, center_freq, bw_pulse, n_cycles)
h_time = np.real(np.roll(np.fft.ifft(H_tot), n_samples//2))
t = np.arange(n_samples) * samp_time_hi

plt.figure()
plt.plot(t, h_time)
plt.grid()
plt.tight_layout()
plt.xlim([5e-7, 25e-7])
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
fp = os.path.join(os.path.dirname(__file__), "figures", "_fig4p3b.pdf")
plt.savefig(fp, dpi=300)

plt.show()
