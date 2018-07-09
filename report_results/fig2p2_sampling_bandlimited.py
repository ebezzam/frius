import numpy as np
import plot_settings
import os
import matplotlib.pyplot as plt

# fix random number generator
seed = 0
rng = np.random.RandomState(seed)

# create synthetic spectrum
n_vals = 1000
spectrum = rng.rand(n_vals)
freqs = np.linspace(-1e4, 1e4, n_vals)

plt.figure()
plt.plot(freqs, 20*np.log10(abs(spectrum)))
plt.grid()
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel("Frequencies [Hz]")

ax = plt.gca()
ax.axes.yaxis.set_ticklabels([])
plt.tight_layout()

fp = os.path.join(os.path.dirname(__file__), "figures", "_fig2p2a.pdf")
plt.savefig(fp, dpi=300)

"""
Ideal low pass filter
"""
cutoff = 2400
lpf = np.ones(n_vals)
lpf[freqs<-cutoff] = 0
lpf[freqs>cutoff] = 0

plt.figure()
plt.plot(freqs, lpf)
plt.grid()
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel("Frequencies [Hz]")

ax = plt.gca()
ax.axes.yaxis.set_ticklabels([])
plt.tight_layout()

fp = os.path.join(os.path.dirname(__file__), "figures", "_fig2p2b.pdf")
plt.savefig(fp, dpi=300)

plt.show()
