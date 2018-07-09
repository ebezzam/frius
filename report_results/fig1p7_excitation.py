import numpy as np
import plot_settings
import matplotlib.pyplot as plt
import os
from test_utilities import create_square_excitation, square_excitation_ft

# time domain plot
fc = 5e6
n_cycles = 2.5
samp_freq = fc*100

excitation, t_excite = create_square_excitation(n_cycles, fc, samp_freq)

plt.figure()
plt.plot(t_excite, excitation)
plt.grid()
plt.xlabel("Time [seconds]")
plt.xlim([0, 5e-7])
plt.tight_layout()

fp = os.path.join(os.path.dirname(__file__), "figures", "_fig1p7a.pdf")
plt.savefig(fp, dpi=300)


f_vals = np.fft.fftfreq(len(excitation), d=1/samp_freq)
H_ex = square_excitation_ft(f_vals, n_cycles, fc, centered=False)
pulse_an = np.fft.ifft(H_ex)
time_an = np.arange(len(pulse_an))/samp_freq


H_ex /= max(abs(H_ex))

plt.figure()
plt.semilogx(f_vals, 20*np.log10(np.abs(H_ex)))
plt.axvline(x=fc, c='r', label="$f_c$")
plt.grid()
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylabel("[dB]")
plt.legend(loc="upper right")
plt.xlabel("Frequency [Hz]")
plt.ylim([-40,0])
plt.tight_layout()

fp = os.path.join(os.path.dirname(__file__), "figures", "_fig1p7b.pdf")
plt.savefig(fp, dpi=300)

plt.show()
