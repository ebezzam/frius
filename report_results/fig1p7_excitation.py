import numpy as np
from scipy.signal import square
import plot_settings
import matplotlib.pyplot as plt


def create_square_excitation(n_cycles, center_freq, samp_freq):

    n_rects = int(n_cycles/0.5)
    if abs(n_rects*0.5-n_cycles) > 0.01:
        print("Removing extra %f of cycle..." % abs(n_rects*0.5-n_cycles))

    n_cycles = n_rects*0.5
    n_neg = n_rects//2
    n_pos = n_rects-n_neg

    t_stop = int(n_cycles / center_freq * samp_freq) / samp_freq
    n_samp = int(n_cycles / center_freq * samp_freq) + 1 
    t_excite = np.linspace(0, t_stop, n_samp)
    excitation = square(2 * np.pi * center_freq * t_excite)

    return excitation, t_excite

def square_excitation_ft(f_vals, n_cycles, center_freq, centered=True):

    n_rects = int(n_cycles/0.5)
    if abs(n_rects*0.5-n_cycles) > 0.01:
        print("Removing extra %f of cycle..." % abs(n_rects*0.5-n_cycles))

    n_cycles = n_rects*0.5
    n_neg = n_rects//2
    n_pos = n_rects-n_neg

    if centered:
        duration = n_cycles * (1/center_freq)
        t_off = duration/2
    else:
        t_off = 0

    pos_rects = np.zeros(len(f_vals), dtype=np.complex)
    neg_rects = np.zeros(len(f_vals), dtype=np.complex)

    for k in range(n_pos):
        delay = 1/(4*center_freq) + k/center_freq - t_off
        pos_rects += np.exp(-1j*2*np.pi*f_vals*delay)
    for m in range(n_neg):
        delay = 3/(4*center_freq) + m/center_freq - t_off
        neg_rects += np.exp(-1j*2*np.pi*f_vals*delay)

    # rect_ft = np.sinc(f_vals/2/center_freq) / (2*center_freq)
    rect_ft = np.sinc(f_vals/2/center_freq)

    return rect_ft * (pos_rects - neg_rects)


# time domain plot
fc = 5e6
n_cycles = 2.5
samp_freq = fc*100

excitation, t_excite = create_square_excitation(n_cycles, fc, samp_freq)

plt.figure()
plt.plot(t_excite, excitation)
plt.grid()
plt.xlabel("Time [seconds]")
plt.tight_layout()
plt.savefig("excitation.pdf", dpi=1000)


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
plt.savefig("excitation_ft.pdf", dpi=1000)

plt.show()