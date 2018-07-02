import numpy as np
import plot_settings
import matplotlib.pyplot as plt


def gausspuls_coeff(fc, bandwidth, bwr=-6.):
    ref = pow(10.0, bwr / 20.0)
    return -(np.pi * fc * bandwidth) ** 2 / (4.0 * np.log(ref))

def gausspulse(t_vals, fc, bandwidth, bwr=-6.):
    a = gausspuls_coeff(fc, bandwidth, bwr)
    return np.exp(-a * t_vals**2) * np.cos(2*np.pi*fc*t_vals)

def gauss_ft(f_vals, a, fc=0):
    pos_half = np.sqrt(np.pi / a) * np.exp(-1.*(np.pi*(f_vals-fc))**2 / a)
    neg_half = np.sqrt(np.pi / a) * np.exp(-1.*(np.pi*(f_vals+fc))**2 / a)
    return (pos_half + neg_half)/2

# time domain plot
fc = 5e6
bandwidth = 2/3
bwr = -6
t_vals = np.linspace(-3/fc, 3/fc, 200)
h = gausspulse(t_vals, fc, bandwidth, bwr)

plt.figure()
plt.plot(t_vals, h)
plt.xlim([-6e-7, 6e-7])
plt.grid()

plt.xlabel("Time [seconds]")

ax = plt.gca()
ax.axes.yaxis.set_ticklabels([])
# ax.axes.xaxis.set_ticklabels([])

plt.tight_layout()
plt.savefig("_fig1p6a.pdf", dpi=300)

# frequency domain pulse
f_vals = np.linspace(-3*fc-1e3, 3*fc+1e3, 1000)
a = gausspuls_coeff(fc, bandwidth, bwr)
H = gauss_ft(f_vals, a, fc=fc)
H = H / max(H)

plt.figure()
plt.semilogx(f_vals, 20*np.log10(np.abs(H)))
plt.axvline(x=fc, c='r', label="$f_c$")
plt.grid()
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylabel("[dB]")
plt.legend(loc=3)
plt.xlabel("Frequency [Hz]")
plt.ylim([-40,0])

plt.tight_layout()
plt.savefig("_fig1p6b.pdf", dpi=300)


plt.show()