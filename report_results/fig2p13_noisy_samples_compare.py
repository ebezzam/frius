import numpy as np
import os

import plot_settings
import matplotlib.pyplot as plt
ALPHA = 0.7
MARKER_SIZE = 10

# load data
results_dir = 'noisy_samples_10_06_14_14h40'
results_dir_gen = 'noisy_samples_gen_10_06_14_19h25'

results_dir_20 = 'noisy_samples_20_06_14_15h23'
results_dir_gen_20 = 'noisy_samples_gen_20_06_14_21h37'


# load 10 pulse data
npzfile = np.load(os.path.join(results_dir, "results.npz"))
npzfile_gen = np.load(os.path.join(results_dir_gen, "results.npz"))

n_diracs = npzfile['n_diracs']
snr_vals = npzfile['snr_vals']
oversampling_vals = npzfile['oversampling_vals']
sig_err = np.mean(npzfile['sig_err'][0], axis=1)
tk_err = np.mean(npzfile['tk_err'][0], axis=1)

sig_err_gen = np.mean(npzfile_gen['sig_err'][0], axis=1)
tk_err_gen = np.mean(npzfile_gen['tk_err'][0], axis=1)


# load 20 pulse data
npzfile = np.load(os.path.join(results_dir_20, "results.npz"))
npzfile_gen = np.load(os.path.join(results_dir_gen_20, "results.npz"))

n_diracs_20 = npzfile['n_diracs']
snr_vals_20 = npzfile['snr_vals']
oversampling_vals_20 = npzfile['oversampling_vals']
sig_err_20 = np.mean(npzfile['sig_err'][0], axis=1)
tk_err_20 = np.mean(npzfile['tk_err'][0], axis=1)

sig_err_gen_20 = np.mean(npzfile_gen['sig_err'][0], axis=1)
tk_err_gen_20 = np.mean(npzfile_gen['tk_err'][0], axis=1)


# plot
plt.figure()
plt.plot(snr_vals, tk_err, 'b--', marker='^', markersize=MARKER_SIZE,
        alpha=ALPHA, label="standard, %d puls"%(n_diracs))
plt.plot(snr_vals_20, tk_err_gen, 'b-', marker='o', markersize=MARKER_SIZE,
        alpha=ALPHA, label="general, %d puls"%(n_diracs))
plt.plot(snr_vals, tk_err_20, 'g--', marker='^', markersize=MARKER_SIZE,
        alpha=ALPHA, label="standard, %d puls"%(n_diracs_20))
plt.plot(snr_vals_20, tk_err_gen_20, 'g-', marker='o', markersize=MARKER_SIZE,
        alpha=ALPHA, label="general, %d puls"%(n_diracs_20))
plt.xlabel("SNR [dB]")
plt.ylabel("$ SRR_{dev}$ [dB]")
plt.ylim([0,90])
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("_fig2p13a.pdf", format='pdf', dpi=300)

plt.figure()
plt.plot(snr_vals, sig_err, 'b--', marker='^', markersize=MARKER_SIZE,
        alpha=ALPHA, label="standard, %d puls"%(n_diracs))
plt.plot(snr_vals_20, sig_err_gen, 'b-', marker='o', markersize=MARKER_SIZE,
        alpha=ALPHA, label="general, %d puls"%(n_diracs))
plt.plot(snr_vals, sig_err_20, 'g--', marker='^', markersize=MARKER_SIZE,
        alpha=ALPHA, label="standard, %d puls"%(n_diracs_20))
plt.plot(snr_vals_20, sig_err_gen_20, 'g-', marker='o', markersize=MARKER_SIZE,
        alpha=ALPHA, label="general, %d puls"%(n_diracs_20))
plt.xlabel("SNR [dB]")
plt.ylabel("$ SRR$ [dB]")
plt.ylim([-5,50])
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("_fig2p13b.pdf", format='pdf', dpi=300)

plt.show()