import numpy as np
import time, warnings, os, datetime

# plotting settings
import plot_settings
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..",))
from frius import distance2time, total_freq_response, time2distance, sample_rf
from frius import gen_fri, estimate_time_param, estimate_amplitudes, compute_srr_db


"""
User parameters
"""

# load previously computed results
results_dir = 'nde_general_06_27_16h25'

# test parameters if no results file
n_points = 7
min_depth = 0.01575
max_depth = 0.075
oversample_fact = 9.5
seed = 0
max_ini = 5
stop_cri = 'max_iter'

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
n_elem_tx = 64
probe_geometry = np.arange(n_elem_tx)*dx

# load results if available, otherwise run algo
try:
    npzfile = np.load(os.path.join(os.path.dirname(__file__), results_dir, "results.npz"))
    rx_echoes = npzfile['rx_echoes']
    amplitudes = npzfile['amplitudes']
    resynth_score = npzfile['resynth_score']
    t0 = npzfile['t0']
    tN = npzfile['tN']
    print("Loading data from %s..." % results_dir)
    run_sweep = False
except:
    run_sweep = True
    print("No data available. Loading NDE data and recovering for each channel...")
    print()

if run_sweep:

    # signed 16 bit integer [-128,127]
    ndt_rawdata = np.genfromtxt(os.path.join(os.path.dirname(__file__), '..', 'data', 'ndt_rawdata.csv'), delimiter=',')
    n_samples = len(ndt_rawdata)
    time_vec = np.arange(n_samples)/samp_freq
    depth = time_vec[-1]/2*speed_sound

    """ Select portion of recording """
    min_samp = int(distance2time(min_depth, speed_sound) * samp_freq)
    t0 = time_vec[min_samp]
    max_samp = int(distance2time(max_depth, speed_sound) * samp_freq)
    tN = time_vec[max_samp]
    t_samp = time_vec[min_samp:max_samp] - t0
    n_samples = len(t_samp)

    """ From FRI parameters, determine necessary sampling """
    period = t_samp[-1]-t_samp[0] + 1/samp_freq
    fs_ind_center = int(np.ceil(center_freq*period))
    period = fs_ind_center/center_freq  # adjust period

    K = n_points
    M = K * oversample_fact
    n_samples_fri = 2*M+1   # also number of FS coefficients

    # subsampling
    skip = n_samples//n_samples_fri
    sub_idx =  np.arange(0, n_samples, skip).astype(np.int)
    n_samples_fri = len(sub_idx)
    M = (n_samples_fri-1)//2
    oversample_fact = M/K
    print("Oversample factor : %f" % oversample_fact)
    t_samp_sub = t_samp[sub_idx]

    # fourier coefficients - ideal low pass by only using these coefficients
    fs_ind_base = np.arange(-M, M+1)
    fs_ind = fs_ind_base + fs_ind_center    # around center frequency

    """ # build linear transformation """
    # pulse estimate
    H_tot = total_freq_response(fs_ind/period, center_freq, bw_pulse, n_cycles)
    t_samp_sub = t_samp[sub_idx]
    fs_ind_grid, t_samp_grid = np.meshgrid(fs_ind, t_samp_sub)
    idft_trunc = np.exp(2j*np.pi*fs_ind_grid*t_samp_grid/period)
    G = idft_trunc*H_tot

    # "TGC" of sorts
    atten = 2*np.pi*time2distance(t_samp_sub+t0, speed_sound)

    """ Sweep though all channels """
    rx_echoes = np.zeros((n_elem_tx, n_points))
    amplitudes = np.zeros((n_elem_tx, n_points))
    resynth_score = np.zeros(n_elem_tx)
    start_time = time.time()
    print()
    for chan_idx in range(n_elem_tx):

        print("CHANNEL %d/%d" % (chan_idx+1, n_elem_tx))

        y_samp = ndt_rawdata[min_samp:max_samp, chan_idx]

        # basic clean up
        y_samp = y_samp - np.mean(y_samp) # remove DC component

        # bandpass (essentially IQ demod) and sample (sub-sample original)
        y_samp_demod = np.exp(-1j*2*np.pi*center_freq*t_samp) * y_samp
        Y_shift = np.fft.fft(y_samp_demod)
        Y_lpf = np.zeros(Y_shift.shape, dtype=np.complex)
        Y_lpf[fs_ind_base] = Y_shift[fs_ind_base]
        y_samp_lpf = np.fft.ifft(Y_lpf)
        y_samp_sub = y_samp_lpf[sub_idx] 

        # TGC
        y_samp_sub = y_samp_sub * atten

        # recovery
        warnings.filterwarnings("ignore")
        fs_coeff_hat, min_error, c_opt, ini = gen_fri(G, y_samp_sub, n_points, max_ini=max_ini, stop_cri=stop_cri, seed=seed)
        warnings.filterwarnings("default")

        tk_hat_gen = estimate_time_param(c_opt, period)
        ck_hat_gen = estimate_amplitudes(fs_coeff_hat, fs_ind/period, tk_hat_gen, period)

        # evaluate
        y_rf = sample_rf(ck_hat_gen, tk_hat_gen, period, samp_freq, center_freq, bw_pulse, n_cycles)[0]
        srr = compute_srr_db(y_samp, y_rf)
        print("SRR : %f dB" % srr)

        rx_echoes[chan_idx] = tk_hat_gen
        amplitudes[chan_idx] = ck_hat_gen


    tot_time = time.time() - start_time
    print("TOTAL TIME : %f min" % (tot_time/60))

    """ save data """
    time_stamp = datetime.datetime.now().strftime("%m_%d_%Hh%M")
    results_dir = os.path.join(os.path.dirname(__file__), "nde_general_%s" % (time_stamp))
    os.makedirs(results_dir)
    np.savez(os.path.join(results_dir, "results"), rx_echoes=rx_echoes, 
        amplitudes=amplitudes, resynth_score=resynth_score, t0=t0, tN=tN,
        n_points=n_points, min_depth=min_depth, max_depth=max_depth, 
        oversample_fact=oversample_fact, seed=seed, max_ini=max_ini,
        stop_cri=stop_cri)

    print("Results saved to %s" % results_dir)

"""
Plot 
"""
scal_fact = 1e5

plt.figure()
plt.scatter(np.arange(n_elem_tx)+1, np.zeros(n_elem_tx), c='r', marker= 's', label="Transducers")
for k in range(n_points):
    plt.scatter(np.arange(n_elem_tx)+1, scal_fact*(rx_echoes[:,k]+t0))
plt.grid()
plt.ylim([0, scal_fact*distance2time(max_depth, speed_sound)])
plt.ylabel("Time [%ss]" % '1e-5')
plt.xlabel("Array element")
plt.ylim([0, scal_fact*tN])
ax = plt.gca()
ax.invert_yaxis()
plt.tight_layout()
fp = os.path.join(os.path.dirname(__file__), "figures", "_fig4p5b.pdf")
plt.savefig(fp, dpi=300)

plt.show()
