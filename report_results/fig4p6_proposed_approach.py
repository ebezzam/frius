import numpy as np
import time, warnings, os

# plotting settings
import plot_settings
import matplotlib.pyplot as plt
LEGEND_FONTSIZE = 15

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..",))
from frius import distance2time, total_freq_response, time2distance, das_beamform, image_bf_data, estimate_2d_loc
from frius import gen_fri, estimate_time_param, cadzow_denoising, estimate_fourier_coeff, compute_ann_filt
from frius import tof_sort_nde, edm


"""
User parameters
"""

# load previously computed TOFs
results_dir = 'nde_general_06_27_16h25'
# results_dir = 'nde_standard_06_27_16h04'

# test parameters if no results file
n_points = 7
min_depth = 0.01575
max_depth = 0.075
oversample_fact = 9.5
verbose = True

denoising_method = "genfri"  # 'genfri' or 'cadzow'
seed = 0
max_ini = 5
stop_cri = 'max_iter'
cadzow_iter = 20

n_elem_rx = 8
width = 1.5   # in receive element pitch to prune out recovered locations from other references
tol = 1e-5  # tolerance for removing duplicates

# load results if available, otherwise run algo
try:
    npzfile = np.load(os.path.join(os.path.dirname(__file__), results_dir, "results.npz"))
    rx_echoes = npzfile['rx_echoes']
    amplitudes = npzfile['amplitudes']
    resynth_score = npzfile['resynth_score']
    t0 = npzfile['t0']
    tN = npzfile['tN']
    print("Loading TOFs from %s..." % results_dir)
    run_sweep = False
except:
    run_sweep = True
    print("No TOFs available. Recovering for each channel...")
    print()

"""
Raw data and probe parameters
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

# signed 16 bit integer [-128,127]
ndt_rawdata = np.genfromtxt(os.path.join(os.path.dirname(__file__), '..', 'data', 'ndt_rawdata.csv'), delimiter=',')
n_samples = len(ndt_rawdata)
time_vec = np.arange(n_samples)/samp_freq
depth = time_vec[-1]/2*speed_sound


if not run_sweep:

    sub_idx = np.linspace(0, n_elem_tx-1, n_elem_rx).astype(np.int)
    array_pos = probe_geometry[sub_idx]

    rx_echoes = rx_echoes[sub_idx] + t0

else:

    """
    Subsample array
    """
    sub_idx = np.linspace(0, n_elem_tx-1, n_elem_rx).astype(np.int)
    array_pos = probe_geometry[sub_idx]

    """
    Estimate TOFs with pulse stream recovery
    """
    # determine region to sample
    min_samp = int(distance2time(min_depth, speed_sound) * samp_freq)
    t0 = time_vec[min_samp]
    max_samp = int(distance2time(max_depth, speed_sound) * samp_freq)
    tN = time_vec[max_samp]
    t_samp = time_vec[min_samp:max_samp] - t0
    n_samples = len(t_samp)

    # FRI parameters
    period = t_samp[-1]-t_samp[0]
    fs_ind_center = int(np.ceil(center_freq*period))
    period = fs_ind_center/center_freq   
    K = n_points
    M = K * oversample_fact
    n_samples_fri = 2*M+1   # also number of FS coefficients

    # subsampling
    skip = n_samples//n_samples_fri
    sub_idx_time =  np.arange(0, n_samples, skip).astype(np.int)
    n_samples_fri = len(sub_idx_time)
    M = (n_samples_fri-1)//2
    oversample_fact = M/K
    print("Undersampling factor : %f" % (n_samples/n_samples_fri))
    print("Oversample factor : %f" % oversample_fact)

    t_samp_sub = t_samp[sub_idx_time]
    atten = 2*np.pi*time2distance(t_samp_sub+t0, speed_sound)

    # fourier coefficients to sample - ideal low pass by only using this coefficients
    fs_ind_base = np.arange(-M, M+1)
    fs_ind = fs_ind_base + fs_ind_center    # around center frequency

    # pulse fourier coefficients for forward mapping
    H_tot = total_freq_response(fs_ind/period, center_freq, bw_pulse, n_cycles)

    if denoising_method is 'genfri':
        # build linear transformation
        fs_ind_grid, t_samp_grid = np.meshgrid(fs_ind, t_samp_sub)
        idft_trunc = np.exp(2j*np.pi*fs_ind_grid*t_samp_grid/period)
        G = idft_trunc*H_tot
            
    rx_echoes = np.zeros((n_elem_rx, n_points))
    start_time = time.time()
    print()
    for k, chan_idx in enumerate(sub_idx):

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
        y_samp_sub = y_samp_lpf[sub_idx_time] 
        y_samp_sub = y_samp_sub * atten  # TGC in a way

        if denoising_method is 'genfri': 
            warnings.filterwarnings("ignore")
            fs_coeff_hat, min_error, ann_filt, ini = gen_fri(G, y_samp_sub, n_points, max_ini=max_ini, stop_cri=stop_cri, seed=seed)
            warnings.filterwarnings("default")
        else:
            fs_coeff_hat = estimate_fourier_coeff(y_samp_sub, t_samp_sub, H=H_tot)
            fs_coeff_hat_clean = cadzow_denoising(fs_coeff_hat, K, n_iter=cadzow_iter)
            ann_filt = compute_ann_filt(fs_coeff_hat_clean, K)

        # solve for time instances
        tk_hat = estimate_time_param(ann_filt, period)

        # save
        rx_echoes[k] = tk_hat + t0

    proc_time = time.time() - start_time
    print("FRI TIMING : %f seconds per channel" % (proc_time/n_elem_rx))
    print("TOTAL TIME FRI : %f minutes" % (proc_time/60.))


"""
Sort TOFs with EDM/Gram test
"""
array_pos_2d = np.vstack((array_pos, np.zeros(n_elem_rx)))
sorted_tof_est, sorted_tof_chan, sorted_tof_scores = tof_sort_nde(rx_echoes, array_pos_2d, speed_sound, verbose=verbose)


"""
Solve intersection
"""
x_est = dict()
z_est = dict()
for ref in range(n_elem_rx):
    n_recov_points = len(sorted_tof_est[ref])
    x_est[ref] = np.zeros(n_recov_points)
    z_est[ref] = np.zeros(n_recov_points)
    for idx, point_idx in enumerate(sorted_tof_chan[ref].keys()):
        x_est[ref][idx], z_est[ref][idx] = estimate_2d_loc(
            array_pos[sorted_tof_chan[ref][point_idx]], 
            sorted_tof_est[ref][point_idx], 
            speed_sound, 
            avg=True)


"""
For each reference, only take points within its "column", assuming uniform array
"""
print()
print("Heuristics to remove duplicates...")
dx_rx = np.mean(array_pos[1:]-array_pos[:-1])
x_est_prun = dict()
z_est_prun = dict()
scores_prun = dict()
for ref in range(n_elem_rx):
    dx_right = min(array_pos[ref]+dx_rx*width, array_pos[-1])
    dx_left = max(array_pos[ref]-dx_rx*width, array_pos[0])

    if verbose:
        print("Reference channel %d column : " % ref, end="")
        print("[%f,%f)" % (dx_left, dx_right))

    mask = ((x_est[ref] < dx_right) & (x_est[ref] >= dx_left))
    
    x_est_prun[ref] = x_est[ref][mask]
    z_est_prun[ref] = z_est[ref][mask]
    scores_ref = list(sorted_tof_scores[ref].values())
    scores_prun[ref] = [scores_ref[idx] for idx, val in enumerate(mask) if val]

x_est_all = []
z_est_all = []
scores_all = []
for ref in range(n_elem_rx):
    scores_ref = scores_prun[ref]
    for k in range(len(x_est_prun[ref])):
        x_est_all.append(x_est_prun[ref][k])
        z_est_all.append(z_est_prun[ref][k])
        scores_all.append(scores_ref[k])
order = np.argsort(scores_all).astype(int)
x_est_all = [x_est_all[k] for k in order]
z_est_all = [z_est_all[k] for k in order]
scores_all = [scores_all[k] for k in order]


est_loc = np.concatenate((np.array(x_est_all)[:,np.newaxis], 
    np.array(z_est_all)[:,np.newaxis]),axis=1)
D = edm(est_loc.T)

prun_loc = []
for k in np.arange(len(D)):
    mask = (D[k] < tol)
    # print(np.where(mask==True)[0][0])
    prun_loc.append(np.where(mask==True)[0][0])
prun_loc = list(set(prun_loc))
x_est_prun_uni = np.array([x_est_all[k] for k in prun_loc])
z_est_prun_uni = np.array([z_est_all[k] for k in prun_loc])
scores_prun_uni = np.array([scores_all[k] for k in prun_loc])


"""
DAS beamform and overlay localization results
"""
res = das_beamform(ndt_rawdata.T, samp_freq, 
    dx, probe_geometry, center_freq, speed_sound, depth)[0]

scal_fact = 1e2
image_bf_data(res, probe_geometry, depth, dynamic_range=40, scal_fact=scal_fact)
plt.scatter(array_pos*scal_fact, np.zeros(len(array_pos)), c='r', marker='s', label="RX elem.", s=70)
plt.scatter(x_est_prun_uni*scal_fact, z_est_prun_uni*scal_fact, label="Estimate", s=100, facecolors='none', edgecolors='r')
plt.xlabel("Lateral [cm]")
plt.ylabel("Axial [cm]")
plt.ylim([max_depth*scal_fact, 0])
plt.legend(fontsize=LEGEND_FONTSIZE, loc=3)
plt.tight_layout()
fp = os.path.join(os.path.dirname(__file__), "figures", "_fig4p6.png")
plt.savefig(fp, dpi=300)

plt.show()
