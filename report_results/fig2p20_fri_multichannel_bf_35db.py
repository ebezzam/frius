import numpy as np
import h5py, os, time

import plot_settings
import matplotlib.pyplot as plt
ALPHA = 0.7

import sys
sys.path.append('..')
from frius import das_beamform, image_bf_data, gen_echoes, time2distance, \
    distance2time, sample_rf
from frius import recover_parameters, compute_srr_db_points, compute_srr_db, demodulate

"""
Example of multichannel FRI (separate problem for each channel) and then standard
DAS beamforming without apodization
"""

K = 30
oversample_freq = 3
min_depth = 0.01
max_depth = 0.055   # put a bit more than desired, because we "chop" of a bit of end when demodulating since we can't sample at exactly 2M/T (sample at larger rate)
display_depth = 0.05

""" Load data """
raw_file = 'raw_data_20points_SNR35dB_1pw_1p0cycles_1-05_09h11.h5'
n_points = 20
f = h5py.File(os.path.join('..', 'data', raw_file), 'r')
plot_original = True

# extract all fields
rf_data = np.squeeze(np.array(f['rf_data']))
time_vec = np.squeeze(np.array(f['time_vec']))
true_positions = np.squeeze(np.array(f['true_positions']))
center_freq = float(np.squeeze(np.array(f['center_freq'])))
samp_freq = float(np.squeeze(np.array(f['samp_freq'])))
samp_time = 1./samp_freq
speed_sound = float(np.squeeze(np.array(f['speed_sound'])))
bandwidth = float(np.squeeze(np.array(f['bandwidth'])))
num_cycles = float(np.squeeze(np.array(f['num_cycles'])))
pitch = float(np.squeeze(np.array(f['pitch'])))
n_elements = int(np.squeeze(np.array(f['n_elements'])))

period = time_vec[-1]+samp_time
probe_geometry = pitch*np.arange(n_elements)
probe_geometry -= np.mean(probe_geometry)

# extract locations of strong reflectors and corresponding echo locations across channels
x_true = true_positions[0,:][-n_points:]
z_true = true_positions[2,:][-n_points:]
true_tofs = gen_echoes(x_true, z_true, probe_geometry, speed_sound)
print("%d strong reflectors" % n_points)
print("%d total reflectors" % true_positions.shape[1])


""" Crop and beamform """ 

# prepare data
min_time = distance2time(min_depth, speed_sound)
max_time = distance2time(max_depth, speed_sound)
first_sample = int(np.floor(min_time/samp_time))
last_sample = int(np.ceil(max_time/samp_time))
t0 = samp_time*first_sample
rf_data_trunc = rf_data[:,first_sample:last_sample]
time_vec_trunc = time_vec[first_sample:last_sample]-t0
duration = time_vec_trunc[-1] - time_vec_trunc[0] + samp_time
n_samples = rf_data_trunc.shape[1]

""" FRI compression and resynthesis """
channel_idx = n_elements//2
ck_hat, tk_hat, period = recover_parameters(
    rf_data_trunc[channel_idx,:], time_vec_trunc, K, oversample_freq,
    center_freq, bandwidth,
    num_cycles)
print("Locations SRR [dB] : %f " % compute_srr_db_points(
    true_tofs[channel_idx,:], tk_hat+t0))

# resynthesize
y_resynth = sample_rf(ck_hat, tk_hat, duration,
    samp_freq, center_freq, bandwidth, num_cycles)[0]
print("Resynthesized SRR [dB] : %f " % compute_srr_db(
    rf_data_trunc[channel_idx,:]/max(rf_data_trunc[channel_idx,:]), 
    y_resynth/max(y_resynth)))

plt.figure()
plt.plot(time_vec_trunc+t0, 
    rf_data_trunc[channel_idx,:]/max(rf_data_trunc[channel_idx,:]), alpha=ALPHA, 
    label="Original")
plt.plot(time_vec_trunc+t0, y_resynth/max(y_resynth), alpha=ALPHA, 
    label="Resynthesized")
plt.xlim([min_time,max_time])
plt.grid()
plt.xlabel("Time [s]")
plt.legend()
plt.tight_layout()
ax = plt.gca()
ax.axes.yaxis.set_ticklabels([])
plt.savefig("_fig2p20b.pdf", format='png', dpi=300)


""" Sweep through all channels """
location_score = np.zeros(n_elements)
resynth_data = np.zeros(rf_data_trunc.shape)
resynth_score = np.zeros(n_elements)
resynth_score_norm = np.zeros(n_elements)

tk_hat_multi = np.zeros((n_elements, K))
ck_hat_multi = np.zeros((n_elements, K))

start_time = time.time()
for channel_idx in range(n_elements):

    print("%d/%d" % (channel_idx+1, n_elements), end=" ")
    
    # recover parameters
    ck_hat_multi[channel_idx], tk_hat_multi[channel_idx], period = \
        recover_parameters(
            rf_data_trunc[channel_idx,:], time_vec_trunc,
            K=K, oversample_freq=oversample_freq,
            center_freq=center_freq, bandwidth=bandwidth, num_cycles=num_cycles)
    
    # remember to add offset!
    location_score[channel_idx] = compute_srr_db_points(
        true_tofs[channel_idx,:], tk_hat_multi[channel_idx]+t0)
    
    # resynthesize
    resynth_data[channel_idx] = sample_rf(ck_hat_multi[channel_idx], 
        tk_hat_multi[channel_idx], duration, samp_freq, 
        center_freq, bandwidth, num_cycles)[0]

    # resynth error
    resynth_score[channel_idx] = compute_srr_db(rf_data_trunc[channel_idx,:], 
        resynth_data[channel_idx])
    resynth_score_norm[channel_idx] = compute_srr_db(
        rf_data_trunc[channel_idx]/max(rf_data_trunc[channel_idx]), 
        resynth_data[channel_idx]/max(resynth_data[channel_idx]))
    
    print("(%f, %f)" % (location_score[channel_idx], 
        resynth_score_norm[channel_idx]))

tot_time = time.time() - start_time
print("FRI across all channels : %f min" % (tot_time/60.))


""" Beamform resynthesize signals """
scal_fact = 1e2
bf_fri = das_beamform(resynth_data, samp_freq, pitch, probe_geometry, 
    center_freq, speed_sound, depth=display_depth, t_offset=t0)[0]
image_bf_data(bf_fri, probe_geometry, max_depth=display_depth, 
    min_depth=time2distance(t0, speed_sound), scal_fact=scal_fact)
plt.ylabel('Axial [cm]')
plt.xlabel('Lateral [cm]')
plt.tight_layout()
plt.savefig("_fig2p20a.png", format='png', dpi=300)


fs_iq = center_freq*bandwidth
n_samples_fri = 2*K*oversample_freq+1
fs_fri = n_samples_fri/duration

print()
print("Num of samples (RF) : %d" % n_samples)
print("Num of samples (FRI) : %d" % n_samples_fri)
print("Sampling rate (FRI) : %f" % fs_fri)

print("Data rate reduction from RF : %f" % (n_samples/n_samples_fri))
print("Sampling rate reduction from IQ : %f" % (fs_iq/fs_fri))
print("Compression : %f" % (n_samples/(2*K)))

plt.show()