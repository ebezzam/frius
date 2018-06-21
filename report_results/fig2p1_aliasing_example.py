import numpy as np
from scipy.signal import resample
from scipy.io import wavfile
import plot_settings
import matplotlib.pyplot as plt


input_wav = "speech_48k.wav"
duration = 0.5
ultrasound_freq = 30e3
ultrasound_alias = 48e3-ultrasound_freq

samp_freq, signal = wavfile.read(input_wav)
signal = signal[:int(duration*samp_freq)]

freq_resp = 20*np.log10(np.abs(np.fft.fft(signal)))
f_vals = np.fft.fftfreq(len(signal), d=1/samp_freq)

# keep only positive
freq_resp = freq_resp[f_vals>=0]
f_vals = f_vals[f_vals>=0]

"""
Non-aliasing, higher sampling frequency
"""
noise_floor = freq_resp[np.argmin(abs(f_vals-samp_freq//2))]

# synthesize rest of signal
f_vals_hi = np.arange(0, 96000//2, f_vals[1])
freq_resp_hi = np.ones(len(f_vals_hi))*noise_floor
freq_resp_hi[:len(freq_resp)] = freq_resp

f_lo = np.argmin(abs(f_vals_hi-16000))
f_hi = np.argmin(abs(f_vals_hi-20000))
avg_hi_freq = np.mean(freq_resp_hi[f_lo:f_hi])
std_hi_freq = np.std(freq_resp_hi[f_lo:f_hi])
freq_resp_hi[len(freq_resp):] = avg_hi_freq + \
    std_hi_freq*np.random.randn(len(freq_resp_hi[len(freq_resp):]))

f_ultrasound_idx = np.argmin(abs(f_vals_hi-ultrasound_freq))
freq_resp_hi[f_ultrasound_idx] = max(freq_resp_hi) + 6  # plus 6 dB

plt.figure()
plt.plot(f_vals_hi, freq_resp_hi)
plt.grid()
plt.ylabel("dB")
plt.xlabel("Frequency [kHz]")
plt.xticks([8000, 16000, 24000, 32000, 40000, 48000], 
    ('8', '16', '24', '32', '40', '48'))
plt.tight_layout()
plt.xlim([0, max(f_vals_hi)])

plt.savefig("no_aliasing_ultrasound.pdf", dpi=1000)

"""
Aliasing
"""
f_alias_idx = np.argmin(abs(f_vals-ultrasound_alias))
freq_resp[f_alias_idx] = max(freq_resp)

plt.figure()
plt.plot(f_vals, freq_resp)
plt.grid()
plt.ylabel("dB")
plt.xlabel("Frequency [kHz]")
plt.xticks([8000, 16000, 24000], 
    ('8', '16', '24'))
plt.tight_layout()
plt.xlim([0, 48000/2])

plt.savefig("aliasing_ultrasound.pdf", dpi=1000)

plt.show()


