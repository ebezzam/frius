import numpy as np
import plot_settings
import matplotlib.pyplot as plt
import os

f_hi = 1000
f_lo = 100
ear_dist = 0.215  # 21.5 cm
speed_sound = 343
max_time = ear_dist/speed_sound
time = np.linspace(0, 1.6*max_time, num=1000)

left_ear = 0.05*max_time
right_ear = left_ear+max_time

s_hi = np.sin(2*np.pi*f_hi*time)
s_lo = np.sin(2*np.pi*f_lo*time)

# find intersect for both ears
s_hi_start = np.sin(2*np.pi*f_hi*left_ear)
s_hi_stop = np.sin(2*np.pi*f_hi*right_ear)
s_lo_start = np.sin(2*np.pi*f_lo*left_ear)
s_lo_stop = np.sin(2*np.pi*f_lo*right_ear)

plt.figure(figsize=(10, 4))

plt.plot(time, s_hi, label="%d Hz" % f_hi, ls='-')
plt.plot(time, s_lo, label="%d Hz" % f_lo, ls='--')

plt.xlabel("Time [seconds]")
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(loc='center right')
plt.xticks([left_ear, right_ear], 
    ('Left', 'Right'))
plt.yticks([s_hi_start, s_hi_stop, s_lo_start, s_lo_stop], 
    ('High', 'High', 'Low', 'Low'))
ax = plt.gca()
ax.xaxis.tick_top()
plt.grid()

fp = os.path.join(os.path.dirname(__file__), "figures", "_fig1p1.pdf")
plt.savefig(fp, dpi=300)


plt.show()
