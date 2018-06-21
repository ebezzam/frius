import numpy as np
from scipy.signal import gausspulse, square
from scipy.interpolate import InterpolatedUnivariateSpline, interp2d
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from itertools import combinations


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx


def time2distance(t, speed_sound):

    return t*speed_sound/2


def distance2time(d, speed_sound):

    return d*2/speed_sound


def das_beamform(recordings, samp_freq, pitch, array_pos, center_freq, 
    speed_sound, depth=None, t_offset=0, dx_ratio=1/3, dz_ratio=1/4, 
    interp='cubic', angle=0):
    """
    recordings must be of shape [n_channels, n_samples]
    """

    if interp == 'cubic':
        k = 3
    elif interp == 'linear':
        k = 1
    else:
        raise ValueError("Only `linear` or `cubic` interpolation is supported.")

    dtype = recordings.dtype

    element_width = 0

    n_channels, n_samples = recordings.shape
    time_samples = np.arange(n_samples) / samp_freq + t_offset
    if depth is None:
        depth = time_samples[-1]*speed_sound/2

    wavelength = speed_sound / center_freq
    lat_pitch_ratio = pitch / (dx_ratio * wavelength)

    lateral_image_positions = np.linspace(start=min(array_pos), stop=max(array_pos),
                                          num=int(np.ceil(lat_pitch_ratio * n_channels)))

    t_round_min = time_samples[0]
    t_round_max = 2 * depth / speed_sound

    ax_sample_ratio = speed_sound / wavelength / dz_ratio / samp_freq
    ax_time_samples = np.linspace(start=t_round_min, stop=t_round_max,
                                   num=int(np.ceil(ax_sample_ratio * n_samples)))

    beamformed_data = np.zeros([np.size(lateral_image_positions, axis=0), 
        np.size(ax_time_samples, axis=0)], dtype=dtype)

    # Pre-computations
    time_mesh_array, array_mesh = np.meshgrid(ax_time_samples, array_pos)
    time_mesh_array = time_mesh_array.astype(dtype=dtype)
    array_mesh = array_mesh.astype(dtype=dtype)
    
    tx_delay = 0.5 * time_mesh_array   # normal incidence

    # interpolation functions for each channel
    f = []
    for d in recordings:
        f_d = InterpolatedUnivariateSpline(time_samples, d, k=k, ext='zeros')
        f.append(f_d)

    # beamform along each image line
    for it_x, pos in enumerate(lateral_image_positions):

        # receive delay
        tx_return = (pos - array_mesh) / speed_sound
        tz_return = tx_delay
        rx_delay = np.sqrt(tx_return*tx_return + tz_return*tz_return)

        # Round trip time of flight
        txrx_delay = tx_delay + rx_delay

        # delay-and-sum along this line
        for it_chan, t in enumerate(txrx_delay):
            beamformed_data[it_x,:] += f[it_chan](t)

    return beamformed_data, ax_time_samples, lateral_image_positions


def image_bf_data(beamformed_data, array_pos, max_depth, dynamic_range=60, 
    min_depth=0, scal_fact=1, figsize=None):

    # extract envelope
    beamformed_data_env = abs(hilbert(beamformed_data))
    im = 20*np.log10(beamformed_data_env/np.max(beamformed_data_env))

    # image dimensions, allow to scale image
    nx = beamformed_data_env.shape[0]
    nz = beamformed_data_env.shape[1]

    # create image
    x = np.linspace(min(array_pos), max(array_pos), nx)*scal_fact
    z = np.linspace(min_depth, max_depth, nz)*scal_fact
    X, Z = np.meshgrid(x, z)

    if figsize is None:
        plt.figure()
    else:
        plt.figure(figsize=figsize)

    fig = plt.pcolormesh(X, Z, im.T)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axis([min(x), max(x), min(z), max(z)])
    fig.set_cmap('gray')
    fig.set_clim(vmin=-dynamic_range, vmax=0)
    cbar = plt.colorbar()
    cbar.ax.set_xlabel('dB', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()

    ax = plt.gca()
    ax.invert_yaxis()

    return im, x, z







def sample_rf(ck, tk, duration, samp_freq, 
    center_freq, bandwidth, num_cycles, bwr=-6):
    """
    Given pulse stream locations, (ideally) sample RF within bandwidth of
    [-samp_freq/2, samp_freq/2].
    """

    Ts = 1/samp_freq
    t_samp = np.arange(start=0, stop=duration, step=Ts)
    n_samples = len(t_samp)

    # frequencies within bandwidth
    freqs = np.fft.fftfreq(n_samples, Ts)

    # pulse / system response
    H_tot = total_freq_response(freqs, center_freq, bandwidth, num_cycles, bwr)

    # ground truth FS coefficients
    tk_grid, freqs_grid = np.meshgrid(tk, freqs)
    fs_coeff = np.dot(np.exp(-1j*2*np.pi*freqs_grid*tk_grid), ck) / duration
    fs_coeff *= H_tot

    # compute samples, around baseband so real
    y_samp = np.real(np.fft.ifft(fs_coeff) * n_samples)

    return y_samp, t_samp


def sample_iq(ck, tk, duration, samp_freq, 
    center_freq, bandwidth, num_cycles, bwr=-6):
    """
    Given pulse stream locations, (ideally) sample IQ within bandwidth of
    ``[center_freq-samp_freq/2, center_freq+samp_freq/2]``.

    This can be physically realized by multiplying the input signal with 
    ``cos(2 pi fc t)`` and ``sin(2 pi fc t)``, where the cosine "channel"
    would contain the in-phase / real component and the sinus "channel" would
    contain the quadrature / imaginary component.

    This function returns sampled after IQ demodulation, i.e. complex-valued 
    samples.
    """

    Ts = 1/samp_freq
    t_samp = np.arange(start=0, stop=duration, step=Ts)
    n_samples = len(t_samp)

    # frequencies within bandwidth
    freqs_base = np.fft.fftfreq(n_samples, Ts)
    freqs = freqs_base+center_freq

    # pulse / system response
    H_tot = total_freq_response(freqs, center_freq, bandwidth, num_cycles, bwr)

    # ground truth FS coefficients
    tk_grid, freqs_grid = np.meshgrid(tk, freqs)
    fs_coeff = np.dot(np.exp(-1j*2*np.pi*freqs_grid*tk_grid), ck) / duration
    fs_coeff *= H_tot

    # compute samples with IDFT around baseband as spectrum around `center_freq`
    # has been shifted down to baseband
    y_samp = np.fft.ifft(fs_coeff) * n_samples

    return y_samp, t_samp


def add_noise(sig, snr_db, seed=0):
    """
    Add random noise from the standard normal distribution according to the 
    given SNR level in dB.

    :param sig: Signal to add noise to.
    :param snr_db: SNR in dB; add noise to get this level of SNR.
    """

    n_samples = len(sig)
    rng = np.random.RandomState(seed)

    if np.sum(np.iscomplex(sig)): # check if any sample complex
        
        noise = rng.randn(n_samples) + 1j * rng.randn(n_samples)
        noise_std = np.linalg.norm(sig) / (10**(snr_db/20.))
        noise = noise/np.linalg.norm(noise)*noise_std

    else:
        noise = rng.randn(n_samples)
        noise_std = np.linalg.norm(sig) / (10**(snr_db/20.))
        noise = noise/np.linalg.norm(noise)*noise_std

    return noise


def total_freq_response(freqs, center_freq, bandwidth, num_cycles, bwr=-6,
    return_all=False):
    """
    square excitation * gaus puls * gaus puls
    """

    # bandwidth increases when convolving gaussian pulse with another
    bandwidth_2 = np.sqrt(2)*bandwidth/2

    # compute alpha according to "Time domain compressive beam forming of ultrasound signals" [Guillaume David]
    alpha_2 = gausspuls_coeff(center_freq, bandwidth_2, bwr=-6.)

    # gaus * gaus
    H_gaus_2 = gausspulse_ft(freqs, alpha_2, center_freq)

    # square excitation
    H_excitation = square_excitation_ft(freqs, num_cycles, center_freq)

    # pointwise multiplication in frequency domain
    if return_all:
        return H_gaus_2*H_excitation, H_gaus_2, H_excitation
    else:
        return H_gaus_2*H_excitation


def gausspuls_coeff(fc, bandwidth, bwr=-6.):
    ref = pow(10.0, bwr / 20.0)
    return -(np.pi * fc * bandwidth) ** 2 / (4.0 * np.log(ref))


def gausspulse_ft(f_vals, alpha, fc):
    """
    Also mentioned in comment of scipy docs: https://github.com/scipy/scipy/blob/v1.1.0/scipy/signal/waveforms.py#L165-L261
    
    g(t) = exp(-a t^2 ) * cos(2*pi*fc*t) --> G(f) = sqrt(pi/a) * exp(-(pi*f)^2 / a) conv [d(f-fc)+d(f+fc)]/2
    """
    
    pos_half = np.sqrt(np.pi/alpha) * np.exp(-1.*(np.pi*(f_vals-fc))**2 / alpha)
    neg_half = np.sqrt(np.pi/alpha) * np.exp(-1.*(np.pi*(f_vals+fc))**2 / alpha)
    return (pos_half + neg_half)/2


def square_excitation_ft(f_vals, n_cycles, center_freq, centered=True):
    """
    :param f_vals: Frequency values for which to compute Fourier Transform.
    :param n_cycles: Number of cycles for the square pulse excitation. It will 
    truncated to the nearest x.0 or x.5
    :param center_freq: 1/period of the square pulse
    :param centered: (boolean) whether or not the square pulse is centered around
    t0 (True) or if it starts at t=0 (False).
    """

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








def gen_echoes(x_coord, z_coord, array_pos, speed_sound):

    # compute roundtrip times (tx+rx for point sources in each channel)
    z_coord_mesh, array_pos_mesh = np.meshgrid(z_coord, array_pos)
    x_coord_mesh = np.meshgrid(x_coord, array_pos)[0]

    # transmit time
    tx_time = z_coord_mesh / speed_sound

    # receive time
    rx_time_x = (x_coord_mesh-array_pos_mesh) / speed_sound
    rx_time_z = tx_time
    rx_time = np.sqrt(rx_time_z*rx_time_z + rx_time_x*rx_time_x)

    return (tx_time + rx_time)


def estimate_2d_loc(array_pos, sorted_echoes, speed_sound, avg=True):
    """
    For a plane wave of normal incidence, we can express the z coordinate of a 
    2D point as:

    zk = tau_mk * speed_sound/2 - (ym - xk)^2 / (2*tau_mk*speed_sound)

    where xk is the x coordinate, ym is the position of the sensor on the 
    emission surface, and tau_mk is the time of flight for transmission and
    reception.

    :param array_pos: length M vector containing the sensor positions, where M 
    is the number of sensors.
    :param sorted_echoes: M*P matrix where M is number of sensors and P is the number
    of 2D points. The echoes in each column should correspond to the same 2D
    point!
    :param speed_sound: in meters/second
    :param avg: whether or not to obtain an estimate for all pairs and average
    the result for each point. If False, the sensors that are furthest apart 
    will be used.

    TODO : for plane waves that are not normal incidence
    """

    echoes_shape = sorted_echoes.shape
    if len(echoes_shape) > 1: 
        n_points = echoes_shape[1]
    else:
        n_points = 1
    n_sensors = echoes_shape[0]

    x_coord_est = np.zeros(n_points)
    z_coord_est = np.zeros(n_points)
    pair_combos = list(combinations(array_pos,2))

    if avg:
        n_combos = len(pair_combos)
        x_coord_k = np.zeros(n_combos)
        z_coord_k = np.zeros(n_combos)
    else: # find array sensors furthest apart
        dist = [np.abs(pair[0]-pair[1]) for pair in pair_combos]
        max_dist_pair_idx = np.argmax(dist)
        max_dist_pair = pair_combos[max_dist_pair_idx]


    for k in range(n_points):

        if len(echoes_shape) > 1:
            time_combos = list(combinations(sorted_echoes[:,k], 2))
        else:
            time_combos = list(combinations(sorted_echoes, 2))

        if avg:
            for p in range(n_combos):
                x_coord_k[p], z_coord_k[p] = localize_point(pair_combos[p], time_combos[p], speed_sound)
            x_coord_est[k] = np.mean(x_coord_k[~np.isnan(x_coord_k)])
            z_coord_est[k] = np.mean(z_coord_k[~np.isnan(z_coord_k)])

        else:
            x_coord_est[k], z_coord_est[k] = localize_point(max_dist_pair, time_combos[max_dist_pair_idx], speed_sound)

    order_x = np.argsort(x_coord_est)
    x_coord_est = x_coord_est[order_x]
    z_coord_est = z_coord_est[order_x]

    return x_coord_est, z_coord_est


def z_from_t_and_dx(t, dx, speed_sound):
    return t*speed_sound/2 - dx*dx/2/t/speed_sound


def localize_point(pos_tup, echo_tup, speed_sound):
    x0 = pos_tup[0]
    x1 = pos_tup[1]
    t0_hat = echo_tup[0]
    t1_hat = echo_tup[1]

    # solve quadratic formula
    a = (t0_hat-t1_hat)
    b = (2*t1_hat*x0 - 2*t0_hat*x1)
    c = (t0_hat*speed_sound)**2 * t1_hat - t1_hat*(x0)**2 - (t1_hat*speed_sound)**2 * t0_hat + t0_hat*(x1)**2
    d = b*b - 4*a*c

    x_hat = np.array([(-b - np.sqrt(d))/(2*a), 
        (-b + np.sqrt(d))/(2*a)])
    t = t0_hat
    z_hat = z_from_t_and_dx(t, x0-x_hat, speed_sound)

    # take one that has positive z
    if z_hat[0] < z_hat[1]:
        return x_hat[1], z_hat[1]
    else:
        return x_hat[0], z_hat[0]


def compute_snr_db(y_true, y_hat):

    return 20*np.log10(np.linalg.norm(y_true)/np.linalg.norm(y_true-y_hat))


def loc_2d_snr(x_coord_true, z_coord_true, x_coord_est, z_coord_est):

    # order both by x
    order_x = np.argsort(x_coord_true)
    x_coord_true = x_coord_true[order_x]
    z_coord_true = z_coord_true[order_x]

    order_x = np.argsort(x_coord_est)
    x_coord_est = x_coord_est[order_x]
    z_coord_est = z_coord_est[order_x]

    # compute errors along x and z
    n_points = len(x_coord_true)
    x_snr = compute_snr_db(x_coord_true, x_coord_est)
    z_snr = compute_snr_db(z_coord_true, z_coord_est)
    
    # compute 2d error
    true_points = np.concatenate((x_coord_true[:,np.newaxis], z_coord_true[:,np.newaxis]), axis=1).T
    est_points = np.concatenate((x_coord_est[:,np.newaxis], z_coord_est[:,np.newaxis]), axis=1).T
    snr_2d = np.zeros(n_points)
    for k in range(n_points):
        snr_2d[k] = compute_snr_db(true_points[:,k], est_points[:,k])
    snr_2d[snr_2d>500] = 500
    avg_snr_2d = np.mean(snr_2d)
    std_snr_2d = np.std(snr_2d)

    return (avg_snr_2d, std_snr_2d) , x_snr, z_snr


def convolve_time(x1, x2, t1, t2, samp_freq):

    t_start = t1[0] + t2[0]
    t_end = t1[-1] + t2[-1]

    n_samp_conv = len(x1) + len(x2) - 1
    t_conv = np.arange(t_start, t_end+1./samp_freq, 1./samp_freq)
    t_conv = t_conv[:n_samp_conv]

    res = np.convolve(x1, x2)

    return res, t_conv


def compute_dft(sig, samp_freq):
    """
    Compute and return DFT of given signal and it's corresponding frequency
    values.
    """

    n_samples = len(sig)
    H = np.fft.fft(sig) / n_samples
    f = np.fft.fftfreq(n_samples,1./samp_freq)

    half_spec = int(np.ceil(n_samples/2))

    H = np.concatenate((H[half_spec:],H[:half_spec]))
    f = np.concatenate((f[half_spec:],f[:half_spec]))

    return H, f


def create_gaussian_pulse(center_freq, samp_freq, bandwidth=2/3, bwr=-6, 
    tpr=-60, baseband=False, viz=False):
    """
    Modification to `scipy.signal.gausspulse`:
    https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/waveforms.py#L161

    Their version doesn't let you set a center frequency of 0 (baseband case).

    :return:
    """

    # compute within [+/- cutoff]
    t_cutoff = gausspulse('cutoff', fc=center_freq, bw=bandwidth, bwr=bwr, 
        tpr=tpr)

    # gaussian coeff (Time domain compressive beam forming of ultrasound signals, David)
    denom = 2 * np.pi * bandwidth * center_freq
    tv = -(8 * np.log(10 ** (bwr / 20)) / (denom * denom))
    alpha = 1. / 2 / tv

    # compute pulse
    t_vals = np.arange(-1. * t_cutoff, t_cutoff, 1 / samp_freq)
    pulse = np.exp(-alpha * t_vals * t_vals)
    if not baseband:
        pulse *= np.cos(2 * np.pi * center_freq * t_vals)

    if viz:

        f, (ax1, ax2) = plt.subplots(2,1)
        ax1.set_title("Gaussian pulse")
        ax1.plot(t_vals, pulse)
        ax1.grid()
        ax1.autoscale(enable=True, axis='x', tight=True)
        ax1.set_xlabel("Time [s]")
        # ax1.get_xaxis().set_ticklabels([])

        if baseband:
            f_vals = np.linspace(-center_freq, center_freq, num=1000)
            H = gauss_ft(f_vals, alpha)
            ax2.plot(f_vals, 20*np.log10(np.abs(H)))

        else:
            f_vals = np.linspace(-2*center_freq, 2*center_freq, num=1000)
            H = gauss_ft(f_vals, alpha, fc=center_freq)
            ax2.plot(f_vals, 20*np.log10(np.abs(H)), 
                label="pulse")
            ax2.axvline(x=center_freq, c='r', label="center frequency")
            ax2.legend()

        ax2.grid()
        ax2.set_xlabel("Frequency [Hz]")
        ax2.autoscale(enable=True, axis='x', tight=True)
        ax2.set_ylabel("dB")

    return pulse, t_vals, alpha


def rect_excitation_ft(f_vals, dur):

    a = 1/dur

    return np.sinc(f_vals/a)


def create_square_excitation(n_cycles, center_freq, samp_freq, data_type=None,
    viz=False):

    n_rects = int(n_cycles/0.5)
    if abs(n_rects*0.5-n_cycles) > 0.01:
        print("Removing extra %f of cycle..." % abs(n_rects*0.5-n_cycles))

    n_cycles = n_rects*0.5
    n_neg = n_rects//2
    n_pos = n_rects-n_neg

    t_stop = int(n_cycles / center_freq * samp_freq) / samp_freq
    n_samp = int(n_cycles / center_freq * samp_freq) + 1 
    t_excite = np.linspace(0, t_stop, n_samp)
    if data_type is not None:
        excitation = square(2 * np.pi * center_freq * t_excite).astype(data_type)
    else:
        excitation = square(2 * np.pi * center_freq * t_excite)

    if viz:

        # compute analytic FT and time domain
        H_pulse = np.fft.rfft(excitation)
        f_pulse = np.linspace(0, samp_freq/2, len(H_pulse))
        H_an = square_excitation_ft(f_pulse, n_cycles, center_freq, centered=False)
        pulse_an = np.fft.irfft(H_an)
        time_an = np.arange(len(pulse_an))/samp_freq


        f, (ax1, ax2) = plt.subplots(2,1)
        ax1.set_title("Square pulse excitation")
        ax1.plot(t_excite, excitation, label="num")
        ax1.plot(time_an, pulse_an/max(pulse_an), label="irfft(analytic)")
        ax1.grid()
        ax1.autoscale(enable=True, axis='x', tight=True)
        ax1.set_xlabel("Time [s]")
        ax1.legend()

        ax2.semilogx(f_pulse, 20*np.log10(np.abs(H_pulse/max(abs(H_pulse)))), label="excitation")
        ax2.semilogx(f_pulse, 20*np.log10(np.abs(H_an/max(abs(H_an)))), label="analytic")
        ax2.axvline(x=center_freq, c='r', label="center frequency")
        ax2.grid()
        ax2.legend()
        ax2.set_xlabel("Frequency [Hz]")
        ax2.autoscale(enable=True, axis='x', tight=True)
        ax2.set_ylabel("dB")
        ax2.set_ylim([-40, 0])

    return excitation, t_excite, n_cycles



class Probe:

    def __init__(self, n_elements, pitch, samp_freq, 
        oversample_fact=50, dtype=np.float32):

        """
        oversample_fact : factor for oversampling to create an "analog" signal 
        which will be low-pass filtered and downsampled appropriately.
        """

        self.data_type = np.float32

        self.n_elements = n_elements
        self.pitch = pitch
        
        self.oversample_fact = oversample_fact # to "fake" analog signal
        self.samp_freq_a = oversample_fact * samp_freq
        self.samp_time_a = 1/self.samp_freq_a
        self.samp_freq = samp_freq
        self.samp_time = 1/samp_freq

        self.center_freq = None
        self.bandwidth = None
        self.tpr = None
        self.impulse_response = None
        self.excitation = None
        self.total_pulse = None

        # create uniformly spaced linear array
        self.array_pos = np.arange(n_elements)*pitch
        self.array_pos -= np.mean(self.array_pos)  # center

        # medium to image
        self.medium = None


    def set_response(self, center_freq, n_cycles, bandwidth=2/3, bwr=-6, 
        viz=False):

        """
        Set both impulse response and excitation. From this compute the total
        response.

        :param center_freq: Center frequency for gaussian pulse.
        :param n_cycles: number of cycles for excitation. Setting to 0 will
            assume dirac-like excitation.
        :param bandwidth: Bandwidth for the modulated gaussian pulse
        """

        # impulse response
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self.n_cycles = n_cycles
        self.bwr = bwr

        # # compute total response from analytic form of Fourier Transform
        # n_samples = 2*len(t_ir) + len(t_excite) - 2
        # self.f_pulse = np.linspace(0, self.samp_freq_a/2, n_samples//2+1)
        # total_freq_response(freqs, center_freq, bandwidth, num_cycles, bwr=-6,
        #     return_all=False)

        # alpha_2 = gausspuls_coeff(center_freq, np.sqrt(2)*bandwidth/2, bwr=bwr)



        # self.impulse_response, t_ir, self.alpha = create_gaussian_pulse(
        #     center_freq, self.samp_freq_a, bandwidth=bandwidth, tpr=tpr)
        # self.impulse_response.astype(self.data_type)

        # # excitation
        # self.excitation, t_excite, self.transmit_cycles = create_square_excitation(n_cycles, 
        #     center_freq, self.samp_freq_a, data_type=self.data_type)

        # # compute total response from analytic form of Fourier Transform
        # n_samples = 2*len(t_ir) + len(t_excite) - 2
        # self.f_pulse = np.linspace(0, self.samp_freq_a/2, n_samples//2+1) 
        # # gaus pulse convovled with itself is still gaus pulse but with increased bandwidth
        # alpha_2 = create_gaussian_pulse(center_freq, self.samp_freq_a, 
        #     bandwidth=np.sqrt(2)*bandwidth/2, tpr=tpr)[2]
        # self.H_gaus_2 = gauss_ft(self.f_pulse, alpha_2, fc=center_freq)
        # if self.transmit_cycles > 0:
        #     self.H_excitation = square_excitation_ft(self.f_pulse, self.transmit_cycles, center_freq)
        # else:
        #     self.H_excitation = np.ones(len(self.f_pulse))
        # self.H_tot = self.H_gaus_2 * self.H_excitation

        # self.pulse = np.fft.irfft(self.H_tot)
        # self.pulse = np.roll(self.pulse, shift=len(self.pulse)//2).astype(self.data_type)
        # self.t_pulse = np.arange(len(self.pulse))*self.samp_time_a

        # if viz:

        #     f, (ax1, ax2) = plt.subplots(2,1)

        #     ax1.set_title("Transducer impulse response and excitation (%dx oversampling)" % self.oversample_fact, fontsize=18)
        #     ax1.set_ylabel("Impulse response", fontsize=18)
        #     ax1.plot(t_ir, self.impulse_response)
        #     ax1.grid()
        #     ax1.autoscale(enable=True, axis='x', tight=True)

        #     ax2.set_ylabel("Excitation", fontsize=18)
        #     if n_cycles == 0:  # dirac-like excitation
        #         excitation = np.zeros(11)
        #         excitation[5] = 1
        #         ax2.stem(np.arange(11)/10-0.5, excitation)
        #     else:
        #         ax2.plot(t_excite, self.excitation)
        #     ax2.grid()
        #     ax2.autoscale(enable=True, axis='x', tight=True)
        #     ax2.set_xlabel("Time [s]", fontsize=18)

        #     f, (ax1, ax2) = plt.subplots(2,1)
        #     ax1.set_title("Total response / pulse (%dx oversampling)" % self.oversample_fact, fontsize=18)
        #     ax1.plot(self.t_pulse, self.pulse/max(self.pulse))
        #     ax1.grid()
        #     ax1.autoscale(enable=True, axis='x', tight=True)
        #     ax1.set_xlabel("Time [s]", fontsize=18)
        #     ax1.get_xaxis().set_ticklabels([])

        #     ax2.semilogx(self.f_pulse, 20*np.log10(np.abs(self.H_tot/max(abs(self.H_tot)))), 
        #         label="pulse")
        #     ax2.axvline(x=self.center_freq, c='r', label="center frequency")
        #     ax2.grid()
        #     ax2.legend()
        #     ax2.set_xlabel("Frequency [Hz]", fontsize=18)
        #     ax2.autoscale(enable=True, axis='x', tight=True)
        #     ax2.set_ylabel("dB")
        #     ax2.set_ylim([-40, 0])


    def visualize_medium_rec(self):

        if self.medium is None:
            raise ValueError("No medium given to image!")

        MARKER_SIZE = 90

        f, (ax1, ax2) = plt.subplots(1, 2)

        # ground truth 2d image with array placement
        ax1.set_title("Reflectors")
        ax1.scatter(self.array_pos, np.zeros(self.n_elements) , c='r', 
            marker= 's', label="Transducers", s=MARKER_SIZE)
        for k in range(self.medium.n_points):
            if k==0:
                ax1.scatter(self.medium.x_coord[k], self.medium.z_coord[k], 
                    label="Point source")
            else:
                ax1.scatter(self.medium.x_coord[k], self.medium.z_coord[k])
        ax1.set_xlim(self.array_pos[0],self.array_pos[-1])
        ax1.set_ylabel("Depth [cm]")
        
        depth_cm = self.medium.depth*100
        ax1.set_yticks(np.arange(int(np.ceil(depth_cm)+1))/100) 
        ax1.set_yticklabels(np.arange(int(np.ceil(depth_cm)+1)))
        ax1.set_ylim(0,self.medium.depth)
        ax1.invert_yaxis()

        ax1.set_xticks(self.array_pos)
        ax1.set_xticklabels(np.around(self.array_pos*10000,2))
        ax1.set_xticklabels("")
        ax1.grid()

        # Shrink current axis's height by 10% on the bottom to fit legend
        # box = ax1.get_position()
        # ax1.set_position([box.x0, box.y0 + box.height * 0.1,
        #                  box.width, box.height * 0.9])
        # ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
        #   fancybox=True, shadow=True, ncol=5)


        # measurements along each grid line
        scal_fact = 1e-5
        ax2.set_title("TOFs")

        ax2.scatter(self.array_pos, np.zeros(self.n_elements) , c='r', 
            marker= 's', s=MARKER_SIZE)
        for k in range(self.medium.n_points):
            if k==0:
                ax2.scatter(self.array_pos, self.round_trip[:,k], marker='^',
                    label="Recorded point")
            else:
                ax2.scatter(self.array_pos, self.round_trip[:,k], marker='^')
        ax2.set_yticks(np.arange(int(np.ceil(self.duration/scal_fact)+1))*scal_fact) 
        ax2.set_yticklabels(np.arange(int(np.ceil(self.duration/scal_fact)+1)))
        ax2.set_ylim(0,self.duration)
        ax2.set_ylabel("Time [%s]" % str(scal_fact))
        ax2.invert_yaxis()

        ax2.set_xlim(self.array_pos[0],self.array_pos[-1])
        ax2.set_xticks(self.array_pos)
        ax2.set_xticklabels("")
        ax2.grid()

        # # Shrink current axis's height by 10% on the bottom to fit legend
        # box = ax2.get_position()
        # ax2.set_position([box.x0, box.y0 + box.height * 0.1,
        #                  box.width, box.height * 0.9])
        # ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
        #   fancybox=True, shadow=True, ncol=5)

        f.tight_layout()



    def record(self, medium, depth, angle=0, snr_db=float('inf'), viz=False):
        """
        Simulate plane wave recording for specified depth in centimeters.

        Assuming sinc sampling kernel, whose BW = samp_freq

        TODO: add noise
              implement `angle` for tilting plane wave.
              seperate recording/measurement class.
              only receive points in direct path from plane wave
              raw data viz --> color map
        """

        self.medium = medium

        # compute duration and number of samples
        self.duration = 2*depth/medium.speed_sound  # measurement time, back and forth [s]

        # compute roundtrip times (tx+rx for point sources in each channel)
        z_coord_mesh, array_pos_mesh = np.meshgrid(medium.z_coord, self.array_pos)
        x_coord_mesh = np.meshgrid(medium.x_coord, self.array_pos)[0]

        # transmit time
        self.tx_time = z_coord_mesh / medium.speed_sound

        # receive time
        rx_time_x = (x_coord_mesh-array_pos_mesh) / medium.speed_sound
        rx_time_z = self.tx_time
        self.rx_time = np.sqrt(rx_time_z*rx_time_z + rx_time_x*rx_time_x)

        self.round_trip = self.tx_time + self.rx_time
        self.amps = 1/(4*np.pi*self.rx_time) * medium.amps

        """ Assuming ideal low-pass filter + using analytic form of pulse """
        n_samples = int(np.ceil(self.duration/self.samp_time))
        self.t_rec = np.arange(n_samples) * self.samp_time
        self.recordings = np.zeros((self.n_elements, n_samples), 
            dtype=self.data_type)

        # frequencies within bandwidth and ideal pulse shape
        freqs = np.fft.fftfreq(n_samples, self.samp_time)
        H_puls = total_freq_response(freqs,
            self.center_freq, self.bandwidth, 
            self.n_cycles, self.bwr)

        # compute each channel's recording
        for idx_elem in range(self.n_elements):

            # ground truth FS coefficients
            tk_grid, freqs_grid = np.meshgrid(self.round_trip[idx_elem,:], freqs)
            fs_coeff = np.dot(np.exp(-1j*2*np.pi*freqs_grid*tk_grid), 
                self.amps[idx_elem,:]) / self.duration
            fs_coeff *= H_puls

            # compute samples, around baseband so real
            self.recordings[idx_elem,:] = np.real(np.fft.ifft(fs_coeff) * 
                n_samples).astype(self.data_type)

        if viz:

            scal_fact = 1e-5
            f, axes = plt.subplots(self.n_elements,1)

            # axes[0].set_title("Recorded signals, starting from (0,0)")

            for m, ax in enumerate(axes):
                ax.plot(self.t_rec, self.recordings[m,:])
                ax.grid()
                ax.set_ylabel("Chan. %d" % (m+1), fontsize=18)
                ax.autoscale(enable=True, axis='x', tight=True)
                ax.get_yaxis().set_ticklabels([])
                ax.set_xticks(np.arange(int(np.ceil(self.duration/ \
                    scal_fact)+1))*scal_fact) 
                ax.set_xlim(0,self.duration)
                ax.get_xaxis().set_ticklabels([])

            axes[-1].set_xticklabels(np.arange(int(np.ceil(self.duration/ \
                scal_fact)+1)))
            axes[-1].set_xlabel("Time [%s seconds]" % str(scal_fact), fontsize=18)
            f.tight_layout()

        return self.recordings



class Medium:

    def __init__(self, width, depth, speed_sound, min_depth=0, n_points=None, x_coord=None, 
        z_coord=None, amps=None, seed=0, unit_amp=False):

        """
        Place `n_points` randomly or (TODO) specify location of pulses.
        """

        self.width = width
        self.depth = depth
        self.min_depth = min_depth
        self.speed_sound = speed_sound


        if n_points is not None:    # randomly place points

            rng = np.random.RandomState(seed)

            self.n_points = n_points

            self.x_coord = rng.rand(n_points)*width - width/2
            self.z_coord = rng.rand(n_points)*(self.depth-self.min_depth) + self.min_depth
            self.amps = rng.randn(n_points)

        if unit_amp:
            self.amps = np.sign(self.amps)


    def visualize(self):

        plt.figure()
        plt.title("GROUND TRUTH IMAGE")

        plt.scatter(self.x_coord[0], self.z_coord[0], 
                    label="Point sources", c='b')
        plt.scatter(self.x_coord[1:], self.z_coord[1:], c='b')
        plt.xlim(0,self.width)

        plt.xlabel("Width [m]")
        plt.ylabel("Depth [cm]")
        depth_cm = self.depth*100

        plt.yticks(np.arange(int(np.ceil(depth_cm)+1))/100,
            np.arange(int(np.ceil(depth_cm)+1)))
        plt.ylim(0,self.depth)

        ax = plt.gca()
        ax.invert_yaxis()

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.legend(loc='upper center', bbox_to_anchor=(0.8, -0.05),
          fancybox=True, shadow=True, ncol=5)

        plt.grid()






