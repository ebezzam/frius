import numpy as np
from scipy.linalg import solve_toeplitz, toeplitz, svd, lstsq
import matplotlib.pyplot as plt
from scipy import linalg
import warnings

from .us_utils import total_freq_response


def create_pulse_param(K, period=1, seed=0, unit_amp=False, pos=False,
    viz=False, figsize=None, fontsize=20):
    """
    Parameters
    ----------
    K : int
        Number of pulses.
    period : float
        Period/duration of pulse stream in seconds.
    seed : int
        Seed for random number generator to obtain pulse location and 
        amplitudes. Default is 0.
    unit_amp : bool
        Set all weights ``ck`` to unit amplitudes. Default is False.
    pos : bool
        Set all weights ``ck`` to be positive. Default is False.
    viz : bool
        Whether or not to plot the pulse parameters. Default is False.
    figsize : tuple of ints
        Size of figure.
    fontsize : int
        Size of font for title and xlabel.
    """
    
    rng = np.random.RandomState(seed)
    
    # amplitudes and locations
    ck = rng.randn(K)
    tk = np.sort(rng.rand(K)*period)

    if unit_amp:
        ck = np.sign(ck)

    if pos:
        ck = np.abs(ck)
    
    if viz:
        if figsize is not None:
            plt.figure(figsize=figsize)
        else:
            plt.figure()
        plt.title("Signal parameters", fontsize=fontsize)
        baseline = plt.stem(tk, ck, 'g', markerfmt='go',)[2]
        plt.setp(baseline, color='g')
        baseline.set_xdata([0, period])
        plt.xlim([0,period])
        plt.grid()
        plt.xlabel("Time [seconds]", fontsize=fontsize)
        
    return ck, tk


def sample_ideal_project(ck, tk, period, fc=0, K=None, H=None,
    oversample_freq=1, n_samples=None, samp_freq=None,
    viz=False, figsize=None, fontsize=20):
    """
    Sample ideal projection (box filter) of pulse stream.
    
    Parameters
    ----------
    ck : numpy float array
        Pulse amplitudes.
    tk : numpy float array
        Pulse locations.
    period : float
        Period/duration of pulse stream in seconds.
    fc : float
        Center frequency of ideal kernel (box). Default is 0, i.e. lowpass.
    K : int
        Number of assumed pulses for critical sampling.
    oversample_freq : float
        Oversampling factor of frequency coefficients by increasing the 
        bandwidth of the necessary filter and taking more samples. Bandwidth
        is increased (in terms of Fourier Series coefficients) from [-K,K] to  
        [-int(oversample_freq*K),int(oversample_freq*K)]. Default is 1.
    viz : bool
        Whether or not to plot the pulse parameters. Default is False.
    figsize : tuple of ints
        Size of figure.
    fontsize : int
        Size of font for title and xlabel.
    """
    
    if len(ck)!=len(tk):
        raise ValueError("Must have equal number of amplitudes and locations!")
    if K is None:
        K = len(ck)
        
    # bandwidth of sampling kernel
    M = int(oversample_freq*K)
    fs_ind_base = np.arange(-M, M+1)
    if n_samples is None:
        n_samples = 2*M+1
    else:
        if n_samples < len(fs_ind_base):
            raise ValueError("Number of samples must be >= number of Fourier coefficients.")
    if samp_freq is not None:
        n_samples = int(np.round(period*samp_freq))
        M = n_samples//2
        fs_ind_base = np.arange(-M, M+1)
    
    # ground truth Fourier coefficient within our bandpass filter
    tk_grid, freqs_grid = np.meshgrid(tk, fs_ind_base/period+fc)
    fs_coeff = np.dot(np.exp(-1j*2*np.pi*freqs_grid*tk_grid), ck)/period
    if H is not None:
        fs_coeff *= H
    
    # compute samples
    Ts = period / n_samples
    t_samp = np.arange(n_samples)*Ts
    freqs_grid, t_samp_grid = np.meshgrid(fs_ind_base/period+fc, t_samp)
    W_idft = np.exp(1j*2*np.pi*freqs_grid*t_samp_grid)
    y_samp = np.dot(W_idft, fs_coeff)
    
    if fc==0:  # conjugate symmetric kernel so expect real samples!
        y_samp = np.real(y_samp)
        
    if viz:
        BW = 1/Ts

        if figsize is not None:
            plt.figure(figsize=figsize)
        else:
            plt.figure()
        plt.title("Signal parameters", fontsize=fontsize)
        baseline = plt.stem(tk, ck, 'g', markerfmt='go', 
            label="Signal parameters")[2]
        plt.setp(baseline, color='g')
        baseline.set_xdata([0, period])
        if fc==0:
            plt.scatter(t_samp, y_samp/BW, label="Projected samples (LPF)")
            plt.plot(t_samp, y_samp/BW)
        else: # plot real and imaginary separately
            plt.scatter(t_samp, np.real(y_samp), 
                label="Projected samples (BP, real)")
            plt.plot(t_samp, np.real(y_samp))
            plt.scatter(t_samp, np.imag(y_samp), 
                label="Projected samples (BP, imag)")
            plt.plot(t_samp, np.imag(y_samp))
        plt.xlim([0,period])
        plt.grid()
        plt.legend(fontsize=fontsize)
        plt.xlabel("Time [seconds]", fontsize=20);
        
    return y_samp, t_samp, fs_ind_base


def estimate_fourier_coeff(y_samp, t_samp, fs_ind=None, fc=0, H=None):
    """
    Estimate Fourier Series coefficients from uniform samples.
    """
    n_samples = len(y_samp)
    if not (n_samples%2):
        raise ValueError("Only accept odd number of samples.")
    if fs_ind is None:
        n_coeff = n_samples
    else:
        n_coeff = len(fs_ind)

    if n_samples==n_coeff:
        freqs_fft = np.fft.fftfreq(n_samples)
        increasing_order = np.argsort(freqs_fft)
        fs_coeff_hat = (np.fft.fft(y_samp*np.exp(-2j*np.pi*fc*t_samp))/n_samples)[increasing_order]

    else:
        # DFT at corresponding frequencies
        t_samp_grid, fs_ind_grid = np.meshgrid(t_samp, fs_ind)
        W = np.exp(-1j*2*np.pi*fs_ind_grid*t_samp_grid) / n_samples
        fs_coeff_hat = np.dot(W, y_samp)

    # equalize to remove pulse/channel effect
    if H is not None:
        fs_coeff_hat /= H

    return fs_coeff_hat


def compute_ann_filt(fs_coeff, K):
    """
    Compute annihilating filter of length ``K+1``, e.g. for ``K`` complex
    exponentials from the Fourier coefficients given by ``fs_coeff``.

    Parameters
    ----------
    fs_coeff : numpy float array
        At least ``2K+1`` Fourier coefficients to annihilate.
    K : int
        Number of complex exponentials.
    """

    n_coeff = len(fs_coeff)
    if n_coeff <= 2*K:
        raise ValueError("Need at least %d Fourier coefficients to \
            annihilate %d complex exponentials!" % (2*K+1, K))
    elif n_coeff==(2*K+1):
        A_1K = solve_toeplitz(
            c_or_cr=(fs_coeff[K:-1], fs_coeff[K:0:-1]),
            b=-1*fs_coeff[K+1:])
        ann_filt = np.insert(A_1K, 0, 1)
    else: # subspace approach for oversampled case, total least squares
        col1 = fs_coeff[K:]
        row1 = np.flipud(fs_coeff[:K+1])
        A_top = toeplitz(col1, r=row1)
        U, s, Vh = svd(A_top)
        ann_filt = np.conj(Vh[-1, :])  

    return ann_filt


def estimate_time_param(ann_filt, period):
    """
    Compute time parameters for periodic delta stream from annihilitating 
    filter.

    From the roots `uk` for the annihilitating filter; solve for time parameters
    `tk` as such:

    tk = 1j * period * ln(uk) / (2 * pi)

    :param ann_filt: annihilitating filter
    :param period: period of the Dirac stream
    """

    # solve for time locations
    uk = np.roots(ann_filt)
    uk = uk / np.abs(uk)
    tk_hat = np.real(1j * period * np.log(uk) / (2 * np.pi))
    tk_hat = tk_hat - np.floor(tk_hat / period) * period

    return tk_hat


def estimate_amplitudes(fs_coeff_hat, freqs, tk_hat, period, return_cond=False):
    """
    Estimate amplitudes for periodic delta stream given estimated locations.

    We know that the Fourier coefficients are given by:

    X[m] = (1/period) * <ck, exp(-j*2*pi*m*tk/period)>

    Or in matrix form:

    X = Phi * ck,

    where the mth row of Phi is given by:

    (Phi)_m = exp(-j*2*pi*m*tk/period)

    So given the estimate Fourier coefficients `fs_coeff_hat` and the estimated 
    locations `tk_hat`, we can solve for `ck` in a least square fashion.

    :param fs_coeff_hat: estimated FS coefficients
    :param freqs: corresponding frequencies
    :param tk_hat: estimated dirac locations
    :param n_diracs: number of expected diracs
    :param period: period/duration of the dirac stream
    :param return_cond: return condition number of Phi matrix
    """

    tk_grid, freqs_grid = np.meshgrid(tk_hat, freqs)
    Phi = np.exp(-1j*2*np.pi*freqs_grid*tk_grid)/period
    ck_hat = np.real(lstsq(Phi, fs_coeff_hat)[0])

    if return_cond:
        s = svd(Phi, full_matrices=False, compute_uv=False)
        cond = s[0]/s[-1]
        return ck_hat, cond
    else:
        return ck_hat


def evaluate_recovered_param(ck, tk, ck_hat, tk_hat, viz=False, figsize=None, 
    fontsize=20, t_max=None):
    """
    Compute L2 error for recovered amplitude and locations parameters. In this 
    function, we simply sort in increasing time order. A better metric (in a 
    later notebook) would compare pulses that are closest together.
    
    This metric requires the recovered number of pulses to be equal to the true 
    number of pulses.
    
    Parameters
    ----------
    ck : numpy float array
        Pulse amplitudes.
    tk : numpy float array
        Pulse locations.
    ck_hat : numpy float array
        Recovered pulse amplitudes.
    tk_hat : numpy float array
        Recovered pulse locations.
    viz : bool
        Whether or not to plot the pulse parameters. Default is False.
    figsize : tuple of ints
        Size of figure.
    fontsize : int
        Size of font for title and xlabel.
    t_max : float
        Maximum length to plot along x-axis.
    """
    
    increasing_order = np.argsort(tk)
    tk = tk[increasing_order]
    ck = ck[increasing_order]
    
    increasing_order = np.argsort(tk_hat)
    tk_hat = tk_hat[increasing_order]
    ck_hat = ck_hat[increasing_order]
    
    print("||tk - tk_hat||_2 = %f " % np.linalg.norm(tk-tk_hat))
    print("||ck - ck_hat||_2 = %f " % np.linalg.norm(ck-ck_hat))
    
    if viz:
        if t_max is None:
            t_max = max(max(tk), max(tk_hat))
            
        if figsize is not None:
            plt.figure(figsize=figsize)
        else:
            plt.figure()
        plt.title("FRI reconstruction", fontsize=fontsize)

        baseline = plt.stem(tk, ck, 'g', markerfmt='go', label="True")[2]
        plt.setp(baseline, color='g')
        baseline.set_xdata([0, t_max])

        baseline = plt.stem(tk_hat, ck_hat, 'r', markerfmt='r^', 
            label="Estimate")[2]
        plt.setp(baseline, color='r')
        baseline.set_xdata([0, t_max])

        plt.xlim([0,t_max])
        plt.grid()
        plt.legend(fontsize=fontsize)
        plt.xlabel("Time [seconds]", fontsize=fontsize);
    

def cadzow_denoising(fs_coeff, K, L=None, n_iter=10):
    """
    Perform Cadzow denoising to project the given noisy Fourier coefficients
    onto a set of coefficients whose Toeplitz matrix has a rank 
    of ``K`` as expected for the Toeplitz matrix of Fourier coefficients of an 
    FRI signal composed of ``K`` degrees of freedom in the innovation locations.

    See "Sparse Sampling of Signal Innovations" (Blu 2008).

    Parameters
    ----------
    fs_coeff : numpy float array
        At least ``2K+1`` Fourier coefficients. Odd number of Fourier 
        coefficients in increasing order of frequency.
    K : int
        Target rank after Cadzoe denoising.
    L : int
        Number of columns is equal to L+1. Default will create square
        Toeplitz matrix (recommended).
    n_iter : int
        Number of iterations to run.
    """

    n_coeff = len(fs_coeff)
    if n_coeff <= 2*K:
        raise ValueError("Need at least %d Fourier coefficients!" % (2*K+1))

    if not (n_coeff%2):
        raise ValueError("Only accept odd number of coefficients")

    if L is None:
        # p. 36 of "Sparse Sampling of Signal Innovations", create square matrix
        L = int(np.floor(0.5*n_coeff))

    if L < K:
        raise ValueError("L should be larger than %d!" % K)

    # fs_coeff_tild = fs_coeff[L:]
    col1 = fs_coeff[L:]
    row1 = np.flipud(fs_coeff[:L+1])
    for k in range(n_iter):

        # create toeplitz, ideally square
        A_top = toeplitz(col1, r=row1)
        U, s, Vh = svd(A_top)

        # project onto matrix of target rank (K)
        s_tild = s
        s_tild[K:] = 0
        A_top_tild = np.dot(np.dot(U, np.diag(s_tild)), Vh)

        # average along diagonals for denoised approximation of FS coefficients
        fs_coeff_denoised = np.zeros(n_coeff, dtype=np.complex)
        diag_idx = np.arange(n_coeff) - L
        for i, m in enumerate(diag_idx):
            fs_coeff_denoised[i] = np.mean(np.diag(A_top_tild, m))

        col1 = fs_coeff_denoised[L:]
        row1 = np.flipud(fs_coeff_denoised[:L+1])

    return fs_coeff_denoised


def compute_srr_db(y_true, y_hat):
    """
    Compute signal-to-residual ratio (in dB) between true signal ``y_true`` and 
    estimated residuak ``y_true-y_hat``.

    Parameters
    ----------
    y_true : numpy float array, 1D
        True values.
    y_hat : numpy float array, 1D
    """

    return 20*np.log10(np.linalg.norm(y_true)/np.linalg.norm(y_true-y_hat))


def compute_srr_db_points(y_true, y_hat):
    """
    Compute minimum L2 error between true set of values ``y_true`` and 
    estimated set of points ``y_hat``.

    Return as 20*log10( ||y_true||_2 / err )

    Parameters
    ----------
    y_true : numpy float array, 1D
        True values.
    y_hat : numpy float array, 1D
        Estimated values.
    """

    min_norm = distance(y_true, y_hat)[0]

    return 20*np.log10(np.linalg.norm(y_true)/min_norm)


def demodulate(y_samp, t_samp, fc, K, oversample_freq=1):
    """
    Demodulate given samples with complex exponential at known frequency. 

    Parameters
    ----------
    y_samp : numpy float array
        Sample values.
    t_samp : numpy float array
        Sample times..
    fc : float
        Center/carrier frequency.
    K : float
        Half of rate of innovation. Sampling kernel of at least 2K (i.e. rate 
        of innovation) will be used, depending on ``oversample_freq``. For 
        pulse stream this is the number of pulses; for bandlimited this is the
        highest frequency component.
    oversample_freq : float
        Oversampling factor of frequency coefficients by increasing the 
        bandwidth of the necessary filter. Bandwidth is increased (in terms of 
        Fourier Series coefficients) from:
            [-K,K] --> [-int(oversample_freq*K),int(oversample_freq*K)]
        Default is 1.
    """
    
    # downmix
    Y_shift = np.fft.fft(y_samp * np.exp(-2j*np.pi*fc*t_samp))
    
    # determine Fourier coefficients to extract around baseband of demodulated signal
    M = int(np.ceil(oversample_freq*K))
    fs_ind_base = np.arange(-M, M+1)

    # apply sampling kernel to dowmixed signal
    Y_lpf = np.zeros(Y_shift.shape, dtype=np.complex)
    Y_lpf[fs_ind_base] = Y_shift[fs_ind_base]
    y_samp_lpf = np.fft.ifft(Y_lpf)
    
    # decimate
    n_samples = len(y_samp)
    n_samples_decimate = len(fs_ind_base)
    skip = n_samples//n_samples_decimate
    sub_idx = np.arange(0, n_samples, skip).astype(np.int)[:n_samples_decimate]
    y_iq = y_samp_lpf[sub_idx]
    t_iq = t_samp[sub_idx]
    
    return y_iq, t_iq, fs_ind_base


def recover_parameters(y_samp, time_vec, K, oversample_freq, center_freq, 
    bandwidth, num_cycles, bwr=-6, cadzow_iter=20):
    """
    y_samp : time domain samples, must be odd number!
    time_vec : corresponding sample locations, must start at 0!
    K : number of diracs to recover
    oversample_freq : oversampling for fourier coefficients
    
    Pulse parameters
    ----------------
    center_freq
    bandwidth
    num_cycles
    
    Optional parameters
    -------------------
    cadzow_iter : number of iterations for fourier coefficient de-noising with cadzow
    
    """

    # demodulate
    y_iq, t_iq, fs_ind_base = demodulate(y_samp, time_vec, 
        center_freq, K=K, oversample_freq=oversample_freq)
    samp_time = t_iq[1]-t_iq[0]
    period = t_iq[-1]-t_iq[0]+samp_time

    # estimate FS coefficients of dirac pulse
    freqs = fs_ind_base/period+center_freq
    H_tot = total_freq_response(freqs, 
        center_freq, bandwidth, num_cycles, bwr)
    fs_coeff_hat = estimate_fourier_coeff(y_iq, t_iq, H=H_tot)

    # denoise and estimate parameters
    fs_coeff_hat_clean = cadzow_denoising(fs_coeff_hat, K, n_iter=cadzow_iter)
    ann_filt = compute_ann_filt(fs_coeff_hat_clean, K)
    tk_hat = estimate_time_param(ann_filt, period)
    ck_hat = estimate_amplitudes(fs_coeff_hat_clean, freqs, tk_hat, period)

    return ck_hat, tk_hat, period


def recover_parameters_gen(y_samp, time_vec, K, oversample_freq, G, freqs, 
    center_freq, max_ini=5, stop_cri='max_iter', seed=0):

    # demodulate
    y_iq, t_iq, fs_ind_base = demodulate(y_samp, time_vec, 
        center_freq, K=K, oversample_freq=oversample_freq)
    samp_time = t_iq[1]-t_iq[0]
    period = t_iq[-1]-t_iq[0]+samp_time

    # gen FRI
    warnings.filterwarnings("ignore")
    fs_coeff_hat, min_error, c_opt, ini = gen_fri(G, y_iq, K, max_ini=max_ini, 
        stop_cri=stop_cri, seed=seed)
    warnings.filterwarnings("default")

    # solve for time instances
    tk_hat = estimate_time_param(c_opt, period)
    ck_hat = estimate_amplitudes(fs_coeff_hat, freqs, tk_hat, period)

    return ck_hat, tk_hat, period


"""
ADAPTED FROM: https://github.com/hanjiepan/FRI_pkg/blob/master/alg_tools_1d.py
"""

def gen_fri_batches(G, a, K, K_batch, fs_ind, period, 
    noise_level=np.inf, max_ini=100, stop_cri='mse', max_iter=50, 
    seed=0, verbose=True):
    
    # initialize
    K_max = K
    K_left = K_max
    n_coeff = G.shape[1]
    n_samp = len(a)
    compute_mse = (stop_cri == 'mse')

    # already found locations and the corresponding sum of sinusoid to append to G
    tk_hat_batch = np.zeros(K)
    G_loc = np.zeros((n_samp,0), dtype=G.dtype)
    fs_coeff_batch = np.zeros(n_coeff, dtype=complex)
    recon_err = []
    ck_hat_batch = []

    # obtain in batches
    while K_left > 0:

        if verbose:
            print("K_left = %d" % K_left)
        
        K = min(K_left, K_batch)
        min_error = float('inf')

        # concatenate already known sum of sinusoids
        G_tild = np.concatenate((G, G_loc), axis=1)
        K0 = G_loc.shape[1]
        E = np.eye(G_tild.shape[1])[:n_coeff,:]

        # pre-compute constants during algo
        gram_G_tild = np.dot(G_tild.conj().T, G_tild)
        gram_G = np.dot(G.conj().T, G)
        beta = linalg.lstsq(G_tild, a)[0]
        Tbeta = Tmtx(np.dot(E, beta), K)

        rhs_c = np.concatenate((np.zeros((K+1)+(n_coeff-K)+(n_coeff+K0)), [1.]))
        rhs_b = np.concatenate((np.dot(G_tild.conj().T, a), np.zeros(n_coeff-K)))

        for ini in range(max_ini):

            rng = np.random.RandomState(seed+ini)
            c = rng.randn(K + 1) + 1j * rng.randn(K + 1)
            c0 = c.copy()

            R_loop = np.dot(Rmtx(c, K, n_coeff), E)

            # first row of c update matrix
            mtx_loop_first_rows = np.hstack((np.zeros((K + 1, K + 1)), 
                Tbeta.conj().T,
                np.zeros((K + 1, G_tild.shape[1])), 
                c0[:, np.newaxis]))

            # last row of c update matrix
            mtx_loop_last_row = np.hstack((c0[np.newaxis].conj(),
                np.zeros((1, R_loop.shape[0]+gram_G_tild.shape[0]+1))))

            for loop in range(max_iter):

                # update c
                mtx_loop = np.vstack((
                    mtx_loop_first_rows,
                    np.hstack((Tbeta, np.zeros((n_coeff-K, n_coeff-K)), -R_loop, 
                        np.zeros((n_coeff-K, 1)))),
                    np.hstack((np.zeros((G_tild.shape[1], K+1)), -R_loop.conj().T, gram_G_tild, 
                        np.zeros((G_tild.shape[1], 1)))),
                    mtx_loop_last_row
                ))
                mtx_loop += mtx_loop.conj().T
                mtx_loop *= 0.5
                c = linalg.solve(mtx_loop, rhs_c)[:K+1]
                
                # update b
                R_loop = np.dot(Rmtx(c, K, n_coeff), E)
                mtx_brecon = np.vstack((
                    np.hstack((gram_G_tild, R_loop.conj().T)),
                    np.hstack((R_loop, np.zeros((n_coeff-K, n_coeff-K))))
                ))
                mtx_brecon += mtx_brecon.conj().T
                mtx_brecon *= 0.5
                b_tild = linalg.solve(mtx_brecon, rhs_b)[:G_tild.shape[1]]

                # check error
                error = linalg.norm(a - np.dot(G_tild, b_tild))

                if error < min_error:
                    min_error = error
                    b_opt = b_tild
                    c_opt = c
                if min_error < noise_level and compute_mse:
                    break
            if min_error < noise_level and compute_mse:
                break

        if verbose:
            srr = 20*np.log10(np.linalg.norm(a)/min_error)
            print("SRR : %f dB" % srr)

        # prepare for next batch
        fs_coeff_batch += b_opt[:n_coeff]
        tk_hat_K = estimate_time_param(c_opt, period)
        res = np.meshgrid(tk_hat_K, fs_ind)
        G_batch = np.exp(-1j*2*np.pi*res[1]*res[0]/period)
        recon_err.append(min_error)

        # save locations and get current amplitude estimates
        tk_hat_batch[K_max-K_left:K_max-K_left+K] = tk_hat_K  
        ck_hat_batch.append(
            estimate_amplitudes(fs_coeff_batch, fs_ind/period, tk_hat_batch[:K_max-K_left+K], period)
            ) 

        G_loc = np.concatenate((G_loc, np.dot(G, G_batch)), axis=1)

        K_left = K_left - K

    
    return ck_hat_batch, tk_hat_batch, fs_coeff_batch, np.array(recon_err)


def Tmtx(data, K):
    """Construct convolution matrix for a filter specified by 'data'
    """
    return linalg.toeplitz(data[K::], data[K::-1])


def Rmtx(data, K, seq_len):
    """A dual convolution matrix of Tmtx. Use the commutativness of a convolution:
    a * b = b * c
    Here seq_len is the INPUT sequence length
    """
    col = np.concatenate(([data[-1]], np.zeros(seq_len - K - 1)))
    row = np.concatenate((data[::-1], np.zeros(seq_len - K - 1)))
    return linalg.toeplitz(col, row)


def gen_fri(G, a, K, noise_level=np.inf, max_ini=100, stop_cri='mse', 
    max_iter=50, seed=None):

    compute_mse = (stop_cri == 'mse')
    M = G.shape[1]
    GtG = np.dot(G.conj().T, G)
    Gt_a = np.dot(G.conj().T, a)

    min_error = float('inf')
    beta = linalg.lstsq(G, a)[0]

    Tbeta = Tmtx(beta, K)
    rhs = np.concatenate((np.zeros(2 * M + 1), [1.]))
    rhs_bl = np.concatenate((Gt_a, np.zeros(M - K)))

    for ini in range(max_ini):
        if seed is None:
            c = np.random.randn(K + 1) + 1j * np.random.randn(K + 1)
        else:
            rng = np.random.RandomState(seed+ini)
            c = rng.randn(K + 1) + 1j * rng.randn(K + 1)
        c0 = c.copy()
        error_seq = np.zeros(max_iter)
        R_loop = Rmtx(c, K, M)

        # first row of mtx_loop
        mtx_loop_first_row = np.hstack((np.zeros((K + 1, K + 1)), Tbeta.conj().T,
                                        np.zeros((K + 1, M)), c0[:, np.newaxis]))
        # last row of mtx_loop
        mtx_loop_last_row = np.hstack((c0[np.newaxis].conj(),
                                       np.zeros((1, 2 * M - K + 1))))

        for loop in range(max_iter):
            mtx_loop = np.vstack((mtx_loop_first_row,
                                  np.hstack((Tbeta, np.zeros((M - K, M - K)),
                                             -R_loop, np.zeros((M - K, 1)))),
                                  np.hstack((np.zeros((M, K + 1)), -R_loop.conj().T,
                                             GtG, np.zeros((M, 1)))),
                                  mtx_loop_last_row
                                  ))

            # matrix should be Hermitian symmetric
            mtx_loop += mtx_loop.conj().T
            mtx_loop *= 0.5
            # mtx_loop = (mtx_loop + mtx_loop.conj().T) / 2.

            c = linalg.solve(mtx_loop, rhs)[:K + 1]

            R_loop = Rmtx(c, K, M)

            mtx_brecon = np.vstack((np.hstack((GtG, R_loop.conj().T)),
                                    np.hstack((R_loop, np.zeros((M - K, M - K))))
                                    ))

            # matrix should be Hermitian symmetric
            mtx_brecon += mtx_brecon.conj().T
            mtx_brecon *= 0.5
            # mtx_brecon = (mtx_brecon + mtx_brecon.conj().T) / 2.

            b_recon = linalg.solve(mtx_brecon, rhs_bl)[:M]

            error_seq[loop] = linalg.norm(a - np.dot(G, b_recon))
            if error_seq[loop] < min_error:
                min_error = error_seq[loop]
                b_opt = b_recon
                c_opt = c
            if compute_mse and min_error < noise_level:
                break
        if compute_mse and min_error < noise_level:
            break

    return b_opt, min_error, c_opt, ini


def distance(x1, x2):
    """
    Given two arrays of numbers x1 and x2, pairs the cells that are the
    closest and provides the pairing matrix index: x1(index(1,:)) should be as
    close as possible to x2(index(2,:)). The function outputs the average of the
    absolute value of the differences abs(x1(index(1,:))-x2(index(2,:))).
    :param x1: vector 1
    :param x2: vector 2
    :return: d: minimum distance between d
             index: the permutation matrix
    """
    x1 = np.reshape(x1, (1, -1), order='F')
    x2 = np.reshape(x2, (1, -1), order='F')
    N1 = x1.size
    N2 = x2.size
    diffmat = np.abs(x1 - np.reshape(x2, (-1, 1), order='F'))
    min_N1_N2 = np.min([N1, N2])
    index = np.zeros((min_N1_N2, 2), dtype=int)
    if min_N1_N2 > 1:
        for k in range(min_N1_N2):
            d2 = np.min(diffmat, axis=0)
            index2 = np.argmin(diffmat, axis=0)
            index1 = np.argmin(d2)
            index2 = index2[index1]
            index[k, :] = [index1, index2]
            diffmat[index2, :] = float('inf')
            diffmat[:, index1] = float('inf')
        # d = np.mean(np.abs(x1[:, index[:, 0]] - x2[:, index[:, 1]]))
        d = np.linalg.norm(np.abs(x1[:, index[:, 0]] - x2[:, index[:, 1]]))
    else:
        d = np.min(diffmat)
        index = np.argmin(diffmat)
        if N1 == 1:
            index = np.array([1, index])
        else:
            index = np.array([index, 1])
    return d, index

