import numpy as np
import scipy.fftpack as sfft

H = 469600      # avg altitude above ground [m]
gsd_y = 3.59    # avg GSD in flight direction [m]

w_px = 6e-6   # pixel size [m]
t_int = 1 / 2000    # integration time [s]
t_s = 1 / 2000  # sample time [s]
f = .65         # effective focal length [m]

p_x = w_px      # across-sensor pixel pitch [m]
p_y = w_px      # pitch between TDI stages [m]
p_y_eff = gsd_y * f / H     # effective along-flight pixel pitch [m]
v_y = gsd_y / t_s   # avg in-flight velocity [m/s]

D = .105        # primary aperture [m]
D_obs = .05   # obscuration mirror diameter [m]
F = f / D       # F number
eps = D_obs / D
k_Nyq_x = 1 / (2 * p_x)
k_Nyq_y = 1 / (2 * p_y_eff)

Aa = 0      # atmospheric absorption [km^-1]
Sa = .25e-3 # atmospheric scattering [km^-1]
z = 6.6     # optical path length [km]

W_RMS = .1    # wavefront error RMS

sigma_jitter = 0.6e-6   # jitter standard deviation [m]

l_diffusion = 60e-6     # diffusion length [m]
l_depl = 6e-6         # depletion width [m]
alpha = 1e6             # absorption coefficient

theta = np.deg2rad(.3)  # angle between direction of flight/orientation of TDI stages [rad]
d_vy = 60e-6            # velocity mismatch between satellite movement/TDI charge transfer in y direction [m/s]
N_TDI = 16              # number of TDI stages

d_smear = v_y * t_int * f / H


def MTF_atmosphere(k_x, k_y, lbda, k_co):
    k_r = np.sqrt(k_x**2 + k_y**2)
    r_0 = 0.185 * lbda**1.2 * (.75 * H**(-1 / 3) * 4.16e-13)**-.6
    turb = np.exp(-3.44 * (lbda * f * k_r / r_0)**(5 / 3) * (1 - .5 * (lbda * f * k_r / D)**(1 / 3)))
    sol = np.exp(-Aa * z - Sa * z * np.minimum((k_r / k_co)**2, 1))
    return turb * sol


def MTF_diffraction(k_x, k_y, k_co):
    k_r = np.sqrt(k_x**2 + k_y**2)
    k_n = k_r / k_co

    def C_simple(k_n):
        if k_n <= (1 - eps) / 2:
            return - np.pi * eps**2
        elif k_n <= (1 + eps) / 2:
            phi = np.arccos((1 + eps**2 - 4 * k_n**2) / (2 * eps))
            return - np.pi * eps**2 + eps * np.sin(phi) + .5 * phi * (1 + eps**2) - (1 - eps**2) * np.arctan((1 + eps) / (1 - eps) * np.tan(.5 * phi))
        else:
            return 0
    C = np.vectorize(C_simple)

    def B_simple(k_n):
        if k_n <= eps:
            return eps**2 * (np.arccos(k_n / eps) - k_n / eps * np.sqrt(1 - (k_n / eps)**2))
        else:
            return 0
    B = np.vectorize(B_simple)

    def A_simple(k_n):
        if k_n < 1:
            return np.arccos(k_n) - k_n * np.sqrt(1 - k_n**2)
        else:
            return 0
    A = np.vectorize(A_simple)

    return 2 / np.pi * (A(k_n) + B(k_n) + C(k_n)) / (1 - eps**2)


def MTF_WFE(k_x, k_y, k_co):
    k_r = np.sqrt(k_x**2 + k_y**2)
    return 1 - (W_RMS / .18)**2 * (1 - 4 * (k_r / k_co - .5)**2)


def MTF_aperture_y(k_x, k_y):
    return abs(np.sinc(k_y * w_px))


def MTF_jitter(k_x, k_y):
    k_r = np.sqrt(k_x**2 + k_y**2)
    return np.exp(-2 * np.pi**2 * sigma_jitter**2 * k_r**2)


def MTF_diffusion(k_x, k_y):
    k_r = np.sqrt(k_x**2 + k_y**2)
    L = l_diffusion / (np.sqrt(1 + (2 * np.pi * l_diffusion)**2 * k_r**2))
    return (1 - np.exp(-alpha * l_depl) / (1 + alpha * L)) / (1 - np.exp(-alpha * l_depl) / (1 + alpha * l_diffusion))


def MTF_TDI_theta(k_x, k_y):
    def TDI_theta_simple(k_x):
        if k_x and theta:
            return abs(np.sin(np.pi * N_TDI * w_px * np.tan(theta) * k_x) / (N_TDI * np.sin(np.pi * w_px * np.tan(theta) * k_x)))
        else:
            return 1

    return np.vectorize(TDI_theta_simple)(k_x)


def MTF_TDI_dv(k_x, k_y):
    def TDI_dvy_simple(k_y):
        if k_y and d_vy:
            return abs(np.sin(np.pi * N_TDI * d_vy * t_int * k_y) / (N_TDI * np.sin(np.pi * d_vy * t_int * k_y)))
        else:
            return 1
    return np.vectorize(TDI_dvy_simple)(k_y)


def MTF_sampling(k_x, k_y): return abs(np.sinc(k_x / (2 * k_Nyq_x)) * np.sinc(k_y / (2 * k_Nyq_y)))


def MTF_smear(k_x, k_y): return abs(np.sinc(d_smear * k_y))


def MTF_aperture_x(k_x, k_y):
    return abs(np.sinc(k_x * w_px))


def MTF_sim(k_x, k_y, lbda):
    k_co = 1 / (lbda * F)
    return MTF_atmosphere(k_x, k_y, lbda, k_co) * MTF_diffraction(k_x, k_y, k_co) * MTF_WFE(k_x, k_y, k_co) * MTF_jitter(k_x, k_y) * MTF_aperture_y(k_x, k_y) * MTF_diffusion(k_x, k_y) * MTF_TDI_dv(k_x, k_y) * MTF_TDI_theta(k_x, k_y)


def MTF_tot(k_x, k_y, lbda):
    return MTF_sim(k_x, k_y, lbda) * MTF_sampling(k_x, k_y) * MTF_smear(k_x, k_y) * MTF_aperture_x(k_x, k_y)

def get_PSF():
    """
    Computes MTF based on imaging chain approach, and Fourier-transforms it into PSF.
    Computations for RGB bands at 665, 560, and 490 nm.
    Note that parameters can only be adjusted directly in the mtf.py file for now.

    Returns
    -------
    PSF : np.ndarray
        filter mask corresponding to PSF, sampled at pixel grid nodes.

    """

    n_samples = 250  # must be even!
    f_max = 2 / w_px

    k_x = np.linspace(1, f_max, int(n_samples / 2))
    k_x = np.append(k_x, k_x[:0:-1])
    k_y = np.linspace(1, f_max, int(n_samples / 2))
    k_y = np.append(k_y, k_y[:0:-1])
    kk_x, kk_y = np.meshgrid(k_x, k_y)

    PSF_2Ds = []
    lambdas = [665e-9, 560e-9, 490e-9]
    for lbda in lambdas:
        MTF_2D = MTF_sim(kk_x, kk_y, lbda)
        PSF_2Ds.append(np.fft.ifftshift(np.fft.ifft2(MTF_2D).real))

    PSF = np.array(PSF_2Ds)
    return PSF


def trim_PSF(psf, size):
    """
    Trims PSF filter to approximately size*size central elements.

    Parameters
    ----------
    psf : np.ndarray
        array containing PSF filter.
    size : int
        preferred size of filter mask.

    Returns
    -------
    trimmed : np.ndarray
        array containing PSF filter, trimmed to approximately size*size.

    """
    wid = int(size / 2)
    _, nx, ny = psf.shape
    trimmed = psf[:, int(nx / 2) - wid + 1: int(nx / 2) + wid + 2, int(ny / 2) - wid + 1: int(ny / 2) + wid + 2]
    trimmed *= (psf.sum(axis=(1, 2)) / trimmed.sum(axis=(1, 2)))[:, None, None]
    return trimmed
