"""
Response spectrum enhancements
"""
from multiprocessing import Pool, cpu_count
import timeit
import numpy as np
import numba
#import numexpr as nx
from math import sqrt
#from numba import jit
#from sm_utils import get_time_vector, _save_image, nextpow2, convert_accel_units
from dynamicgmm.process.sm_utils import (
    get_time_vector, convert_accel_units, nextpow2, get_velocity_displacement)

#class AccelerationResponseSpectrum(object):
#    """
#    Base Class to implement a response spectrum calculation
#    """
#    def __init__(self, acceleration: np.ndarray, time_step: float,
#                 periods: np.ndarray, damping: float = 0.05,
#                 units: str = "cm/s/s"):
#        """
#        Setup the response spectrum calculator
#        :param numpy.ndarray time_hist:
#            Acceleration time history [Time, Acceleration]
#        :param numpy.ndarray periods:
#            Spectral periods (s) for calculation
#        :param float damping:
#            Fractional coefficient of damping
#        :param str units:
#            Units of the acceleration time history {"g", "m/s", "cm/s/s"}
#
#        """
#        self.periods = periods
#        self.num_per = len(periods)
#        self.acceleration = convert_accel_units(acceleration, units)
#        self.damping = damping
#        self.d_t = time_step
#        self.velocity, self.displacement = get_velocity_displacement(
#            self.d_t, self.acceleration)
#        self.num_steps = len(self.acceleration)
#        self.omega = (2. * np.pi) / self.periods
#        self.response_spectrum = None
#
#    def __call__(self):
#        """
#        Evaluates the response spectrum
#        :returns:
#            Response Spectrum - Dictionary containing all response spectrum
#                                data
#                'Time' - Time (s)
#                'Acceleration' - Acceleration Response Spectrum (cm/s/s)
#                'Velocity' - Velocity Response Spectrum (cm/s)
#                'Displacement' - Displacement Response Spectrum (cm)
#                'Pseudo-Velocity' - Pseudo-Velocity Response Spectrum (cm/s)
#                'Pseudo-Acceleration' - Pseudo-Acceleration Response Spectrum
#                                       (cm/s/s)
#
#            Time Series - Dictionary containing all time-series data
#                'Time' - Time (s)
#                'Acceleration' - Acceleration time series (cm/s/s)
#                'Velocity' - Velocity time series (cm/s)
#                'Displacement' - Displacement time series (cm)
#                'PGA' - Peak ground acceleration (cm/s/s)
#                'PGV' - Peak ground velocity (cm/s)
#                'PGD' - Peak ground displacement (cm)
#
#            accel - Acceleration response of Single Degree of Freedom Oscillator
#            vel - Velocity response of Single Degree of Freedom Oscillator
#            disp - Displacement response of Single Degree of Freedom Oscillator
#        """
#        raise NotImplementedError("Cannot call Base Response Spectrum")
#
#
#class NigamJennings(AccelerationResponseSpectrum):
#    """
#    Evaluate the response spectrum using the algorithm of Nigam & Jennings
#    (1969)
#    In general this is faster than the classical Newmark-Beta method, and
#    can provide estimates of the spectra at frequencies higher than that
#    of the sampling frequency.
#    """
#
#    def __call__(self):
#        """
#        Define the response spectrum
#        """
#        omega = (2. * np.pi) / self.periods
#        omega2 = omega ** 2.
#        omega3 = omega ** 3.
#        omega_d = omega * sqrt(1.0 - (self.damping ** 2.))
#        const = {  # noqa
#            'f1': (2.0 * self.damping) / (omega3 * self.d_t),
#            'f2': 1.0 / omega2,
#            'f3': self.damping * omega,
#            'f4': 1.0 / omega_d
#        }
#        const['f5'] = const['f3'] * const['f4']
#        const['f6'] = 2.0 * const['f3']
#        const['e'] = np.exp(-const['f3'] * self.d_t)
#        const['s'] = np.sin(omega_d * self.d_t)
#        const['c'] = np.cos(omega_d * self.d_t)
#        const['g1'] = const['e'] * const['s']
#        const['g2'] = const['e'] * const['c']
#        const['h1'] = (omega_d * const['g2']) - (const['f3'] * const['g1'])
#        const['h2'] = (omega_d * const['g1']) + (const['f3'] * const['g2'])
#        x_a, x_v, x_d = self._get_time_series(const, omega2)
#
#        self.response_spectrum = {
#            'Period': self.periods,
#            'Acceleration': np.max(np.fabs(x_a), axis=0),
#            'Velocity': np.max(np.fabs(x_v), axis=0),
#            'Displacement': np.max(np.fabs(x_d), axis=0)}
#        self.response_spectrum['Pseudo-Velocity'] =  omega * \
#            self.response_spectrum['Displacement']
#        self.response_spectrum['Pseudo-Acceleration'] =  (omega ** 2.) * \
#            self.response_spectrum['Displacement']
#        time_series = {
#            'Time-Step': self.d_t,
#            'Acceleration': self.acceleration,
#            'Velocity': self.velocity,
#            'Displacement': self.displacement,
#            'PGA': np.max(np.fabs(self.acceleration)),
#            'PGV': np.max(np.fabs(self.velocity)),
#            'PGD': np.max(np.fabs(self.displacement))}
#
#        return self.response_spectrum, time_series, x_a, x_v, x_d
#        
#    def _get_time_series(self, const, omega2):
#        """
#        Calculates the acceleration, velocity and displacement time series for
#        the SDOF oscillator
#        :param dict const:
#            Constants of the algorithm
#        :param np.ndarray omega2:
#            Square of the oscillator period
#        :returns:
#            x_a = Acceleration time series
#            x_v = Velocity time series
#            x_d = Displacement time series
#        """
#        x_d = np.zeros([self.num_steps - 1, self.num_per], dtype=float)
#        x_v = np.zeros_like(x_d)
#        x_a = np.zeros_like(x_d)
#        
#        for k in range(0, self.num_steps - 1):
#            yval = k - 1
#            dug = self.acceleration[k + 1] - self.acceleration[k]
#            z_1 = const['f2'] * dug
#            z_2 = const['f2'] * self.acceleration[k]
#            z_3 = const['f1'] * dug
#            z_4 = z_1 / self.d_t
#            if k == 0:
#                b_val = z_2 - z_3
#                a_val = (const['f5'] * b_val) + (const['f4'] * z_4)
#            else:    
#                b_val = x_d[k - 1, :] + z_2 - z_3
#                a_val = (const['f4'] * x_v[k - 1, :]) +\
#                    (const['f5'] * b_val) + (const['f4'] * z_4)
#
#            x_d[k, :] = (a_val * const['g1']) + (b_val * const['g2']) +\
#                z_3 - z_2 - z_1
#            x_v[k, :] = (a_val * const['h1']) - (b_val * const['h2']) - z_4
#            x_a[k, :] = (-const['f6'] * x_v[k, :]) - (omega2 * x_d[k, :])
#        return x_a, x_v, x_d


@numba.njit
def get_time_series_fast(acceleration, periods, damping, d_t):
    """Implements Nigam and Jennings algoritm in a numbafied manner
    """
    # Instantiate the SDOF response array as a 3D array [len(accel), len(periods), 3]
    # where the three dimensions are acceleration, velocity and displacement respectively
    num_steps = acceleration.shape[0]
    num_per = periods.shape[0]
    x = np.zeros((num_steps, num_per, 3), dtype=numba.float64)
    
    omega = (2. * np.pi) / periods
    omega2 = omega ** 2.
    omega3 = omega ** 3.
    omega_d = omega * np.sqrt(1.0 - (damping ** 2.))
    
    f1 = (2.0 * damping) / (omega3 * d_t)
    f2 = 1.0 / omega2
    f3 = damping * omega
    f4 = 1.0 / omega_d
    f5 = f3 * f4
    f6 = 2.0 * f3
    e = np.exp(-f3 * d_t)
    s = np.sin(omega_d * d_t)
    c = np.cos(omega_d * d_t)
    g1 = e * s
    g2 = e * c
    h1 = (omega_d * g2) - (f3 * g1)
    h2 = (omega_d * g1) + (f3 * g2)
    for k in range(num_steps - 1):
        dug = acceleration[k + 1] - acceleration[k]
        z_1 = f2 * dug
        z_2 = f2 * acceleration[k]
        z_3 = f1 * dug
        z_4 = z_1 / d_t
        if not k:
            b_val = z_2 - z_3
            a_val = (f5 * b_val) + (f4 * z_4)
        else:
            b_val = x[k - 1, :, 2] + z_2 - z_3
            a_val = f4 * x[k - 1, :, 1] + (f5 * b_val + f4 * z_4)
        x[k, :, 2] = (a_val * g1) + (b_val * g2) + z_3 - z_2 - z_1
        x[k, :, 1] = (a_val * h1) - (b_val * h2) - z_4
        x[k, :, 0] = (-f6 * x[k, :, 1]) - (omega2 * x[k, :, 2])
    return x, omega


class AccelerationResponseSpectrum():
    """
    Evaluate the response spectrum using the algorithm of Nigam & Jennings
    (1969)
    In general this is faster than the classical Newmark-Beta method, and
    can provide estimates of the spectra at frequencies higher than that
    of the sampling frequency.
    """

    def __init__(
        self, 
        acceleration: np.ndarray,
        time_step: float,
        periods: np.ndarray,
        damping: float = 0.05,
        units: str = "cm/s/s"
    ):
        """
        Setup the response spectrum calculator
        :param numpy.ndarray time_hist:
            Acceleration time history [Time, Acceleration]
        :param numpy.ndarray periods:
            Spectral periods (s) for calculation
        :param float damping:
            Fractional coefficient of damping
        :param str units:
            Units of the acceleration time history {"g", "m/s", "cm/s/s"}

        """
        self.periods = periods
        self.num_per = len(periods)
        self.acceleration = convert_accel_units(acceleration, units)
        self.damping = damping
        self.d_t = time_step
        self.velocity, self.displacement = get_velocity_displacement(
            self.d_t, self.acceleration)
        self.num_steps = len(self.acceleration)
        self.omega = (2. * np.pi) / self.periods
        self.response_spectrum = None

    def __call__(self):
        """
        Define the response spectrum
        """
        x, omega = get_time_series_fast(self.acceleration, self.periods,
                                        self.damping, self.d_t)
        x_max = np.max(np.fabs(x), axis=0)
        self.response_spectrum = {
            'Period': self.periods,
            'Acceleration': x_max[:,0],
            'Velocity': x_max[:, 1],
            'Displacement': x_max[:, 2]}
        self.response_spectrum['Pseudo-Velocity'] =  omega * \
            self.response_spectrum['Displacement']
        self.response_spectrum['Pseudo-Acceleration'] =  (omega ** 2.) * \
            self.response_spectrum['Displacement']
        time_series = {
            'Time-Step': self.d_t,
            'Acceleration': self.acceleration,
            'Velocity': self.velocity,
            'Displacement': self.displacement,
            'PGA': np.max(np.fabs(self.acceleration)),
            'PGV': np.max(np.fabs(self.velocity)),
            'PGD': np.max(np.fabs(self.displacement))}

        return self.response_spectrum, time_series, x[:, :, 0], x[:, :, 1], x[:, :, 2]


def get_response_spectrum(acceleration, time_step, periods, damping=0.05, 
                          units="cm/s/s", method="Nigam-Jennings"):
    """
    Returns the elastic response spectrum of the acceleration time series.
    :param numpy.ndarray acceleration:
        Acceleration time series
    :param float time_step:
        Time step of acceleration time series in s
    :param numpy.ndarray periods:
        List of periods for calculation of the response spectrum
    :param float damping:
        Fractional coefficient of damping
    :param str units:
        Units of the INPUT ground motion records
    :param str method:
        Choice of method for calculation of the response spectrum
        - "Newmark-Beta"
        - "Nigam-Jennings"
    :returns:
        Outputs from :class: smtk.response_spectrum.BaseResponseSpectrum
    """
    response_spec = AccelerationResponseSpectrum(acceleration,
                                                 time_step,
                                                 periods, 
                                                 damping,
                                                 units)
    spectrum, time_series, accel, vel, disp = response_spec()
    spectrum["PGA"] = time_series["PGA"]
    spectrum["PGV"] = time_series["PGV"]
    spectrum["PGD"] = time_series["PGD"]
    return spectrum, time_series, accel, vel, disp


def equalise_series(series_x, series_y):
    """
    For two time series from the same record but of different length
    cuts both records down to the length of the shortest record
    N.B. This assumes that the start times and the time-steps of the record
    are the same - if not then this may introduce biases into the record
    :param numpy.ndarray series_x:
         X Time series
    :param numpy.ndarray series_y:
         Y Time series
    """
    n_x = len(series_x)
    n_y = len(series_y)
    if n_x > n_y:
        return series_x[:n_y], series_y
    elif n_y > n_x:
        return series_x, series_y[:n_x]
    else:
        return series_x, series_y


def _get_basic_response_spectrum_for_angle(theta, acceleration_x, acceleration_y,
                                           time_step_x, time_step_y, periods,
                                           damping=0.05, units="cm/s/s"):
    """

    theta: Angle of rotation in radians
    """
    arot = acceleration_x * np.cos(theta) +\
        acceleration_y * np.sin(theta)
    spectrum, time_series, _, _, _ = AccelerationResponseSpectrum(
        arot, time_step_x, periods, damping, units)()
    spectrum["PGA"] = time_series["PGA"]
    spectrum["PGV"] = time_series["PGV"]
    spectrum["PGD"] = time_series["PGD"]
    return spectrum
    #return get_response_spectrum(arot, time_step_x, periods, damping, units, method)[0]


def rotdpp_parallel(acceleration_x, time_step_x, acceleration_y, time_step_y, periods,
           percentile, damping=0.05, units="cm/s/s", num_proc=None):
    """
    Returns the rotationally dependent spectrum RotDpp as defined by Boore
    (2010) - using parallelisation by theta to speed up calculations
    """
    if np.fabs(time_step_x - time_step_y) > 1E-10:
        raise ValueError("Record pair must have the same time-step!")
    acceleration_x, acceleration_y = equalise_series(acceleration_x,
                                                     acceleration_y)
    theta_set = np.radians(np.arange(0., 180., 1.))
    max_a_theta = np.zeros([len(theta_set), len(periods) + 1])
    max_v_theta = np.zeros_like(max_a_theta)
    max_d_theta = np.zeros_like(max_a_theta)
    if num_proc is None:
        num_proc = cpu_count()
    pool = Pool(num_proc)
    A = pool.starmap(
        _get_basic_response_spectrum_for_angle,
        [(theta, acceleration_x, acceleration_y, time_step_x, time_step_y,
          periods, damping, units) for theta in theta_set]
        )
    max_gm_theta = np.zeros([len(theta_set), len(periods) + 1, 3])
    for i, A_i in enumerate(A):
        max_gm_theta[i, :, 0] = np.hstack([A_i["PGA"], A_i["Pseudo-Acceleration"]])
        max_gm_theta[i, :, 1] = np.hstack([A_i["PGV"], A_i["Pseudo-Velocity"]])
        max_gm_theta[i, :, 2] = np.hstack([A_i["PGD"], A_i["Displacement"]])
    pool.close()
    gm_rotdpp = np.percentile(max_gm_theta, percentile, axis=0)
    return {"PGA": gm_rotdpp[0, 0], "PGV": gm_rotdpp[0, 1], 
            "PGD": gm_rotdpp[0, 2], "Pseudo-Acceleration": gm_rotdpp[1:, 0],
            "Pseudo-Velocity": gm_rotdpp[1:, 1], "Displacement": gm_rotdpp[1:, 2]}, max_gm_theta
