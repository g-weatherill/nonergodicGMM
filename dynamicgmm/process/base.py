"""
"""
from typing import List, Dict, Optional, Union, Tuple
import logging
import h5py
import numpy as np
from scipy.integrate import cumtrapz
import dynamicgmm.process.intensity_measures as ims
from dynamicgmm.process.konno_ohmachi import KonnoOhmachi
from dynamicgmm.process.sm_utils import convert_accel_units


# Default period set is based on those provied by ESM
DEFAULT_PERIODS = np.array(
    [ 0.01 ,  0.02 ,  0.022,  0.025,  0.029,  0.03 ,  0.032,  0.035,
      0.036,  0.04 ,  0.042,  0.044,  0.045,  0.046,  0.048,  0.05 ,
      0.055,  0.06 ,  0.065,  0.067,  0.07 ,  0.075,  0.08 ,  0.085,
      0.09 ,  0.095,  0.1  ,  0.11 ,  0.12 ,  0.13 ,  0.133,  0.14 ,
      0.15 ,  0.16 ,  0.17 ,  0.18 ,  0.19 ,  0.2  ,  0.22 ,  0.24 ,
      0.25 ,  0.26 ,  0.28 ,  0.29 ,  0.3  ,  0.32 ,  0.34 ,  0.35 ,
      0.36 ,  0.38 ,  0.4  ,  0.42 ,  0.44 ,  0.45 ,  0.46 ,  0.48 ,
      0.5  ,  0.55 ,  0.6  ,  0.65 ,  0.667,  0.7  ,  0.75 ,  0.8  ,
      0.85 ,  0.9  ,  0.95 ,  1.   ,  1.1  ,  1.2  ,  1.3  ,  1.4  ,
      1.5  ,  1.6  ,  1.7  ,  1.8  ,  1.9  ,  2.   ,  2.2  ,  2.4  ,
      2.5  ,  2.6  ,  2.8  ,  3.   ,  3.2  ,  3.4  ,  3.5  ,  3.6  ,
      3.8  ,  4.   ,  4.2  ,  4.4  ,  4.6  ,  4.8  ,  5.   ,  5.5  ,
      6.   ,  6.5  ,  7.   ,  7.5  ,  8.   ,  8.5  ,  9.   ,  9.5  ,
      10.0])


# DEFAULT FREQUENCIES TAKEN FROM BAYLESS & ABRAHAMSON (2019)
DEFAULT_FREQUENCIES = np.array(
[ 0.1     ,   0.102329,   0.104713,   0.107152,   0.109648,
  0.112202,   0.114815,   0.11749 ,   0.120226,   0.123027,
  0.125893,   0.128825,   0.131826,   0.134896,   0.138038,
  0.141254,   0.144544,   0.147911,   0.151356,   0.154882,
  0.158489,   0.162181,   0.165959,   0.169824,   0.17378 ,
  0.177828,   0.18197 ,   0.186209,   0.190546,   0.194984,
  0.199526,   0.204174,   0.20893 ,   0.213796,   0.218776,
  0.223872,   0.229087,   0.234423,   0.239883,   0.245471,
  0.251189,   0.25704 ,   0.263027,   0.269153,   0.275423,
  0.281838,   0.288403,   0.295121,   0.301995,   0.30903 ,
  0.316228,   0.323594,   0.331131,   0.338844,   0.346737,
  0.354813,   0.363078,   0.371535,   0.380189,   0.389045,
  0.398107,   0.40738 ,   0.416869,   0.42658 ,   0.436516,
  0.446684,   0.457088,   0.467735,   0.47863 ,   0.489779,
  0.501187,   0.512861,   0.524807,   0.537032,   0.549541,
  0.562341,   0.57544 ,   0.588844,   0.60256 ,   0.616595,
  0.630957,   0.645654,   0.660693,   0.676083,   0.691831,
  0.707946,   0.724436,   0.74131 ,   0.758578,   0.776247,
  0.794328,   0.81283 ,   0.831764,   0.851138,   0.870964,
  0.891251,   0.912011,   0.933254,   0.954992,   0.977237,
  1.      ,   1.023293,   1.047129,   1.071519,   1.096478,
  1.122018,   1.148153,   1.174897,   1.202264,   1.230269,
  1.258926,   1.28825 ,   1.318257,   1.348963,   1.380384,
  1.412537,   1.44544 ,   1.479108,   1.513561,   1.548816,
  1.584893,   1.62181 ,   1.659587,   1.698244,   1.737801,
  1.778279,   1.819701,   1.862087,   1.905461,   1.949844,
  1.995262,   2.041738,   2.089296,   2.137962,   2.187761,
  2.238721,   2.290868,   2.344229,   2.398833,   2.454709,
  2.511886,   2.570396,   2.630268,   2.691535,   2.754228,
  2.818383,   2.884031,   2.951209,   3.019952,   3.090296,
  3.162278,   3.235937,   3.311311,   3.388441,   3.467368,
  3.548134,   3.63078 ,   3.715352,   3.801893,   3.890451,
  3.981071,   4.073803,   4.168694,   4.265795,   4.365158,
  4.466835,   4.570881,   4.677351,   4.7863  ,   4.897787,
  5.011872,   5.128613,   5.248074,   5.370318,   5.495409,
  5.623413,   5.754399,   5.888436,   6.025596,   6.165949,
  6.309573,   6.456542,   6.606934,   6.760828,   6.918308,
  7.079456,   7.24436 ,   7.413103,   7.585776,   7.762471,
  7.943282,   8.128304,   8.317636,   8.511379,   8.709635,
  8.912507,   9.120107,   9.332541,   9.549923,   9.772372,
 10.      ,  10.23293 ,  10.471284,  10.715192,  10.96478 ,
 11.220183,  11.481534,  11.748973,  12.022642,  12.302684,
 12.589251,  12.882492,  13.182563,  13.489624,  13.80384 ,
 14.12537 ,  14.454392,  14.79108 ,  15.135614,  15.48817 ,
 15.848933,  16.218101,  16.59587 ,  16.98244 ,  17.37801 ,
 17.782793,  18.19701 ,  18.62087 ,  19.05461 ,  19.498443,
 19.952621,  20.41738 ,  20.89296 ,  21.37962 ,  21.877611,
 22.38721 ,  22.908672,  23.442283,  23.988321,  24.54708 ,
 25.11886 ,  25.70395 ,  26.30267 ,  26.91534 ,  27.542291,
 28.183832,  28.84032 ,  29.512094,  30.19952 ,  30.902954,
 31.62278 ,  32.359363,  33.11311 ,  33.884414,  34.673683,
 35.481334,  36.3078  ,  37.153514,  38.018932,  38.90451 ,
 39.81071 ,  40.73802 ,  41.68693 ,  42.65794 ,  43.65157 ,
 44.668342,  45.70881 ,  46.7735  ,  47.862991,  48.97789 ,
 50.11873 ,  51.286144,  52.480751,  53.703182,  54.95409 ,
 56.23413 ,  57.543991,  58.884361,  60.255954,  61.6595  ,
 63.09573 ,  64.565414,  66.06933 ,  67.608283,  69.183082,
 70.79456 ,  72.443572,  74.131004,  75.857734,  77.62469 ,
 79.432792,  81.28303 ,  83.17635 ,  85.11377 ,  87.096321,
 89.1251  ,  91.2011  ,  93.32544 ,  95.49926 ,  97.723724,
 100.])


class ResponseSpectrum():
    """General class to hold information describing an acceleration
    response spectrum

    Attributes:
        spectrum: The response spectral acceleration values
        periods: The spectral periods
        frequency: The corresponding frequency (1.0 / periods)
        units: Units of acceleration
        damping: Damping value (fraction)
    """
    def __init__(
        self,
        spectrum: np.ndarray,
        periods: np.ndarray,
        units: str,
        damping: float = 0.05
    ):
        """
        """
        self.periods = periods
        self.spectrum = spectrum
        self.damping = damping
        self.units = units
        self.frequency = 1.0 / self.periods

    def __iter__(self):
        for per, spec in zip(self.periods, self.spectrum):
            yield per, spec
        return

    @classmethod
    def from_timeseries(
            cls,
            acceleration: np.ndarray,
            time_step: float,
            units: str,
            damping: float = 0.05,
            periods: Optional[np.ndarray] = None,
            pseudo: bool = True,
            ):
        """
        Construct a response spectrum from a given timeseries

        Args:
            acceleration: Acceleration time-series
            time_step: Time-step (s) of the time-series
            units: Units of acceleration
            damping: Fractional damping [0 - 100], default to 0.05 (5 % damping
            periods: Target periods for the response spectrum
            pseudo: Acceleration spectrum refers to the pseudo-acceleration spectrum (True)
                    or the acceleration spectrum (False)
        """
        if periods is None:
            periods = DEFAULT_PERIODS
        resp_spectrum = ims.AccelerationResponseSpectrum(
            acceleration, time_step, periods, damping, units
            )
        s_a = resp_spectrum()[0]
        if pseudo:
            spectrum = s_a["Pseudo-Acceleration"]
        else:
            spectrum = s_a["Acceleration"]
        return cls(spectrum, periods, units, damping)


class FourierSpectrum():
    """General class to hold information describing an acceleration
    Fourier Amplitude Spectrum

    Attributes:
        spectrum: The acceleration Fourier amplitude values
        frequency: The corresponding frequencies (Hz)
        periods: The corresponding periods (1.0 / freq)
        units: Units of acceleration
        smoothing_metadata: If provided, then this retains metadata about the smoothing method,
                            (e.g. Konno & Ohmachi configuration parameters)
    """
    def __init__(
            self,
            fas: np.ndarray,
            frequency: np.ndarray, units: str,
            smoothing_metadata: Optional[Dict] = None
            ):
        """
        """
        self.spectrum = fas
        self.frequency = frequency
        self.units = units
        self.periods = 1.0 / self.frequency
        self.smoothing_metadata = smoothing_metadata

    def __iter__(self):
        for freq, amp in zip(self.frequency, self.spectrum):
            yield freq, amp
        return

    @classmethod
    def from_timeseries(
            cls,
            acceleration: np.ndarray,
            time_step: float,
            units: str,
            frequencies: Optional[np.ndarray] = None,
            konno_ohmachi_kwargs: Optional[Dict] = None
            ):
        """
        Calculate the Fourier Amplitude Spectrum from the time-series.

        Args:
            acceleration: Acceleration time-series
            time_step: Time interval of the time-series (s)
            units: Units of the acceleration time-series
            frequencies: Target frequencies for the FAS, which are determined from
                         log-log interpolation of the raw frequencies and FAS
            konno_ohmachi_kwargs: If smoothing using the konno_ohmachi method then provide
                                  a dictionary with the parameters "bandwidth" and "count".
        """
        freq, fas = ims.get_fourier_spectrum(acceleration, time_step)
        if isinstance(konno_ohmachi_kwargs, dict):
            # Apply Konno & Ohmachi smoothing
            konno_ohmachi = KonnoOhmachi(konno_ohmachi_kwargs)
            fas = konno_ohmachi(fas, freq)
            konno_ohmachi_kwargs["method"] = "Konno & Ohmachi (1998)"
        else:
            konno_ohmachi_kwargs = {"method": None}

        # If a set of target frequences is supplied then interpolate to target frequencies
        if frequencies is not None:
            fas = 10.0 ** np.interp(np.log10(frequencies), np.log10(freq), np.log10(fas),
                                    left=np.nan, right=np.nan)
            freq = frequencies.copy()
        return cls(fas, freq, units, konno_ohmachi_kwargs)


class Waveform():
    """
    Class to store properties of a specific acceleration waveform and return intensity measures

    Attributes:
        id: Unique ID of the waveform as a concatenation of the event ID and the station ID
        network: Network code of the station
        station: Station code
        location: Location code of the station
        channel: Code of the specific channel (e.g. HH, HN, etc.)
        component: Orientation (E: east-west, N: north-south, Z: vertical)
        event_id: ID of the event generating the waveform
        rate: Sampling rate (Hz)
        start_time: Start time (either string or datetime)
        n: length of the time series
        dt: Time-step of the record (s)
        time: Scalar time vector (beginning at 0)
        metadata: Dictionary of any other associated metadata
        units: Units of the acceleration trace (e.g. cm/s/s, m/s/s, g)
    """
    def __init__(
            self,
            event_id: str,
            station: str,
            timeseries: np.ndarray,
            sampling_rate: float,
            start_time: Union[str, np.datetime64],
            units: str = "cm/s/s",
            metadata: Optional[Dict] = None,
            response_spectrum: Optional[ResponseSpectrum] = None,
            fourier_spectrum: Optional[FourierSpectrum] = None
            ):
        """
        Args:
            event_id: ID of the event generating the waveform
            station: Full station code as [network].[station].[location].[channel]
            timeseries: Acceleration trace
        """

        self.id = "|".join([event_id, station])
        self.network, self.station, self.location, self.channel = station.split(".")
        self.component = self.channel[-1]
        self.channel = self.channel[:-1]
        self.event_id = event_id
        self.rate = sampling_rate
        self._acceleration = timeseries
        self.start_time = start_time
        self.n = len(timeseries)
        self.dt = 1.0 / self.rate
        self.time = np.cumsum(self.dt * np.ones(self.n)) - self.dt
        self.metadata = metadata
        self.units = units
        self.response_spectrum = response_spectrum
        self.fourier_spectrum = fourier_spectrum
        self._velocity = None
        self._displacement = None

    def __repr__(self):
        return "{:s} (DT = {:6.4f}s, NPTS = {:g})\n".format(
            self.id,
            self.dt,
            self.n
            )

    @property
    def acceleration(self):
        """Returns the acceleration trace
        """
        return self._acceleration

    @property
    def velocity(self):
        """Returns the velocity trace
        """
        if self._velocity is None:
            self._velocity = self.dt * cumtrapz(self.acceleration, initial=0.0)
        return self._velocity

    @property
    def displacement(self):
        """Returns the displacement trace
        """
        if self._displacement is None:
            self._displacement = self.dt * cumtrapz(self.velocity, initial=0.0)
        return self._displacement

    @property
    def PGA(self):
        """Returns the peak of the absolute acceleration trace (i.e. peak ground acceleration)
        """
        return np.max(np.fabs(self.acceleration))

    @property
    def PGV(self):
        """Returns the peak of the absolute velocity trace (i.e. peak ground velocity)
        """
        return np.max(np.fabs(self.velocity))

    @property
    def PGD(self):
        """Returns the peak of the absolute displacement trace (i.e. peak ground displacement)
        """
        return np.max(np.fabs(self.displacement))

    def get_response_spectrum(
            self,
            periods: np.ndarray,
            damping: float = 0.05,
            pseudo: bool = True
            ) -> ResponseSpectrum:
        """Return the response spectrum. If already an attribute of the class and the
        required periods and damping match those requested then return the current attribute,
        otherwise calculate and cache the output

        Args:
            periods: Target periods of the response spectrum
            damping: Fractional damping [0.0 - 100.0]
        """
        if self.response_spectrum is not None and\
                np.allclose(periods, self.response_spectrum.periods) and\
                np.isclose(damping, self.response_spectrum.damping):
            # The required response spectrum is already an attribute of the object
            return self.response_spectrum
        self.response_spectrum = ResponseSpectrum.from_timeseries(
            self.acceleration, self.d_t, periods, damping, self.units, pseudo)
        return self.response_spectrum

    def get_fourier_spectrum(
            self,
            frequencies: Optional[np.ndarray] = None,
            konno_ohmachi_kwargs: Optional[Dict] = None
            ) -> FourierSpectrum:
        """Returns the Fourier Spectrum of the waveform.

        Args:
            frequencies: (Optional) list of target frequencies (Hz)
            konno_ohmachi_kwargs: Parameters controlling the Konno & Ohmachi smoothing
        """
        self.fourier_spectrum = FourierSpectrum.from_timeseries(
            self.acceleration,
            self.dt,
            self.units,
            frequencies,
            konno_ohmachi_kwargs)
        return self.fourier_spectrum


class Record():
    """Class to hold attributes and retrieve intensity measures for a 3-component record
    of ground motion

    Attributes:
        event_id: ID of the event leading to the ground motion
        network: Network code
        station: Station Code
        location: Location Code
        channel: 2-letter channel code (e.g. HN, HG, HH, etc.)
    """

    def __init__(
            self,
            event_id: str,
            network: str,
            station: str,
            location: str,
            channel: str,
            waveforms: List,
            units: str = "cm/s/s"
            ):
        """
        Args:
            waveforms: 3-Component waveforms as list of Waveform objects
        """
        self.event_id = event_id
        self.network = network
        self.station = station
        self.location = location
        self.channel = channel
        self.h1 = None
        self.h2 = None
        self.v = None
        for wvf in waveforms:
            if wvf.component == "E":
                self.h1 = wvf
            elif wvf.component == "N":
                self.h2 = wvf
            elif wvf.component == "Z":
                self.v = wvf
            else:
                pass
        # If e_w and n_w are still not defined then orientation is non-traditional
        h_components = []
        for wvf in waveforms:
            if wvf.component == "Z":
                continue
            h_components.append(wvf)
        self.h1 = h_components[0]
        self.h2 = h_components[1]
        assert self.h1.units == self.h2.units, \
            "Mismatched units between horizontal components (%s /= %s)" % (self.h1.units,
                                                                           self.h2.units)

    def __repr__(self):
        return "{:s}|{:s}".format(
            self.event_id,
            ".".join([self.network, self.station, self.location, self.channel])
            )

    def to_hdf(self, grp: Union[str, h5py.Group]):
        """
        """
        raise NotImplementedError

    def get_horizontal_acceleration_spectrum(
            self,
            horizontal_component: Union[List, str],
            periods: Optional[np.ndarray] = None,
            damping: float = 0.05,
            output_units: str = "cm/s/s",
            num_proc: int = None) -> Dict:
        """Returns the horizontal response spectra from one or more component definitions

        Args:
            horizontal_component: The choice of horiztonal component(s) either as a single
                                  definition or as a list of definitions
            periods: Target periods for the response spectrum
            damping: Fractional damping [0.0 - 100.0]
            output_units: Units for the required accelerations (if different from waveform
                          units then conversion will be made). PGV remains unchanged.
            num_proc: It a RotDpp measure is requested then this will be performed in parallel
                      and num_proc controls the number of processes to be used.

        """
        if periods is None:
            periods = DEFAULT_PERIODS
        if isinstance(horizontal_component, str):
            horizontal_component = [horizontal_component, ]
        horizontal_spectra = {"periods": periods, "damping": damping, "units": output_units}
        max_gm_theta = None
        for component in horizontal_component:
            if component.lower() == "geometric":
                # Get the geometric mean spectrum
                sax = self.h1.get_response_spectrum(periods, damping)
                say = self.h2.get_response_spectrum(periods, damping)
                horizontal_spectra[component] = {"SA": np.sqrt(sax.spectrum * say.spectrum)}
                horizontal_spectra[component]["PGA"] = np.sqrt(self.h1.PGA * self.h2.PGA)
                horizontal_spectra[component]["PGV"] = np.sqrt(self.h1.PGV * self.h2.PGV)
            elif component.lower() == "envelope":
                # Get the envelope (the larger of the two components for each period)
                sax = self.h1.get_response_spectrum(periods, damping)
                say = self.h2.get_response_spectrum(periods, damping)
                horizontal_spectra[component] = \
                    {"SA": np.max(np.column_stack([sax.spectrum, say.spectrum]), axis=1)}
                horizontal_spectra[component]["PGA"] = max(self.h1.PGA, self.h2.PGA)
                horizontal_spectra[component]["PGV"] = max(self.h1.PGV, self.h2.PGV)
            elif component.lower() == "larger pga":
                # Get the response spectrum of the component with larger PGA
                if self.h1.PGA >= self.h2.PGA:
                    sax = self.h1.get_response_spectrum(periods, damping)
                    horizontal_spectra[component] = {"SA": sax.spectrum}
                    horizontal_spectra[component]["PGA"] = self.h1.PGA
                    horizontal_spectra[component]["PGV"] = self.h1.PGV
                else:
                    say = self.h2.get_response_spectrum(periods, damping)
                    horizontal_spectra[component] = {"SA": say.spectrum}
                    horizontal_spectra[component]["PGA"] = self.h2.PGA
                    horizontal_spectra[component]["PGV"] = self.h2.PGV
            elif component.lower() == "random":
                # Select randomly from the two horiztonal components
                sax = self.h1.get_response_spectrum(periods, damping)
                say = self.h2.get_response_spectrum(periods, damping)
                comps = [(sax.spectrum, self.h1.PGA, self.h1.PGA),
                         (say.spectrum, self.h2.PGA, self.h2.PGV)]
                sel_comp = comps[np.random.randint(0, 2, 1)[0]]
                horizontal_spectra[component] = {
                    "SA": sel_comp[0], "PGA": sel_comp[1], "PGV": sel_comp[2]
                    }
            elif component.lower().startswith("rotd"):
                # Calculate RotDpp where the variation with time-series is cached even
                # if multiple pp values are required
                percentile = float(component.lower().replace("rotd", ""))
                if max_gm_theta is None:
                    rotdpp, max_gm_theta = ims.rotdpp_parallel(
                        self.h1.acceleration, self.h1.dt,
                        self.h2.acceleration, self.h2.dt,
                        periods,
                        percentile,
                        damping,
                        self.h1.units,
                        num_proc)
                    horizontal_spectra[component] = {"SA": rotdpp["Pseudo-Acceleration"]}
                    horizontal_spectra[component]["PGA"] = rotdpp["PGA"]
                    horizontal_spectra[component]["PGV"] = rotdpp["PGV"]
                else:
                    rotdpp = np.percentile(max_gm_theta, percentile, axis=0)
                    horizontal_spectra[component] = {"SA": rotdpp[1:, 0]}
                    horizontal_spectra[component]["PGA"] = rotdpp[0, 0]
                    horizontal_spectra[component]["PGV"] = rotdpp[0, 1]
            else:
                logging.info("Horizontal component definition %s not recognised - skipping"
                             % component)
                pass
        # Convert the acceleration values from the original units to output units
        # (PGV values are left unchanged)
        if output_units != self.h1.units:
            for component in horizontal_spectra:
                if component in ("periods", "damping", "units"):
                    continue
                for key in ["SA", "PGA"]:
                    horizontal_spectra[component][key] = convert_accel_units(
                        horizontal_spectra[component][key], self.h1.units, output_units
                        )
        return horizontal_spectra

    def get_effective_amplitude_spectrum(
            self,
            frequencies: Optional[np.ndarray] = None,
            konno_ohmachi_kwargs: Optional[Dict] = None
            ) -> Dict:
        """Returns the effective amplitude spectrum from the Fourier spectra of the two
        components
        """
        if self.h1.fourier_spectrum is None:
            self.h1.fourier_spectrum = self.h1.get_fourier_spectrum(frequencies,
                                                                    konno_ohmachi_kwargs)
        if self.h2.fourier_spectrum is None:
            self.h2.fourier_spectrum = self.h2.get_fourier_spectrum(frequencies,
                                                                    konno_ohmachi_kwargs)

        eas = np.sqrt(0.5 * (self.h1.fourier_spectrum.spectrum ** 2.0 +
                             self.h2.fourier_spectrum.spectrum ** 2.0))
        return {"frequency": self.h1.fourier_spectrum.frequency, "EAS": eas}


def get_horizontal_spectrum(
        waveform1: Waveform,
        waveform2: Waveform,
        periods: np.ndarray,
        horizontal_component: str,
        damping: float = 0.05,
        units: str = "cm/s/s",
        method: str = 'Nigam-Jennings'
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    """
    if horizontal_component.startswith("RotD"):
        # Get the RotD spectrum
        percentile = float(horizontal_component.replace("RotD", ""))
        sa_rotdpp = ims.rotdpp_parallel(
            waveform1.acceleration, waveform1.dt,
            waveform2.acceleration, waveform2.dt,
            periods, percentile, damping, units, method)
        return sa_rotdpp["Pseudo-Acceleration"], sa_rotdpp["PGA"], sa_rotdpp["PGV"], \
            sa_rotdpp["PGD"], periods

    if waveform1.response_spectrum and\
            (len(periods) == len(waveform1.response_spectrum.periods))\
            and np.allclose(periods, waveform1.response_spectrum.periods):
        # Sa calculated for the same requested periods
        sax = {"Pseudo-Acceleration": waveform1.response_spectrum.spectrum,
               "PGA": waveform1.PGA, "PGV": waveform1.PGV, "PGD": waveform1.PGD}
    else:
        # Calculate the response spectrum for the requested periods
        sax = ims.get_response_spectrum(waveform1.acceleration,
                                        waveform1.dt,
                                        periods,
                                        damping, units, method)[0]
    if waveform2.response_spectrum and\
            (len(periods) == len(waveform2.response_spectrum.periods))\
            and np.allclose(periods, waveform2.response_spectrum.periods):
        # Sa calculated for the same requested periods
        say = {"Pseudo-Acceleration": waveform2.response_spectrum.spectrum,
               "PGA": waveform2.PGA, "PGV": waveform2.PGV, "PGD": waveform2.PGD}
    else:
        # Calculate the response spectrum for the requested periods
        say = ims.get_response_spectrum(waveform2.acceleration,
                                        waveform2.dt,
                                        periods,
                                        damping, units, method)[0]
    sa_horiz = {}
    for i_m in list(sax):
        if horizontal_component == "geometric":
            sa_horiz[i_m] = np.sqrt(sax[i_m] * say[i_m])
        elif horizontal_component == "envelope":
            if i_m in ("PGA", "PGV", "PGD"):
                sa_horiz[i_m] = max(sax[i_m], say[i_m])
            else:
                sa_horiz[i_m] = np.max(np.column_stack([sax[i_m], say[i_m]]), axis=1)
        elif horizontal_component == "larger_pga":
            if sax["PGA"] >= say["PGA"]:
                sa_horiz[i_m] = sax[i_m]
            else:
                sa_horiz[i_m] = say[i_m]
        elif horizontal_component == "arithmetic":
            sa_horiz[i_m] = (sax[i_m] + say[i_m]) / 2.0
        elif horizontal_component == "vectorial":
            sa_horiz[i_m] = np.sqrt(sax[i_m] ** 2.0 + say[i_m] ** 2.0)
        else:
            raise ValueError("Horizontal component %s not recognised" % horizontal_component)
    return sa_horiz["Pseudo-Acceleration"], sa_horiz["PGA"], sa_horiz["PGV"], \
        sa_horiz["PGD"], periods
