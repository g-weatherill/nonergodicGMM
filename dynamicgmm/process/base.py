"""
"""
from typing import List, Dict, Optional, Union, Tuple
import h5py
import numpy as np
from scipy.integrate import cumtrapz
import dynamicgmm.process.intensity_measures as ims


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


class FourierSpectrum():
    """General class to hold information describing an acceleration
    Fourier Amplitude Spectrum

    Attributes:
        spectrum: The acceleration Fourier amplitude values
        frequency: The corresponding frequencies (Hz)
        periods: The corresponding periods (1.0 / freq)
        units: Units of acceleration
    """
    def __init__(self, fas: np.ndarray, frequency: np.ndarray, units: str):
        """
        """
        self.spectrum = fas
        self.frequency = frequency
        self.units = units
        self.periods = 1.0 / self.frequency


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
            fourier_amplitude_spectrum: Optional[FourierSpectrum] = None
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
        if response_spectrum:
            self._response_spectrum = response_spectrum
        else:
            self._response_spectrum = None

        if fourier_amplitude_spectrum:
            self._fourier_amplitude_spectrum = fourier_amplitude_spectrum
        else:
            self._fourier_amplitude_spectrum = None
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

    @property
    def response_spectrum(self):
        if self._response_spectrum:
            return self._response_spectrum
        else:
            return None

    @property
    def fourier_amplitude_spectrum(self):
        # Returns the Fourier amplitude spectrum class
        if self._fourier_amplitude_spectrum:
            return self._fourier_amplitude_spectrum
        else:
            return None


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

    def __repr__(self):
        return "{:s}|{:s}".format(
            self.event_id,
            ".".join([self.network, self.station, self.location, self.channel])
            )

    def to_hdf(self, grp: Union[str, h5py.Group]):
        """
        """
        raise NotImplementedError


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
