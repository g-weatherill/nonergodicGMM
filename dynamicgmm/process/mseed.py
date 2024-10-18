"""
Extract ground motion intensity measured from miniseed files
"""
import os
import logging
import json
from typing import Union, List, Dict, Optional, Tuple
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz
from scipy.constants import g
import obspy
from dynamicgmm.process.base import (
    ResponseSpectrum, FourierSpectrum, Waveform, Record, get_horizontal_spectrum
    )
import dynamicgmm.process.intensity_measures as ims
from dynamicgmm.process.sm_utils import convert_accel_units


def trace_to_waveform(
        trace: obspy.core.trace.Trace,
        event_id: str = "-",
        units: str = "m/s/s",
        ) -> Waveform:
    """Converts an Obspy trace object into a current waveform object (converting the
    acceleration values from the input units to cm/s/s in the process)
    """
    trace_id = ".".join([trace.stats.network, trace.stats.station,
                         trace.stats.location, trace.stats.channel])
    return Waveform(
        event_id,
        trace_id,
        timeseries=convert_accel_units(trace.data.copy(), units, "cm/s/s"),
        sampling_rate=trace.stats.sampling_rate,
        start_time=np.datetime64(trace.stats.starttime),
        units="cm/s/s"
    )


class MSEEDEventHandler():
    """
    """
    def __init__(
            self,
            fname: str,
            units: str = "m/s/s",
            source: str = "RRSM",
            event_id: Optional[str] = None,
            metadata: Optional[Dict] = None,
            ):
        """
        """
        self.fname = fname
        self.source = source
        self.units = units
        self.records = {}
        self.metadata = metadata
        self.event_id = event_id if event_id else \
            os.path.split(self.fname)[-1].rstrip(".mseed")

    def event_station_metadata_from_json(self, fname):
        """Load events-stations information from json
        """
        with open(fname, "r") as f:
            events_stations = json.load(f)
        if self.event_id not in events_stations:
            raise ValueError("Event ID %s not found in events_stations data" % self.event_id)
        self.metadata = events_stations[self.event_id]
        return

    def parse(self):
        """
        """
        traces = obspy.read(self.fname, format="MSEED")
        ntr = len(traces)
        records = {}
        # Sort the traces to the respective components per record
        for trace_i in traces:
            trace = trace_to_waveform(trace_i, event_id=self.event_id)
            # Generate a general ID from the network, station and location
            rec_id = ".".join([trace.network, trace.station, trace.location, trace.channel])
            if rec_id not in records:
                records[rec_id] = {} # dict([(trace.channel + trace.component, trace)])
            
            channel_id = trace.channel + trace.component
            if channel_id not in records[rec_id]:
                records[rec_id][channel_id] = trace
            else:
                continue
        # Verify that all three components belong to the same record and then convert these
        # to a Record object
        for rec_id in records:
            print(rec_id, records[rec_id])
            trace_set = []
            for channel_id, trace in records[rec_id].items():
                if not len(trace_set):
                    network = trace.network
                    station = trace.station
                    location = trace.location
                    channel = trace.channel
                else:
                    check = (network == trace.network) and (station == trace.station) and \
                        (location == trace.location) and (channel == trace.channel)

                    assert check, "Component mismatch {:s} \= {:s}".format(
                        ".".join([network, station, location, channel]),
                        str(trace)
                        )
                trace_set.append(trace)
            records[rec_id] = Record(
                    event_id=self.event_id,
                    network=trace_set[0].network,
                    station=trace_set[0].station,
                    location=trace_set[0].location,
                    channel=trace_set[0].channel,
                    waveforms=trace_set,
                    units=self.units)
        # Limit the outputs to just those given in the events_stations dictionary
        self.records = {}
        if self.metadata is not None:
            # Get event metadata
            event_metadata = {}
            for key, val in self.metadata.items():
                if key.startswith("event") or key.startswith("magnitude"):
                    event_metadata[key] = val
            print(event_metadata)
            for key, vals in self.metadata["stations"].items():
                # Get station metadata
                sel_rec_id = ".".join([vals["network-code"],
                                       vals["station-code"],
                                       vals["location-code"]])
                metadata = deepcopy(event_metadata)
                for stn_key in ["network-code", "station-code", "location-code",
                                "station-longitude", "station-latitude",
                                "station-elevation", "epicentral-distance",
                                "review-type"]:
                    metadata[stn_key] = vals[stn_key]
                print(sel_rec_id, metadata)
                for rec_id in records:
                    if sel_rec_id == rec_id[:-3]:
                        if sel_rec_id not in self.records:
                            self.records[sel_rec_id] = {rec_id[-2:]: records[rec_id]}
                        else:
                            self.records[sel_rec_id][rec_id[-2:]] = records[rec_id]
                        self.records[sel_rec_id][rec_id[-2:]].metadata = metadata
                    else:
                        continue
        else:
            self.records = records
        return

#    def parse(self, alternate=True):
#        """
#        """
#        traces = obspy.read(self.fname, format="MSEED")
#        ntr = len(traces)
#        posns = list(range(0, ntr, 6)) if alternate else list(range(0, ntr, 3))
#        records = {}
#        for pos in posns:
#            trace1 = trace_to_waveform(traces[pos])
#            pos2 = (pos + 2) if alternate else (pos + 1)
#            pos3 = (pos + 4) if alternate else (pos + 2)
#            trace2 = trace_to_waveform(traces[pos2])
#            trace3 = trace_to_waveform(traces[pos3])
#            print(pos, pos2, pos3, str(trace1), str(trace2), str(trace3))
#            # Define a unique record ID as a join of network, station and location
#            rec_id = ".".join([trace1.network, trace1.station, trace1.location])
#            # Verify that the three components share the same identifiers
#            assert trace1.network == trace2.network == trace3.network, \
#                "{:g} Mismatched network codes: {:s} {:s} {:s}".format(
#                    pos, trace1.network, trace2.network, trace3.network
#                    )
#            assert trace1.station == trace2.station == trace3.station, \
#                "{:g} Mismatched station codes: {:s} {:s} {:s}".format(
#                    pos, trace1.station, trace2.station, trace3.station
#                    )
#            assert trace1.location == trace2.location == trace3.location, \
#                "{:g} Mismatched location codes: {:s} {:s} {:s}".format(
#                    pos, trace1.location, trace2.location, trace3.location
#                    )
#            assert trace1.channel == trace2.channel == trace3.channel, \
#                "{:g} Mismatched channel codes: {:s} {:s} {:s}".format(
#                    pos, trace1.channel, trace2.channel, trace3.channel
#                    )
#
#            if rec_id not in records:
#                # First channel from the network-station-location
#                records[rec_id] = {
#                    trace1.channel: Record(
#                        event_id="",
#                        network=trace1.network,
#                        station=trace1.station,
#                        location=trace1.location,
#                        channel=trace1.channel,
#                        waveforms=[trace1, trace2, trace3],
#                        units=self.units)
#                }
#            else:
#                # Another channel available for the network-station-location
#                records[rec_id][trace1.channel] = Record(
#                    event_id="",
#                    network=trace1.network,
#                    station=trace1.station,
#                    location=trace1.location,
#                    channel=trace1.channel,
#                    waveforms=[trace1, trace2, trace3],
#                    units=self.units
#                    )
#        # Limit the outputs to just those given in the events_stations dictionary
#        self.records = {}
#        if self.events_stations is not None:
#            for ev_id, ev_data in self.event_stations.items():
#                # Get event metadata
#                event_metadata = {}
#                for key, val in ev_data.items():
#                    if key.startswith("event") or key.startswith("magnitude"):
#                        event_metadata[key] = ev_data[val]
#
#                for key, vals in ev_data["stations"]:
#                    # Get station metadata
#                    sel_rec_id = "{:s}|{:s}".format(
#                        ev_id,
#                        ".".join([vals["network-code"],
#                                  vals["station-code"],
#                                  vals["location-code"]])
#                    )
#                    if sel_rec_id in records:
#                        metadata = deepcopy(event_metadata)
#                        for stn_key in ["network-code", "station-code", "location-code",
#                                        "station-longitude", "station-latitude",
#                                        "station-elevation", "epicentral-distance",
#                                        "review-type"]:
#                            metadata[key] = vals[stn_key]
#                        self.records[sel_rec_id] = records[sel_rec_id]
#                        self.records[sel_rec_id].event_id = ev_id
#                        self.records[sel_rec_id].metadata = metadata
#        else:
#            self.records = records
#        return
