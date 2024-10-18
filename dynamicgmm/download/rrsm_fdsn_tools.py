"""
Tools to access and download RRSM data
"""

import os
import subprocess
import logging
import requests
import json
from copy import deepcopy
from typing import Dict, Union
from datetime import datetime
import urllib
import h5py
import numpy as np
import obspy

logging.basicConfig(level=logging.INFO)

VALID_FORMATS = ["xml", "text", "shapefile"]
VALID_ORDERBY = ["time", "time-asc", "magnitude", "magnitude-asc"]
VALID_MAGTYPE = ["any", "mw", "ml", "ms", "md", "mb"]
VALID_CATALOG = ["ESM", "EMSC", "USGS", "ISC", "IGV"]
VALID_LEVELS = ["network", "station", "channel"]
VALID_DATA_FORMATS = ["hdf5", "mseed", "sac", "ascii"]
VALID_DATA_TYPES = ["ACC", "VEL", "DIS", "SA", "SD"]
VALID_PROCESSING_TYPES = ["CV", "MP", "AP"]


FDSN_SPECS = {
    "starttime": (str, ),
    "endtime": (str, ),
    "minlatitude": (float, -90.0, 90.0),
    "maxlatitude": (float, -90.0, 90.0),
    "minlongitude": (float, -180.0, 180.0),
    "maxlongitude": (float, -180.0, 180.0),
    "latitude": (float, -90.0, 90.0),
    "longitude": (float, -180.0, 180.0),
    "minradius": (float, 0.0, np.inf),
    "maxradiue": (float, 0.0, np.inf),
    "mindepth": (float, 0.0, np.inf),
    "maxdepth": (float, 0.0, np.inf),
    "minmagnitude": (float, -np.inf, np.inf),
    "maxmagnitude": (float, -np.inf, np.inf),
    "format": (str, VALID_FORMATS + VALID_DATA_FORMATS),
    "orderby": (str, VALID_ORDERBY),
    "magnitudetype": (str, VALID_MAGTYPE),
    "includeallmagnitudes": (bool, ),
    "includeallorigins": (bool, ),
    "limit": (int, 0, np.inf),
    "eventid": (str, ),
    "catalog": (str, VALID_CATALOG),
    "level": (str, VALID_LEVELS),
    "network": (str,),
    "location": (str,),
    "channel": (str,),
    "processing-type": (str, VALID_PROCESSING_TYPES),
    "data-type": (str, VALID_DATA_TYPES),
    "add-xml": (bool,),
    "add-auxiliary-data": (bool,),

}

# Base URL for the Event Query
EVENT_QUERY_BASE_URL = "https://orfeus-eu.org/odcws/rrsm/1/peak-motion?"


# Base URL for the station query
WAVEFORM_QUERY_BASE_URL = "http://orfeus-eu.org/odcws/rrsm/1/waveform?"


RRSM_EVENT_KEYS = [
    "event-id", "event-time", "event-magnitude", "magnitude-type",
    "event-longitude", "event-latitude", "event-depth", "event-location-reference",
    "event-magnitude-reference"
    ]


RRSM_STATION_KEYS = [
    "network-code", "station-code", "location-code", "station-longitude",
    "station-latitude", "station-elevation", "epicentral-distance", "review-type"
    ]


RRSM_CHANNEL_KEYS = [
    "channel-code", "sensor-azimuth", "sensor-dip", "sensor-depth",
    "low-cut-corner", "high-cut-corner", "pga-value", "pgv-value"
]


class RRSMEventStationWebService():
    """
    """
    def __init__(self, config: Dict):
        """
        """

        self.config = deepcopy(config)
        self.url = self.construct_query_url()
        # self.catalogue = []
        self.event_ids = []
        self.station_ids = []
        self.events_stations = {}
        logging.info(f"Query URL: {self.url}")
        # self._get_events_stations()

    def construct_query_url(self) -> str:
        """Build the URL for the FDSN query of the RRSM service using the config
        """
        query = []

        RRSM_ARGS = ["starttime", "endtime", "minmagnitude"]
        for fdsnkey, arg in self.config.items():
            if fdsnkey not in RRSM_ARGS:
                continue
            if fdsnkey not in FDSN_SPECS:
                raise ValueError(f"{fdsnkey} not a valid FDSN Option")
            fdsn_spec = FDSN_SPECS[fdsnkey]
            if isinstance(arg, str) and ("," in arg):
                vals = arg.split(",")
            else:
                vals = [arg]
            for val in vals:
                assert isinstance(val, fdsn_spec[0]), \
                    "FDSN option %s has invalid type" % fdsnkey
                if isinstance(val, str) and len(fdsn_spec) > 1:
                    assert val in fdsn_spec[1], \
                        "Categorical value {:s} for FDSN option {:s} not in valid list: {:s}".format(
                        val, fdsnkey, str(fdsn_spec[1])
                    )
                if (isinstance(val, float) or isinstance(val, int)) and (len(fdsn_spec) == 3):
                    # Verify within upper limits
                    assert (val >= fdsn_spec[1]) & (val <= fdsn_spec[2])
            query.append("{:s}={:s}".format(fdsnkey, str(arg)))
        return EVENT_QUERY_BASE_URL + "&".join(query)

    def get_events_stations(self):
        """
        """
        req = urllib.request.Request(self.url)
        raw_data = urllib.request.urlopen(req)
        if raw_data.code == 204:
            logging.info("No data in RRSM for query")
            return
        # Extract the event data in json format
        data = json.load(raw_data)
        min_lon = self.config.get("minlongitude", -180.0)
        max_lon = self.config.get("maxlongitude", 180.0)
        min_lat = self.config.get("minlatitude", -90.0)
        max_lat = self.config.get("maxlatitude", 90.0)
        # Re-organise data per event and per station
        for ev_data in data:
            ev_id = ev_data["event-id"]
            in_bbox = (ev_data["event-longitude"] >= min_lon) &\
                (ev_data["event-longitude"] <= max_lon) &\
                (ev_data["event-latitude"] >= min_lat) &\
                (ev_data["event-latitude"] <= max_lat)
            if not in_bbox:
                continue
            if ev_id not in self.event_ids:
                self.event_ids.append(ev_id)
            if ev_id not in self.events_stations:
                # Add the event metadata
                self.events_stations[ev_id] = dict([(key, ev_data[key])
                                                    for key in RRSM_EVENT_KEYS])

                self.events_stations[ev_id]["stations"] = {}
            else:
                ntwstn = "{:s}.{:s}".format(ev_data["network-code"], ev_data["station-code"])
                if ntwstn not in self.station_ids:
                    self.station_ids.append(ntwstn)
                self.events_stations[ev_id]["stations"][ntwstn] = {}

                for key in RRSM_STATION_KEYS:
                    self.events_stations[ev_id]["stations"][ntwstn][key] = ev_data[key]
                # Get the channel information
                self.events_stations[ev_id]["stations"][ntwstn]["channels"] = [
                    deepcopy(channel) for channel in ev_data["sensor-channels"]
                    ]
        return

    def to_json(self, fname):
        """Save the events-stations information to json

        Args:
            fname: path to json file
        """
        with open(fname, "w") as f:
            json.dump(self.events_stations, f)
        return


class RRSMWaveformWebService():
    """
    """
    def __init__(self, events_stations: Dict, storage_directory: str,
                 by_station: bool = False):
        """
        """
        self.events_stations = events_stations
        self.event_ids = list(self.events_stations)
        if os.path.exists(storage_directory):
            raise OSError("Storage directory %s already exists" % storage_directory)
        self.store = storage_directory
        os.mkdir(storage_directory)
        self.by_station = by_station

    def download_waveforms(self):
        """
        """
        for event_id in self.event_ids:
            event_url = WAVEFORM_QUERY_BASE_URL + f"eventid={event_id}"
            if self.by_station:
                for sid in self.events_stations[event_id]["stations"]:
                    ntw, stn = sid.split(".")
                    event_url += "&network={:s}&station={:s}".format(ntw, stn)

                    fname = os.path.join(
                        self.store,
                        event_id + ".{:s}.{:s}".format(ntw, stn) + ".mseed")
                    logging.info("Requesting data for event: %s, station: %s" %
                                 (event_id, sid))
                    urllib.request.urlretrieve(event_url, fname)
                    logging.info("---- Successful - stored in %s" % fname)
            else:
                fname = os.path.join(self.store, event_id + ".mseed")
                logging.info("Requesting data for event: %s" % event_id)
                urllib.request.urlretrieve(event_url, fname)
            logging.info("---- Successful - stored in %s" % fname)
        return

    def drop_store(self):
        """
        """
        logging.info("Removing waveform data store %s" % self.store)
        os.system("rm -r %s" % self.store)
        return
