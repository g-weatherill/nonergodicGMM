"""
Main code to control the download and storage of ground motion data from the ORFEUS services
ESM and RRSM
"""
import os
import logging
import datetime
from copy import deepcopy
from typing import Dict
import toml
import h5py
import numpy as np
from dynamicgmm.download.esm_fdsn_tools import (
    ESMEventWebService, ESMStationWebservice, ESMWaveformWebService
)
from dynamicgmm.download.rrsm_fdsn_tools import (
    RRSMEventStationWebService,
    RRSMWaveformWebService
)


logging.basicConfig(level=logging.INFO)


def _datetime_to_isoformat(config: Dict) -> Dict:
    """Convert all datetime objects to strings in ISO format
    """
    for key in config:
        if isinstance(config[key], datetime.datetime):
            config[key] = config[key].isoformat()
    return config


class ORFEUSStrongMotionDownloader():
    """Main class to run a download operation from a ORFEUS strong motion services

    Attributes:
        config: Config file as import from toml format
        output_directory: Output directory to store the waveforms
    """
    def __init__(self, config: Dict, run_type: str = "wet"):
        """
        """
        self.config = deepcopy(config)
        self.output_directory = config["output-folder"]
        if os.path.exists(self.output_directory):
            raise OSError("Target output directory %s already exists"
                          % self.output_directory)
        os.mkdir(self.output_directory)
        self.esm_catalogue = None
        self.rrsm_catalogue = None
        assert run_type in ["wet", "damp", "dry"], \
            "Run type must be one of 'wet', 'damp' or 'dry' (%s given)" % run_type
        self.run_type = run_type

    def run(self):
        """
        """
        if "ESM" in self.config:
            # Run the ESM downloader
            logging.info("Running the ESM download process")
            target_dir = os.path.join(self.output_directory,
                                      self.config["ESM"]["esm-output"])
            os.mkdir(target_dir)
            if self.run_type != "dry":
                self.run_esm_download(target_dir)
        if "RRSM" in self.config:
            logging.info("Running the RRSM download process")
            target_dir = os.path.join(self.output_directory,
                                      self.config["RRSM"]["rrsm-output"])
            # os.mkdir(target_dir)
            self.run_rrsm_download(target_dir)
        return

    def run_esm_download(self, target_dir: str):
        """Run the download process for ESM data
        """
        # Get the events from the webservice
        self.config["ESM"]["event"] = _datetime_to_isoformat(
            self.config["ESM"]["event"]
        )
        event_ws = ESMEventWebService(self.config["ESM"]["event"])
        event_ws.get_events()
        self.esm_catalogue = event_ws.catalogue
        if not len(event_ws.event_ids):
            logging.info("No events found in ESM service")
            return
        if self.run_type == "damp":
            return
        for ev_id in event_ws.event_ids:
            # Download the waveforms for a specific event
            wf_config = deepcopy(self.config["ESM"]["waveform"])
            wf_config["eventid"] = ev_id
            logging.info("---- Downloading for event %s" % ev_id)
            event_target_file = os.path.join(
                target_dir,
                ev_id + ".{:s}".format(wf_config["format"])
            )
            wf_downloader = ESMWaveformWebService(wf_config, event_target_file)
            if self.run_type == "wet":
                wf_downloader.download_waveforms()
        return

    def run_rrsm_download(self, target_dir: str):
        """Run the download process for RRSM data
        """
        self.config["RRSM"]["event"] = _datetime_to_isoformat(
            self.config["RRSM"]["event"]
        )
        # Get the events and stations from the webservice
        rrsm_downloader = RRSMEventStationWebService(self.config["RRSM"]["event"])
        rrsm_downloader.get_events_stations()
        if rrsm_downloader.events_stations is None:
            # Download failed - no content
            return
        self.rrsm_catalogue = rrsm_downloader.events_stations
        # Get target directory
        by_station = self.config["RRSM"]["waveform"].get("by-station", False)
        wf_downloader = RRSMWaveformWebService(rrsm_downloader.events_stations,
                                               target_dir,
                                               by_station=by_station)
        rrsm_downloader.to_json(os.path.join(target_dir, "events_stations.json"))
        if self.run_type == "wet":
            wf_downloader.download_waveforms()
        return
