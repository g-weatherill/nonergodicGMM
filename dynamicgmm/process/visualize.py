"""
Tools to visualise waveforms, records and intensity measures
"""
import os
from typing import Union, Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from dynamicgmm.process.base import ResponseSpectrum, FourierSpectrum, Waveform, Record
import dynamicgmm.process.intensity_measures as ims


DEFAULT_COLOR_SET = ["k", "tab:blue", "tab:orange", "tab:green", "tab:red",
                     "tab:purple", "tab:brown", "tab:pink", "tab:grey",
                     "gold", "tab:cyan"]


def plot_response_spectra(
        record: Union[Record, Waveform, ResponseSpectrum],
        filename: Optional[str] = None,
        dpi: int = 300,
        title: str = "",
        fig_size: Optional[Tuple] = (8, 8),
        fig_ax: Optional[Tuple] = None,
        xlim: Optional[Tuple] = (),
        ylim: Optional[Tuple] = (),
        logx: bool = True,
        logy: bool = True,
        color_set: Optional[List] = None,
        lw: float = 2.0,
        ):
    """Plots the response spectra for a single waveform or a 3-component set of waveforms

    Args:
        record: The input ground motion as instance of Record, Waveform or ResponseSpectrum
        filename: Path to file for export (extension determines the filetype)
        dpi: Dots-per-inch of export file
        title: Title to add to plot
        fig_size: Tuple indicating figure size (as per matplotlib)
        fig_ax: To add to an existing figure and axis then add the tuple of matplotlib
                (Figure, Axes) objects
        xlim: (Minimum x-, Maximum x-)
        ylim: (Minimum y-, Maximum y-)
        logx: Plot logarithmic x (period) scale (True) or not (False)
        logy: Plot logarithmic y (spectrum) scale (True) or not (False)
        color_set: List of colors for the plots (first three positions)
        lw: Linewidth
    """
    if color_set is None:
        color_set = DEFAULT_COLOR_SET
    if fig_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        fig, ax = fig_ax
    if record.__class__.__name__ == "Record":
        # Plot 3-components
        assert record.h1.response_spectrum is not None, "Missing h1 response spectrum"
        ax.plot(
            record.h1.response_spectrum.periods,
            record.h1.response_spectrum.spectrum,
            "-", color=color_set[0], lw=lw,
            label=(record.h1.channel + record.h1.component)
        )
        assert record.h2.response_spectrum is not None, "Missing h2 response spectrum"
        ax.plot(
            record.h2.response_spectrum.periods,
            record.h2.response_spectrum.spectrum,
            "-", color=color_set[1], lw=lw,
            label=(record.h2.channel + record.h2.component)
        )
        assert record.v.response_spectrum is not None, "Missing vertical response spectrum"
        ax.plot(
            record.v.response_spectrum.periods,
            record.v.response_spectrum.spectrum,
            "-", color=color_set[2], lw=lw,
            label=(record.v.channel + record.v.component)
        )
        ylabel = "Sa (%s)" % record.h1.response_spectrum.units
        ax.legend(fontsize=16)
    elif record.__class__.__name__ == "Waveform":
        # Plot the response spectrum for the Waveform
        assert record.response_spectrum is not None, "Missing response spectrum"
        ax.plot(record.response_spectrum.periods,
                record.response_spectrum.spectrum,
                "-", color="k", lw=lw,
                label=(record.channel + record.component))
        ax.legend(fontsize=16)
        ylabel = "Sa (%s)" % record.response_spectrum.units
    elif record.__class__.__name__ == "ResponseSpectrum":
        # Just plot the response spectrum from a ResponseSpectrum object
        ax.plot(record.periods, record.spectrum, "-", color="k", lw=lw)
        ylabel = "Sa (%s)" % record.units
    else:
        raise ValueError("Unknown type for record")

    ax.grid(which="both")
    ax.set_xlabel("Period (s)", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    if len(xlim):
        ax.set_xlim(*xlim)
    if len(ylim):
        ax.set_ylim(*ylim)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title, fontsize=18)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    if filename:
        filetype = os.path.splitext(filename)[-1][1:]
        plt.savefig(filename, format=filetype, dpi=dpi, bbox_inches="tight")
    return fig, ax


def plot_timeseries(
        record: Union[Record, Waveform],
        quantity: str = "acceleration",
        filename: Optional[str] = None,
        dpi: int = 300,
        title: str = "",
        fig_size: Optional[Tuple] = (12, 8),
        xlim: Optional[Tuple] = (),
        ylim: Optional[Tuple] = (),
        color_set: Optional[List] = None,
        lw: float = 1.5,
        ):
    """Plots the time-series in terms of acceleration, velocity or displacment for a single
    Waveform or 3-Component record

    Args:
        record: Ground motion record as either Record or Waveform object
        quantity: Choose between 'acceleration', 'velocity' or 'displacement'
        dpi: Dots-per-inch of export file
        title: Title to add to plot
        fig_size: Tuple indicating figure size (as per matplotlib)
        xlim: (Minimum x-, Maximum x-)
        ylim: (Minimum y-, Maximum y-)
        color_set: List of colors for the plots (first three positions)
        lw: Linewidth
    """
    assert quantity.lower() in ["acceleration", "velocity", "displacement"]
    if color_set is None:
        color_set = DEFAULT_COLOR_SET
    if record.__class__.__name__ == "Record":
        if quantity.lower() == "displacement":
            y1 = record.h1.displacement.copy()
            y2 = record.h2.displacement.copy()
            yv = record.v.displacement.copy()
            yunits = record.h1.units.split("/")[0]

        elif quantity.lower() == "velocity":
            y1 = record.h1.velocity.copy()
            y2 = record.h2.velocity.copy()
            yv = record.v.velocity.copy()
            yunits = record.h1.units.split("/")[0] + "/s"
        else:
            y1 = record.h1.acceleration.copy()
            y2 = record.h2.acceleration.copy()
            yv = record.v.acceleration.copy()
            yunits = record.h1.units.split("/")[0] + "/s^2"

        # Plot 3 component
        fig, axs = plt.subplots(3, 1, figsize=fig_size, sharex=True)
        axs[0].plot(record.h1.time, y1, "-", color=color_set[0], lw=lw,
                    label=(record.h1.channel + record.h1.component))
        axs[1].plot(record.h2.time, y2, "-", color=color_set[1], lw=lw,
                    label=(record.h2.channel + record.h2.component))
        axs[2].plot(record.v.time, yv, "-", color=color_set[2], lw=lw,
                    label=(record.v.channel + record.v.component))
        for ax in axs:
            ax.grid(which="both")
            if len(xlim):
                ax.set_xlim(*xlim)
            if len(ylim):
                ax.set_ylim(*ylim)
            ax.set_ylabel("%s\n(%s)" % (quantity, yunits), fontsize=16)
            ax.legend(loc="upper right", fontsize=14)
            ax.tick_params(labelsize=12)
        axs[2].set_xlabel("Time (s)", fontsize=16)
        if title:
            fig.suptitle(title, fontsize=18)
    elif record.__class__.__name__ == "Waveform":
        if quantity.lower() == "displacement":
            y = record.displacement.copy()
            yunits = record.units.split("/")[0]
        elif quantity.lower() == "velocity":
            y = record.velocity.copy()
            yunits = record.units.split("/")[0] + "/s"
        else:
            y = record.acceleration.copy()
            yunits = record.units.split("/")[0] + "/s^2"
        fig, axs = plt.subplots(1, 1, figsize=fig_size, sharex=True)
        axs.plot(record.time, y, "-", color=color_set[0], lw=lw,
                 label=(record.channel + record.component))
        axs.set_xlabel("Time (s)", fontsize=16)
        axs.set_ylabel("%s\n(%s)" % (quantity, yunits), fontsize=16)
        axs.grid(which="both")
        if len(xlim):
            axs.set_xlim(*xlim)
        if len(ylim):
            axs.set_ylim(*ylim)
        axs.legend(loc="upper right", fontsize=14)
        axs.tick_params(labelsize=12)
        if title:
            axs.set_title(title, fontsize=18)
    else:
        raise ValueError("Record type not recognised")
    fig.tight_layout()
    if filename:
        filetype = os.path.splitext(filename)[-1][1:]
        plt.savefig(filename, format=filetype, dpi=dpi, bbox_inches="tight")
    return fig, axs
