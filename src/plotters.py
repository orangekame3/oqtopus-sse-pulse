# plotters.py
from __future__ import annotations
from typing import Any, Dict, Optional,  Mapping
import numpy as np
import matplotlib.pyplot as plt

class BasePlotter:
    name: str = "base"

    def can_plot(self, payload: Mapping[str, Any]) -> bool:
        """ Check if this plotter can handle the given payload."""
        return False

    def plot(
        self,
        payload: Mapping[str, Any],
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        raise NotImplementedError

class RabiPlotter(BasePlotter):
    name = "rabi"
    REQ_KEYS = {"data"}

    def can_plot(self, payload: Mapping[str, Any]) -> bool:
        return self.REQ_KEYS.issubset(payload.keys())

    def plot(
        self,
        payload: Mapping[str, Any],
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        label: Optional[str] = None,
        cmap: Optional[str] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        data_map: Dict[str, Any] = payload["data"]
        if label is None:
            label = next(iter(data_map.keys()))
        data = data_map[label]
        raw = data["normalized"]
        i = np.asarray(raw["I"], dtype=float)
        # q is present in the payload but not used in Rabi plots; omit to avoid unused-variable warnings
        slots = np.arange(len(i))
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure  # type: ignore
        ax.plot(slots, i, label="I", marker="o", markersize=3, linewidth=1)
        ttl = title or f"Rabi : {label}"
        ax.set_title(ttl)
        ax.set_xlabel("Slot index")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig, ax

class ChevronPlotter(BasePlotter):
    name = "chevron"

    REQ_KEYS = {"detuning_range", "frequencies", "time_range", "chevron_data"}

    def can_plot(self, payload: Mapping[str, Any]) -> bool:
        return self.REQ_KEYS.issubset(payload.keys())

    def plot(
        self,
        payload: Mapping[str, Any],
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        label: Optional[str] = None,
        cmap: Optional[str] = None,
    ) -> tuple[plt.Figure, plt.Axes]:

        detuning = np.asarray(payload["detuning_range"], dtype=float)
        time_range = np.asarray(payload["time_range"], dtype=float)
        freqs: Dict[str, float] = payload["frequencies"]
        data_map: Dict[str, Any] = payload["chevron_data"]

        if label is None:

            label = next(iter(data_map.keys()))

        data = np.asarray(data_map[label])

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.figure # type: ignore

        extent = [
            detuning[0] + freqs[label],
            detuning[-1] + freqs[label],
            time_range[0],
            time_range[-1],
        ]

        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=extent, # type: ignore
            cmap=cmap,
        )

        ttl = title or f"Chevron pattern : {label}"
        ax.set_title(ttl)
        ax.set_xlabel("Drive frequency (GHz)")
        ax.set_ylabel("Time (ns)")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Signal intensity")

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig, ax

class WaveformPlotter(BasePlotter):
    name = "waveform"
    REQ_KEYS = {"data"}

    def can_plot(self, payload: Mapping[str, Any]) -> bool:
        return self.REQ_KEYS.issubset(payload.keys())
    def plot(
		self,
		payload: Mapping[str, Any],
		ax: Optional[plt.Axes] = None,
		title: Optional[str] = None,
		save_path: Optional[str] = None,
		label: Optional[str] = None,
		cmap: Optional[str] = None,
	) -> tuple[plt.Figure, plt.Axes]:
        data_map: Dict[str, Any] = payload["data"]
        if label is None:
            label = next(iter(data_map.keys()))
        data = data_map[label]
        raw = data["raw"]
        i = np.asarray(raw["I"], dtype=float)
        q = np.asarray(raw["Q"], dtype=float)
        slots = np.arange(len(i))
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure # type: ignore
        ax.plot(slots, i, label="I", marker="o", markersize=3, linewidth=1)
        ax.plot(slots, q, label="Q", marker="o", markersize=3, linewidth=1)
        ttl = title or f"IQ Waveform : {label}"
        ax.set_title(ttl)
        ax.set_xlabel("Slot index")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig, ax

PLOTTERS: Dict[str, BasePlotter] = {
    "check_waveform": WaveformPlotter(),
    "check_rabi": RabiPlotter(),
    "chevron_pattern": ChevronPlotter(),
}


def plot_payload(
    payload: Mapping[str, Any],
    program: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
	"""_summary_

	Args:
		payload (Mapping[str, Any]): _description_
		program (Optional[str], optional): _description_. Defaults to None.
		ax (Optional[plt.Axes], optional): _description_. Defaults to None.

	Raises:
		ValueError: _description_

	Returns:
		tuple[plt.Figure, plt.Axes]: _description_
	"""
	if program and program in PLOTTERS:
		plotter = PLOTTERS[program]
		return plotter.plot(payload, ax=ax, **kwargs)

	for plotter in PLOTTERS.values():
		if plotter.can_plot(payload):
			return plotter.plot(payload, ax=ax, **kwargs)

	raise ValueError("No suitable plotter found for the given payload.")
