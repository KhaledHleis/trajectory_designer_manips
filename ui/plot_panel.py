"""
ui/plot_panel.py
----------------
Right-side matplotlib canvas.

Changes vs original:
- draw() now also accepts a `waypoint_mode` flag that changes rendering:
  in waypoint mode points are large numbered markers, not a dense scatter.
- Heading arrows are drawn every N samples so you can verify the angle.
- Cable is drawn with its direction indicated.
"""

import math
import tkinter as tk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from .theme import Theme as T


class PlotPanel(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=T.DARK_BG)
        self._xlim = None
        self._ylim = None
        self._axis_initialised = False
        self._build()

    # ── public API ────────────────────────────────────────────────────────────

    def draw(
        self,
        waypoints: np.ndarray,
        headings: np.ndarray,
        params: dict,
        title: str = "Trajectory",
        waypoint_mode: bool = False,
    ):
        if self._axis_initialised:
            saved_x = self.ax.get_xlim()
            saved_y = self.ax.get_ylim()
        else:
            saved_x = self._xlim
            saved_y = self._ylim

        self.ax.clear()
        self._style_axes()

        if len(waypoints) > 1:
            pts = waypoints[:, [1, 0]]  # (lon, lat) → (x, y)

            if waypoint_mode:
                # Large numbered markers for actual mission waypoints
                self.ax.plot(
                    pts[:, 0], pts[:, 1],
                    color=T.SUBTEXT, linewidth=1.2, linestyle="--",
                    zorder=2, alpha=0.6,
                )
                for idx, (x, y) in enumerate(pts):
                    self.ax.scatter(x, y, color=T.ACCENT, s=80, zorder=5,
                                    edgecolors="white", linewidths=0.6)
                    self.ax.annotate(
                        f" WP{idx+1}", (x, y),
                        color=T.ACCENT, fontsize=7, fontfamily="monospace",
                        zorder=6,
                    )
                # Heading arrows at each waypoint
                if headings is not None and len(headings) == len(pts):
                    for (x, y), h in zip(pts, headings):
                        # Convert nav heading (CW from N) to math angle (CCW from E)
                        rad = math.radians(90.0 - h)
                        arrow_len = max(
                            (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.03,
                            1e-7,
                        )
                        self.ax.annotate(
                            "",
                            xy=(x + math.cos(rad) * arrow_len,
                                y + math.sin(rad) * arrow_len),
                            xytext=(x, y),
                            arrowprops=dict(
                                arrowstyle="->",
                                color=T.ACCENT2,
                                lw=1.2,
                            ),
                            zorder=7,
                        )

            else:
                # Dense trajectory: gradient line + scatter
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                colors = plt.cm.plasma(np.linspace(0.05, 0.92, len(segs)))
                self.ax.add_collection(
                    LineCollection(segs, colors=colors, linewidths=1.5, zorder=2, alpha=0.8)
                )
                self.ax.scatter(
                    pts[:, 0], pts[:, 1],
                    c=np.arange(len(pts)), cmap="plasma",
                    s=6, alpha=0.60, zorder=3, linewidths=0,
                    label=f"Samples ({len(pts)})",
                )

                # Heading arrows every ~5% of the path
                if headings is not None and len(headings) == len(pts) and len(pts) > 4:
                    step = max(1, len(pts) // 20)
                    idx_arr = np.arange(0, len(pts), step)
                    for idx in idx_arr:
                        x, y = pts[idx]
                        h = headings[idx]
                        rad = math.radians(90.0 - h)
                        # arrow scale: ~2% of view range (will be computed after autoscale)
                        self.ax.annotate(
                            "",
                            xy=(x + math.cos(rad) * 1e-5,
                                y + math.sin(rad) * 1e-5),
                            xytext=(x, y),
                            arrowprops=dict(
                                arrowstyle="->",
                                color="#ffffff",
                                lw=0.7,
                                alpha=0.5,
                            ),
                            zorder=7,
                        )

            # start / end
            self.ax.scatter(*pts[0],  color=T.ACCENT,  s=100, zorder=8,
                             marker="o", label="Start",
                             edgecolors="white", linewidths=0.8)
            self.ax.scatter(*pts[-1], color=T.ACCENT2, s=100, zorder=8,
                             marker="s", label="End",
                             edgecolors="white", linewidths=0.8)

        # cable
        cx = [params["cable_p1"][1], params["cable_p2"][1]]
        cy = [params["cable_p1"][0], params["cable_p2"][0]]
        self.ax.plot(cx, cy, color="#ff3333", linewidth=3.5, zorder=7,
                     solid_capstyle="round", label="Cable")
        self.ax.scatter(cx, cy, color="#ff3333", s=60, zorder=8,
                        edgecolors="white", linewidths=0.6)

        # cable direction arrow
        mid_x = (cx[0] + cx[1]) / 2
        mid_y = (cy[0] + cy[1]) / 2
        dx = cx[1] - cx[0]; dy = cy[1] - cy[0]
        norm = math.hypot(dx, dy) + 1e-15
        self.ax.annotate(
            "",
            xy=(mid_x + dx / norm * 2e-5, mid_y + dy / norm * 2e-5),
            xytext=(mid_x, mid_y),
            arrowprops=dict(arrowstyle="->", color="#ff3333", lw=1.5),
            zorder=9,
        )

        self.ax.set_title(title, color=T.ACCENT, fontsize=11, pad=10,
                          fontfamily="monospace")
        self.ax.legend(facecolor=T.PANEL_BG, edgecolor=T.SEP,
                       labelcolor=T.TEXT, fontsize=8, loc="upper right",
                       framealpha=0.85)

        if saved_x is not None and saved_y is not None:
            self.ax.set_xlim(saved_x)
            self.ax.set_ylim(saved_y)
        else:
            self.ax.autoscale_view()
            dx2 = (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.12
            dy2 = (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) * 0.12
            self.ax.set_xlim(self.ax.get_xlim()[0] - dx2, self.ax.get_xlim()[1] + dx2)
            self.ax.set_ylim(self.ax.get_ylim()[0] - dy2, self.ax.get_ylim()[1] + dy2)
            self._axis_initialised = True

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def reset_view(self):
        self._axis_initialised = False
        self._xlim = None
        self._ylim = None

    # ── private ───────────────────────────────────────────────────────────────

    def _build(self):
        self.fig = Figure(facecolor=T.DARK_BG)
        self.ax = self.fig.add_subplot(111, facecolor=T.PANEL_BG)
        self._style_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        tb_frame = tk.Frame(self, bg=T.PANEL_BG)
        tb_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, tb_frame)
        toolbar.config(background=T.PANEL_BG)
        for child in toolbar.winfo_children():
            try:
                child.config(background=T.PANEL_BG, foreground=T.TEXT,
                             highlightbackground=T.PANEL_BG)
            except Exception:
                pass
        toolbar.update()

        self.canvas.mpl_connect("button_release_event", self._capture_limits)
        self.canvas.mpl_connect("scroll_event", self._capture_limits)

    def _style_axes(self):
        self.ax.tick_params(colors=T.SUBTEXT, labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(T.SEP)
        self.ax.set_xlabel("Longitude →", fontsize=8, color=T.SUBTEXT)
        self.ax.set_ylabel("Latitude ↑", fontsize=8, color=T.SUBTEXT)
        self.ax.grid(True, color=T.SEP, linewidth=0.5, linestyle="--", alpha=0.5)

    def _capture_limits(self, _event):
        self._xlim = self.ax.get_xlim()
        self._ylim = self.ax.get_ylim()
