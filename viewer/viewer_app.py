"""
viewer/viewer_app.py
--------------------
Dataset viewer — browse generated trajectories, see stats, overlay cable.

Features:
  - Load any output directory
  - List all trajectories with type + crossing count
  - Click to plot: trajectory, cable, crossing events, heading arrows
  - Filter by trajectory type
  - Show per-trajectory stats + feature histogram
  - Keyboard nav: left/right arrows to step through trajectories
"""

import json
import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.theme import Theme as T


class ViewerApp:
    def __init__(self, root: tk.Tk, initial_dir: str = None):
        self.root = root
        self.root.title("Trajectory Dataset Viewer")
        self.root.configure(bg=T.DARK_BG)
        self.root.geometry("1500x900")
        self.root.minsize(1100, 700)

        self.manifest_df: pd.DataFrame = None
        self.dataset_dir: str = None
        self.current_idx: int = 0
        self.filtered_ids: list[str] = []

        self._build()
        self.root.bind("<Left>",  lambda _: self._step(-1))
        self.root.bind("<Right>", lambda _: self._step(+1))

        if initial_dir:
            self._load_dir(initial_dir)

    # ── layout ────────────────────────────────────────────────────────────────

    def _build(self):
        # ── left sidebar ──────────────────────────────────────────────────────
        sidebar = tk.Frame(self.root, bg=T.PANEL_BG, width=300)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0), pady=8)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="⬡  DATASET\n   VIEWER",
                 bg=T.PANEL_BG, fg=T.ACCENT, font=T.FONT_MONO_XL,
                 justify=tk.LEFT).pack(anchor="w", padx=10, pady=(12, 6))

        # Load button
        tk.Button(sidebar, text="⊙  LOAD DATASET",
                  bg=T.ENTRY_BG, fg=T.ACCENT, font=T.FONT_MONO_LG,
                  relief=tk.FLAT, cursor="hand2", pady=5,
                  command=self._browse).pack(fill=tk.X, padx=10, pady=(0, 8))

        # Filter by type
        tk.Label(sidebar, text="FILTER BY TYPE", bg=T.PANEL_BG, fg=T.ACCENT,
                 font=T.FONT_MONO_LG).pack(anchor="w", padx=10, pady=(8, 2))
        tk.Frame(sidebar, bg=T.SEP, height=1).pack(fill=tk.X, padx=8)

        self.type_var = tk.StringVar(value="All")
        self.type_cb  = ttk.Combobox(sidebar, textvariable=self.type_var,
                                      state="readonly", width=22)
        self.type_cb.pack(padx=10, pady=4, fill=tk.X)
        self.type_cb.bind("<<ComboboxSelected>>", lambda _: self._apply_filter())

        # Trajectory list
        tk.Label(sidebar, text="TRAJECTORIES", bg=T.PANEL_BG, fg=T.ACCENT,
                 font=T.FONT_MONO_LG).pack(anchor="w", padx=10, pady=(8, 2))
        tk.Frame(sidebar, bg=T.SEP, height=1).pack(fill=tk.X, padx=8)

        list_frame = tk.Frame(sidebar, bg=T.PANEL_BG)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        scrollbar = tk.Scrollbar(list_frame, bg=T.PANEL_BG)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.traj_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            bg=T.ENTRY_BG, fg=T.TEXT,
            selectbackground=T.ACCENT, selectforeground=T.DARK_BG,
            font=T.FONT_MONO_SM, relief=tk.FLAT, bd=0,
            activestyle="none",
        )
        self.traj_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.traj_listbox.yview)
        self.traj_listbox.bind("<<ListboxSelect>>", self._on_list_select)

        # Stats box
        tk.Label(sidebar, text="STATS", bg=T.PANEL_BG, fg=T.ACCENT,
                 font=T.FONT_MONO_LG).pack(anchor="w", padx=10, pady=(8, 2))
        tk.Frame(sidebar, bg=T.SEP, height=1).pack(fill=tk.X, padx=8)
        self.stats_var = tk.StringVar(value="No dataset loaded.")
        tk.Label(sidebar, textvariable=self.stats_var,
                 bg=T.PANEL_BG, fg=T.SUBTEXT, font=T.FONT_MONO_SM,
                 justify=tk.LEFT, wraplength=270).pack(anchor="w", padx=10, pady=4)

        # Nav buttons
        nav = tk.Frame(sidebar, bg=T.PANEL_BG)
        nav.pack(fill=tk.X, padx=10, pady=(4, 10))
        tk.Button(nav, text="◀ PREV", bg=T.ENTRY_BG, fg=T.ACCENT,
                  font=T.FONT_MONO_LG, relief=tk.FLAT, cursor="hand2",
                  command=lambda: self._step(-1)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        tk.Button(nav, text="NEXT ▶", bg=T.ENTRY_BG, fg=T.ACCENT,
                  font=T.FONT_MONO_LG, relief=tk.FLAT, cursor="hand2",
                  command=lambda: self._step(+1)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2,0))

        # ── main plot area ────────────────────────────────────────────────────
        right = tk.Frame(self.root, bg=T.DARK_BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.fig = Figure(facecolor=T.DARK_BG)
        self.ax_main = self.fig.add_axes([0.06, 0.32, 0.60, 0.62], facecolor=T.PANEL_BG)
        self.ax_dist = self.fig.add_axes([0.06, 0.06, 0.28, 0.20], facecolor=T.PANEL_BG)
        self.ax_hdg  = self.fig.add_axes([0.38, 0.06, 0.28, 0.20], facecolor=T.PANEL_BG)
        self.ax_angle= self.fig.add_axes([0.70, 0.06, 0.26, 0.20], facecolor=T.PANEL_BG)
        self.ax_over = self.fig.add_axes([0.70, 0.32, 0.26, 0.62], facecolor=T.PANEL_BG)

        self._style_all_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        tb_frame = tk.Frame(right, bg=T.PANEL_BG)
        tb_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, tb_frame)
        toolbar.config(background=T.PANEL_BG)
        toolbar.update()

    # ── loading ───────────────────────────────────────────────────────────────

    def _browse(self):
        d = filedialog.askdirectory(title="Select dataset output directory")
        if d:
            self._load_dir(d)

    def _load_dir(self, path: str):
        manifest_path = os.path.join(path, "manifest.csv")
        if not os.path.exists(manifest_path):
            self.stats_var.set(f"No manifest.csv found in:\n{path}")
            return

        self.dataset_dir = path
        self.manifest_df = pd.read_csv(manifest_path)
        self._apply_filter(init=True)
        self._update_dataset_stats()

    def _apply_filter(self, init=False):
        if self.manifest_df is None:
            return
        types = ["All"] + sorted(self.manifest_df["type"].unique().tolist())
        self.type_cb["values"] = types

        sel = self.type_var.get()
        if sel == "All":
            sub = self.manifest_df
        else:
            sub = self.manifest_df[self.manifest_df["type"] == sel]

        self.filtered_ids = sub["id"].tolist()
        self.traj_listbox.delete(0, tk.END)
        for tid in self.filtered_ids:
            row = self.manifest_df[self.manifest_df["id"] == tid].iloc[0]
            self.traj_listbox.insert(
                tk.END,
                f"{tid}  {row['type'][:14]:<14}  ×{int(row['n_crossings'])}"
            )

        if self.filtered_ids:
            self.current_idx = 0
            self.traj_listbox.selection_set(0)
            self._plot_trajectory(self.filtered_ids[0])

    def _update_dataset_stats(self):
        df = self.manifest_df
        info_path = os.path.join(self.dataset_dir, "dataset_info.json")
        extra = ""
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            extra = f"Total samples : {info.get('total_samples',0):,}\nTotal crossings: {info.get('total_crossings',0):,}\n"
        type_str = "\n".join(f"  {k}: {v}" for k, v in df["type"].value_counts().items())
        self.stats_var.set(
            f"Dataset: {os.path.basename(self.dataset_dir)}\n"
            f"Trajectories: {len(df)}\n"
            f"{extra}"
            f"Types:\n{type_str}"
        )

    # ── navigation ────────────────────────────────────────────────────────────

    def _step(self, delta: int):
        if not self.filtered_ids:
            return
        self.current_idx = (self.current_idx + delta) % len(self.filtered_ids)
        self.traj_listbox.selection_clear(0, tk.END)
        self.traj_listbox.selection_set(self.current_idx)
        self.traj_listbox.see(self.current_idx)
        self._plot_trajectory(self.filtered_ids[self.current_idx])

    def _on_list_select(self, _event):
        sel = self.traj_listbox.curselection()
        if sel:
            self.current_idx = sel[0]
            self._plot_trajectory(self.filtered_ids[self.current_idx])

    # ── plotting ──────────────────────────────────────────────────────────────

    def _plot_trajectory(self, traj_id: str):
        traj_dir = os.path.join(self.dataset_dir, traj_id)
        meta_path = os.path.join(traj_dir, "metadata.json")
        csv_path  = os.path.join(traj_dir, "trajectory.csv")
        feat_path = os.path.join(traj_dir, "features.npy")

        if not os.path.exists(meta_path):
            return

        with open(meta_path) as f:
            meta = json.load(f)

        df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
        features = np.load(feat_path) if os.path.exists(feat_path) else None

        pts = np.column_stack([df["latitude"], df["longitude"]]) if df is not None else None
        headings = df["heading_deg"].values if df is not None else None
        labels   = df["is_crossing"].values if df is not None else None

        crossings = meta.get("crossings", [])
        cable_p1  = meta["cable_p1"]
        cable_p2  = meta["cable_p2"]

        for ax in [self.ax_main, self.ax_dist, self.ax_hdg, self.ax_angle, self.ax_over]:
            ax.clear()
        self._style_all_axes()

        if pts is not None and len(pts) > 1:
            self._draw_main(pts, headings, labels, crossings, cable_p1, cable_p2, meta)
            self._draw_dist(features, labels)
            self._draw_hdg(headings, labels)
            self._draw_crossing_angles(crossings)
            self._draw_overview(crossings)

        self.fig.suptitle(
            f"{traj_id}  ·  {meta['type']}  ·  {meta['n_samples']} samples  ·  {meta['n_crossings']} crossings",
            color=T.ACCENT, fontfamily="monospace", fontsize=10, y=0.99
        )
        self.canvas.draw_idle()

    def _draw_main(self, pts, headings, labels, crossings, cable_p1, cable_p2, meta):
        ax = self.ax_main
        xy = pts[:, [1, 0]]  # (lon, lat)

        segs   = np.stack([xy[:-1], xy[1:]], axis=1)
        colors = plt.cm.plasma(np.linspace(0.05, 0.92, len(segs)))
        ax.add_collection(LineCollection(segs, colors=colors, linewidths=1.2, alpha=0.8))

        # Mark crossing events
        for ev in crossings:
            ax.scatter(ev["lon"], ev["lat"], color="#00ff88", s=60, zorder=6,
                        marker="x", linewidths=1.5)

        # Heading arrows every ~5%
        step = max(1, len(pts) // 20)
        import math
        for i in range(0, len(pts), step):
            h = headings[i]
            rad = math.radians(90.0 - h)
            scale = max((ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02, 1e-6)
            ax.annotate("", xy=(xy[i,0]+math.cos(rad)*2e-5, xy[i,1]+math.sin(rad)*2e-5),
                        xytext=(xy[i,0], xy[i,1]),
                        arrowprops=dict(arrowstyle="->", color="white", lw=0.5, alpha=0.4),
                        zorder=5)

        ax.scatter(*xy[0],  color=T.ACCENT,  s=80, zorder=8, marker="o", edgecolors="white", lw=0.6)
        ax.scatter(*xy[-1], color=T.ACCENT2, s=80, zorder=8, marker="s", edgecolors="white", lw=0.6)

        # Cable
        cx = [cable_p1[1], cable_p2[1]]; cy = [cable_p1[0], cable_p2[0]]
        ax.plot(cx, cy, color="#ff3333", lw=3, zorder=9, solid_capstyle="round")
        ax.scatter(cx, cy, color="#ff3333", s=50, zorder=10, edgecolors="white", lw=0.5)

        ax.autoscale_view()
        pad_x = (ax.get_xlim()[1]-ax.get_xlim()[0])*0.1 or 1e-5
        pad_y = (ax.get_ylim()[1]-ax.get_ylim()[0])*0.1 or 1e-5
        ax.set_xlim(ax.get_xlim()[0]-pad_x, ax.get_xlim()[1]+pad_x)
        ax.set_ylim(ax.get_ylim()[0]-pad_y, ax.get_ylim()[1]+pad_y)
        ax.set_xlabel("Longitude", color=T.SUBTEXT, fontsize=7)
        ax.set_ylabel("Latitude",  color=T.SUBTEXT, fontsize=7)
        ax.legend(
            handles=[
                plt.Line2D([0],[0], color="#00ff88", marker="x", ls="", label=f"Crossing ({len(crossings)})"),
                plt.Line2D([0],[0], color="#ff3333", lw=2, label="Cable"),
            ],
            facecolor=T.PANEL_BG, edgecolor=T.SEP, labelcolor=T.TEXT, fontsize=7
        )

    def _draw_dist(self, features, labels):
        ax = self.ax_dist
        if features is None:
            return
        dist = features[:, 0]
        t    = np.arange(len(dist))
        ax.fill_between(t, dist, 0, where=(dist >= 0), alpha=0.4, color=T.ACCENT)
        ax.fill_between(t, dist, 0, where=(dist < 0),  alpha=0.4, color=T.ACCENT2)
        ax.plot(t, dist, color=T.TEXT, lw=0.6)
        ax.axhline(0, color="#ff3333", lw=1)
        if labels is not None:
            cx = np.where(labels > 0)[0]
            ax.scatter(cx, dist[cx], color="#00ff88", s=15, zorder=5)
        ax.set_title("dist to cable (m)", color=T.SUBTEXT, fontsize=7, pad=2)
        ax.tick_params(labelsize=6, colors=T.SUBTEXT)

    def _draw_hdg(self, headings, labels):
        ax = self.ax_hdg
        if headings is None:
            return
        t = np.arange(len(headings))
        ax.scatter(t, headings, c=headings, cmap="hsv", s=1, alpha=0.5)
        if labels is not None:
            cx = np.where(labels > 0)[0]
            ax.scatter(cx, headings[cx], color="#00ff88", s=20, zorder=5)
        ax.set_ylim(0, 360)
        ax.set_title("heading (°)", color=T.SUBTEXT, fontsize=7, pad=2)
        ax.tick_params(labelsize=6, colors=T.SUBTEXT)

    def _draw_crossing_angles(self, crossings):
        ax = self.ax_angle
        if not crossings:
            ax.set_title("no crossings", color=T.SUBTEXT, fontsize=7)
            return
        angles = [ev["incidence_deg"] for ev in crossings]
        ax.hist(angles, bins=min(15, len(angles)), color=T.ACCENT,
                edgecolor=T.DARK_BG, alpha=0.8)
        ax.axvline(np.mean(angles), color=T.ACCENT2, lw=1.5, linestyle="--",
                   label=f"mean {np.mean(angles):.1f}°")
        ax.set_title("crossing angles (°)", color=T.SUBTEXT, fontsize=7, pad=2)
        ax.set_xlabel("incidence (°)", fontsize=6, color=T.SUBTEXT)
        ax.legend(fontsize=6, facecolor=T.PANEL_BG, labelcolor=T.TEXT, edgecolor=T.SEP)
        ax.tick_params(labelsize=6, colors=T.SUBTEXT)

    def _draw_overview(self, crossings):
        """Mini polar plot of crossing directions."""
        ax = self.ax_over
        ax.clear()
        ax.set_facecolor(T.PANEL_BG)
        if not crossings:
            return
        angles_rad = [np.radians(90 - ev["heading_deg"]) for ev in crossings]
        for ang in angles_rad:
            import math
            ax.plot([0, math.cos(ang)], [0, math.sin(ang)],
                    color=T.ACCENT, alpha=0.6, lw=1.5)
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.axhline(0, color=T.SEP, lw=0.5); ax.axvline(0, color=T.SEP, lw=0.5)
        ax.plot([0],[0], "o", color=T.ACCENT2, ms=4, zorder=5)
        ax.set_title("crossing directions", color=T.SUBTEXT, fontsize=7, pad=2)
        ax.tick_params(labelsize=5, colors=T.SUBTEXT)
        # N label
        ax.text(0, 1.2, "N", color=T.SUBTEXT, fontsize=7, ha="center", va="center")

    def _style_all_axes(self):
        for ax in [self.ax_main, self.ax_dist, self.ax_hdg, self.ax_angle, self.ax_over]:
            ax.tick_params(colors=T.SUBTEXT, labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(T.SEP)
            ax.set_facecolor(T.PANEL_BG)
            ax.grid(True, color=T.SEP, lw=0.4, linestyle="--", alpha=0.4)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None,
                        help="Path to dataset output directory (optional)")
    args = parser.parse_args()

    root = tk.Tk()
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TCombobox",
                    fieldbackground=T.ENTRY_BG, background=T.ENTRY_BG,
                    foreground=T.TEXT, selectbackground=T.ACCENT,
                    bordercolor=T.SEP, arrowcolor=T.ACCENT)
    ViewerApp(root, initial_dir=args.dataset)
    root.mainloop()


if __name__ == "__main__":
    main()
