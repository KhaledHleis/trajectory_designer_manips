import csv, os, tkinter as tk
from tkinter import messagebox
import numpy as np
from trajectories import TRAJECTORY_REGISTRY, LawnmowerTrajectory
from .controls import ControlsPanel
from .plot_panel import PlotPanel
from .theme import Theme as T


class WaypointGeneratorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Trajectory Designer v3")
        self.root.configure(bg=T.DARK_BG)
        self.root.geometry("1400x880")
        self.root.minsize(1000, 640)
        self.waypoints = np.empty((0, 2))
        self.headings  = np.empty((0,))
        self._build_layout()
        self._update()

    def _build_layout(self):
        self.controls = ControlsPanel(
            self.root, on_change=self._update,
            on_cable_change=self._reset_and_update,
            algo_names=list(TRAJECTORY_REGISTRY.keys()),
        )
        self.controls.pack(side=tk.LEFT, fill=tk.Y, padx=(10,0), pady=10)
        self.controls.export_btn.config(command=self._export)
        self.plot = PlotPanel(self.root)
        self.plot.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _update(self, *_):
        try:
            p    = self.controls.get_params()
            algo = self.controls.get_algo()
            klass = TRAJECTORY_REGISTRY.get(algo, LawnmowerTrajectory)
            traj  = klass(**p)
            self.waypoints, self.headings = traj.generate_trajectory()
            self.traj = traj
            self.plot.draw(self.waypoints, self.headings, p,
                           title=f"{algo} Waypoints", waypoint_mode=True)
            n   = len(self.waypoints)
            wl  = traj.generate_trajectory_local()[0]
            dist = float(np.sum(np.linalg.norm(np.diff(wl, axis=0), axis=1))) if n > 1 else 0.0
            dur = dist / p["drone_speed"]
            if hasattr(traj, "pass_spacing_m"):
                self.controls.set_spacing(
                    f"Pass spacing  : {traj.pass_spacing_m:.2f} m\n"
                    f"Along cable   : {traj.crossing_spacing_m:.2f} m\n"
                    f"Cable length  : {traj.cable_length_m:.2f} m"
                )
            self.controls.set_info(
                f"Waypoints : {n}\n"
                f"Path      : {dist:.1f} m  (~{dur:.0f} s)\n"
                f"Speed     : {p['drone_speed']:.1f} m/s\n"
                f"Sample Hz : {p['sampling_freq']}\n"
                f"Heading   : nav (0=N, CW)"
            )
        except Exception as exc:
            import traceback
            self.controls.set_info(f"Error:\n{exc}\n{traceback.format_exc()[-200:]}")

    def _reset_and_update(self, *_):
        self.plot.reset_view()
        self._update()

    def _export(self):
        if len(self.waypoints) == 0:
            messagebox.showwarning("Export", "No waypoints to export.")
            return
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(project_root, "output")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "waypoints_export.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            # "heading" (not "heading_nav_deg"): this is the column
            # mSIMU's TrajectoryParser.read_pbp reads, and the one
            # tools/Add_timestamps.py expects to find alongside timestamp.
            w.writerow(["index","latitude","longitude","heading"])
            for i,(lat,lon) in enumerate(self.waypoints):
                w.writerow([i+1,f"{lat:.10f}",f"{lon:.10f}",f"{self.headings[i]:.4f}"])
        from generator.verify import heading_consistency
        # Check on the flown (densified) path: sparse corners can't be
        # finite-differenced meaningfully.
        dense_pts, dense_hdg = self.traj.generate_dense()
        hc = heading_consistency(dense_pts, dense_hdg)
        note = (
            f"\n\nheading vs ground course: {hc['median_deg']:+.4f} deg (median)"
            if hc["consistent"]
            else f"\n\nWARNING: heading disagrees with the ground track these points\n"
                 f"imply by {hc['median_deg']:+.3f} deg. Do NOT feed this to the\n"
                 f"simulator — it will fly every leg crabbed by that amount."
        )
        messagebox.showinfo("Export", f"Saved {len(self.waypoints)} waypoints to:\n{path}{note}")
