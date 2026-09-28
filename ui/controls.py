"""
ui/controls.py — Left-side control panel for interactive designer.
"""
import tkinter as tk
from tkinter import ttk
from typing import Callable
from .theme import Theme as T

_ALGO_SECTIONS = {
    "Lawnmower":        {"lawnmower"},
    "Zigzag":           {"lawnmower"},
    "Parallel":         {"parallel"},
    "Sinusoidal":       {"sinusoidal"},
    "Spiral":           {"spiral"},
    "Starburst":        {"starburst"},
    "Random Walk":      {"random_walk"},
    "Creeping Line":    {"lawnmower"},
    "Expanding Square": {"expanding"},
    "Random Crossings": {"random_cross"},
    "Waypoints":        {"lawnmower"},
}


class ControlsPanel(tk.Frame):
    def __init__(self, parent, on_change: Callable, on_cable_change: Callable,
                 algo_names: list[str]):
        super().__init__(parent, bg=T.PANEL_BG, width=355)
        self.pack_propagate(False)
        self._on_change = on_change
        self._on_cable_change = on_cable_change
        self._section_frames: dict[str, tk.Frame] = {}
        self._build(algo_names)

    def get_params(self) -> dict:
        return dict(
            cable_p1=(float(self.p1_lat.get()), float(self.p1_lon.get())),
            cable_p2=(float(self.p2_lat.get()), float(self.p2_lon.get())),
            sampling_freq=self.sampling_freq.get(),
            drone_speed=self.drone_speed.get(),
            n_crossings=int(self.n_crossings.get()),
            angle_deg=self.angle_deg.get(),
            turn_sharpness=self.turn_sharpness.get(),
            pass_width=self.pass_width.get(),
            offset_m=self.offset_m.get(),
            amplitude_m=self.amplitude_m.get(),
            phase_offset=self.phase_offset.get(),
            n_loops=self.n_loops.get(),
            start_radius_m=self.start_radius_m.get(),
            end_radius_m=self.end_radius_m.get(),
            angle_offset=self.angle_offset.get(),
            tilt_deg=self.tilt_deg.get(),
            ray_length_m=self.ray_length_m.get(),
            centre_offset_m=self.centre_offset_m.get(),
            angle_spread=self.angle_spread.get(),
            angle_start=self.angle_start.get(),
            step_m=self.step_m.get(),
            lateral_bound_m=self.lateral_bound_m.get(),
            attraction=self.attraction.get(),
            growth_m=self.growth_m.get(),
            angle_min_deg=self.angle_min_deg.get(),
            angle_max_deg=self.angle_max_deg.get(),
            lateral_jitter=self.lateral_jitter.get(),
        )

    def set_info(self, text: str):
        self.info_text.set(text)

    def get_algo(self) -> str:
        return self.algo_var.get()

    def _build(self, algo_names):
        tk.Label(self, text="⬡  TRAJECTORY\n   DESIGNER",
                 bg=T.PANEL_BG, fg=T.ACCENT, font=T.FONT_MONO_XL,
                 justify=tk.LEFT).pack(anchor="w", padx=12, pady=(16, 4))

        self._section_label("ALGORITHM")
        self.algo_var = tk.StringVar(value=algo_names[0])
        af = tk.Frame(self, bg=T.PANEL_BG)
        af.pack(fill=tk.X, padx=12, pady=4)
        cb = ttk.Combobox(af, textvariable=self.algo_var, values=algo_names,
                          state="readonly", width=22)
        cb.pack(side=tk.LEFT)
        cb.bind("<<ComboboxSelected>>", lambda _: self._algo_changed())

        self._section_label("CABLE  (lat / lon)")
        self.p1_lat, self.p1_lon = self._coord_row("P1", 48.49227909392259, -4.50434496199812)
        self.p2_lat, self.p2_lon = self._coord_row("P2", 48.492313209033945, -4.503827049110521)
        tk.Button(self, text="⊙  FIT VIEW", bg=T.ENTRY_BG, fg=T.ACCENT,
                  font=T.FONT_MONO_LG, relief=tk.FLAT, bd=0, pady=4,
                  cursor="hand2", command=self._on_cable_change).pack(fill=tk.X, padx=12, pady=(6,0))

        self._section_label("GLOBAL PARAMETERS")
        self.sampling_freq = self._slider("Sampling freq (Hz)", 50.0, 1.0, 200.0, 1.0)
        self.drone_speed   = self._slider("Drone speed (m/s)",  1.5, 0.1, 10.0, 0.1)

        # lawnmower / zigzag / creeping / waypoints
        lf = self._param_section("lawnmower", "LAWNMOWER / ZIGZAG / …")
        self.n_crossings    = self._slider("Crossings",          5,   1,  20,   1,   lf)
        self.angle_deg      = self._slider("Incidence angle (°)", 90, 5, 175,   1,   lf)
        self.turn_sharpness = self._slider("Turn sharpness",     1.0, 0.1, 5.0, 0.1, lf)
        self.pass_width     = self._slider("Pass width (×cable)", 0.55, 0.05, 3.0, 0.05, lf)

        # parallel
        pf = self._param_section("parallel", "PARALLEL")
        self.offset_m = self._slider("Spacing (m)", 2.0, 0.1, 20.0, 0.1, pf)

        # sinusoidal
        sf = self._param_section("sinusoidal", "SINUSOIDAL")
        self.amplitude_m  = self._slider("Amplitude (m)",  3.0, 0.1, 15.0, 0.1, sf)
        self.phase_offset = self._slider("Phase offset",   0.0, 0.0,  1.0, 0.01, sf)

        # spiral
        spf = self._param_section("spiral", "SPIRAL")
        self.n_loops       = self._slider("Loops",           3.0, 0.5,  10.0, 0.5, spf)
        self.start_radius_m= self._slider("Start radius (m)", 8.0, 0.5, 30.0, 0.5, spf)
        self.end_radius_m  = self._slider("End radius (m)",   1.0, 0.1, 10.0, 0.1, spf)
        self.angle_offset  = self._slider("Angle offset (°)", 0.0, 0.0, 360.0, 1.0, spf)
        self.tilt_deg      = self._slider("Tilt (°)",         30.0, 0.0, 80.0, 1.0, spf)

        # starburst
        stf = self._param_section("starburst", "STARBURST")
        self.ray_length_m   = self._slider("Ray length (m)",  6.0, 0.5, 20.0, 0.5, stf)
        self.centre_offset_m= self._slider("Centre offset(m)",0.0,-5.0,  5.0, 0.1, stf)
        self.angle_spread   = self._slider("Angle spread (°)",180, 20, 360, 5, stf)
        self.angle_start    = self._slider("Start angle (°)", 0.0, 0.0, 360.0, 1.0, stf)

        # random walk
        rwf = self._param_section("random_walk", "RANDOM WALK")
        self.step_m         = self._slider("Step (m)",        0.5, 0.1, 3.0, 0.05, rwf)
        self.lateral_bound_m= self._slider("Lateral bound(m)",4.0, 0.5, 15.0, 0.5, rwf)
        self.attraction     = self._slider("Attraction",      0.3, 0.0,  1.0, 0.05, rwf)

        # expanding square
        esf = self._param_section("expanding", "EXPANDING SQUARE")
        self.growth_m = self._slider("Growth/side (m)", 2.0, 0.1, 10.0, 0.1, esf)

        # random crossings
        rcf = self._param_section("random_cross", "RANDOM CROSSINGS")
        self.angle_min_deg  = self._slider("Angle min (°)",  20, 5, 90, 1, rcf)
        self.angle_max_deg  = self._slider("Angle max (°)", 160, 90, 175, 1, rcf)
        self.lateral_jitter = self._slider("Along jitter",  0.8, 0.0, 1.0, 0.05, rcf)

        self._section_label("INFO")
        self.info_text = tk.StringVar(value="—")
        tk.Label(self, textvariable=self.info_text, bg=T.PANEL_BG, fg=T.SUBTEXT,
                 font=T.FONT_MONO_SM, justify=tk.LEFT, wraplength=315).pack(anchor="w", padx=12, pady=4)

        self.export_btn = tk.Button(self, text="◉  EXPORT  WAYPOINTS",
                                     bg=T.ACCENT, fg=T.DARK_BG, font=T.FONT_MONO_LG,
                                     relief=tk.FLAT, bd=0, pady=8, cursor="hand2")
        self.export_btn.pack(fill=tk.X, padx=12, pady=(16, 8))
        self._update_sections()

    def _algo_changed(self):
        self._update_sections()
        self._on_change()

    def _update_sections(self):
        algo = self.algo_var.get()
        visible = _ALGO_SECTIONS.get(algo, {"lawnmower"})
        for key, frame in self._section_frames.items():
            if key in visible:
                frame.pack(fill=tk.X, padx=0, pady=0)
            else:
                frame.pack_forget()

    def _section_label(self, text):
        tk.Label(self, text=text, bg=T.PANEL_BG, fg=T.ACCENT,
                 font=T.FONT_MONO_LG).pack(anchor="w", padx=12, pady=(14, 2))
        tk.Frame(self, bg=T.SEP, height=1).pack(fill=tk.X, padx=10)

    def _param_section(self, key, title):
        outer = tk.Frame(self, bg=T.PANEL_BG)
        if title:
            tk.Label(outer, text=title, bg=T.PANEL_BG, fg=T.ACCENT,
                     font=T.FONT_MONO_LG).pack(anchor="w", padx=12, pady=(14, 2))
            tk.Frame(outer, bg=T.SEP, height=1).pack(fill=tk.X, padx=10)
        self._section_frames[key] = outer
        return outer

    def _slider(self, label, default, from_, to, resolution, parent=None):
        if parent is None:
            parent = self
        frame = tk.Frame(parent, bg=T.PANEL_BG)
        frame.pack(fill=tk.X, padx=12, pady=2)
        tk.Label(frame, text=label, bg=T.PANEL_BG, fg=T.TEXT, font=T.FONT_MONO_MD,
                 width=22, anchor="w").pack(side=tk.LEFT)
        scale_var = tk.DoubleVar(value=default)
        entry_var = tk.StringVar(value=str(default))
        def slider_moved(val):
            entry_var.set(f"{float(val):.4g}")
            self._on_change()
        scale = tk.Scale(frame, variable=scale_var, from_=from_, to=to,
                         resolution=resolution, orient=tk.HORIZONTAL,
                         bg=T.PANEL_BG, fg=T.TEXT, troughcolor=T.ENTRY_BG,
                         activebackground=T.ACCENT, highlightthickness=0,
                         sliderrelief=tk.FLAT, bd=0, width=11, command=slider_moved)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))
        def entry_commit(event=None):
            try:
                val = max(from_, min(to, float(entry_var.get())))
                scale_var.set(val); entry_var.set(f"{val:.4g}"); self._on_change()
            except ValueError:
                entry_var.set(f"{scale_var.get():.4g}")
        entry = tk.Entry(frame, textvariable=entry_var, bg=T.ENTRY_BG, fg=T.TEXT,
                         insertbackground=T.ACCENT, relief=tk.FLAT, font=T.FONT_MONO_MD,
                         width=7, justify="center")
        entry.pack(side=tk.LEFT)
        entry.bind("<Return>", entry_commit)
        entry.bind("<FocusOut>", entry_commit)
        return scale_var

    def _coord_row(self, label, lat_def, lon_def):
        frame = tk.Frame(self, bg=T.PANEL_BG)
        frame.pack(fill=tk.X, padx=12, pady=2)
        tk.Label(frame, text=label, bg=T.PANEL_BG, fg=T.SUBTEXT,
                 font=T.FONT_MONO_SM, width=6).pack(side=tk.LEFT)
        lat_v = tk.StringVar(value=str(lat_def))
        lon_v = tk.StringVar(value=str(lon_def))
        for v, lbl in ((lat_v, "lat"), (lon_v, "lon")):
            tk.Label(frame, text=lbl, bg=T.PANEL_BG, fg=T.SUBTEXT,
                     font=T.FONT_MONO_SM, width=3).pack(side=tk.LEFT)
            e = tk.Entry(frame, textvariable=v, bg=T.ENTRY_BG, fg=T.TEXT,
                         insertbackground=T.ACCENT, relief=tk.FLAT,
                         font=T.FONT_MONO_MD, bd=4, width=12)
            e.pack(side=tk.LEFT, padx=(0, 3))
            e.bind("<Return>", lambda _: self._on_cable_change())
            e.bind("<FocusOut>", lambda _: self._on_cable_change())
        return lat_v, lon_v
