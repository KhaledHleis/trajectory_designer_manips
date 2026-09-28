"""
ui/controls.py — Left-side control panel for interactive designer.
"""
import tkinter as tk
from tkinter import ttk
from typing import Callable
from .theme import Theme as T

_ALGO_SECTIONS = {
    "Lawnmower": {"lawnmower"},
    "Revisit":   {"revisit"},
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
            pass_width=self.pass_width.get(),
            offset_m=self.offset_m.get(),
            overshoot_m=self.overshoot_m.get(),
            start_side=self.start_side.get(),
            reverse=self.reverse_var.get(),
        )

    def set_info(self, text: str):
        self.info_text.set(text)

    def set_spacing(self, text: str):
        self.spacing_text.set(text)

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
        self.reverse_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self, text="Swap start ⇄ end", variable=self.reverse_var,
                       command=self._on_change, bg=T.PANEL_BG, fg=T.TEXT,
                       selectcolor=T.ENTRY_BG, activebackground=T.PANEL_BG,
                       activeforeground=T.ACCENT, font=T.FONT_MONO_MD,
                       relief=tk.FLAT, highlightthickness=0, anchor="w"
                       ).pack(fill=tk.X, padx=12, pady=(4, 0))

        # lawnmower (rectangular)
        lf = self._param_section("lawnmower", "LAWNMOWER")
        self.n_crossings = self._slider("Crossings",           5,   1,  20,   1,    lf)
        self.angle_deg   = self._slider("Incidence angle (°)", 90,  5,  175,  1,    lf)
        self.pass_width  = self._slider("Pass width (×cable)", 0.55, 0.05, 3.0, 0.05, lf)
        self.spacing_text = tk.StringVar(value="—")
        tk.Label(lf, textvariable=self.spacing_text, bg=T.PANEL_BG, fg=T.ACCENT2,
                 font=T.FONT_MONO_MD, justify=tk.LEFT
                 ).pack(anchor="w", padx=12, pady=(4, 2))

        # revisit (C around the cable)
        rf = self._param_section("revisit", "REVISIT")
        self.offset_m    = self._slider("Dist. from cable (m)", 5.0, 0.5, 50.0, 0.5, rf)
        self.overshoot_m = self._slider("Overshoot ends (m)",   5.0, 0.0, 50.0, 0.5, rf)
        self.start_side  = self._slider("Start side (+1 L/-1 R)", 1, -1, 1, 2, rf)

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
