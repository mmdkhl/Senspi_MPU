# -*- coding: utf-8 -*-
# =============================================================================
# LEGACY REFERENCE — do not run or import from this file.
#
# This is the original monolithic script (DigitalTwin_V8.py) preserved for
# historical reference only.  All functionality has been refactored into the
# structured package at:
#
#     src/opensees_model_updating/
#
# To understand how a specific feature was originally implemented, search this
# file and compare it to the corresponding module in the package.
# The entry point for the current codebase is:
#
#     python run.py          (from the repository root)
#
# See docs/developer_guide.md for the full architecture reference.
# =============================================================================
"""
3D 1-bay, N-story aluminum frame in OpenSeesPy with automatic modal calibration
Units: N, m, sec

Main features
-------------
1. GUI for structural model, analysis settings, and calibration settings.
2. Experimental modal data is loaded automatically from:
       input/experimental_modal_data.json
3. Original modal analysis is run first.
4. If calibration is enabled, the script updates:
       - Young's modulus E
       - self-weight floor masses
   to reduce mismatch with experimental frequencies and optionally mode shapes.
5. Exports:
       - original inputs
       - calibrated inputs
       - loaded experimental data
       - original/calibrated modal data
       - comparison report
       - a summary figure
6. Runs transient analysis using the final calibrated model.
7. Added options:
       - per-story/per-column weak-axis or strong-axis column orientation for X shaking
       - per-floor additional masses at slab center and/or four slab corners

Important note on mode shapes
-----------------------------
Mode shapes are normalized here using max(abs(phi)) = 1.
That is usually a good idea because eigenvectors are only defined up to
an arbitrary scale and sign.
"""
# %matplotlib qt   # Uncomment only when running in Spyder/IPython if needed.

import os
import math
import time
import json
import copy
import traceback
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import opsvis as opsv
import openseespy.opensees as ops

import tkinter as tk
from tkinter import ttk, messagebox

from scipy.optimize import least_squares


# =============================================================================
# USER-CONFIGURABLE FIXED FILE PATHS
# =============================================================================
EXPERIMENTAL_MODAL_JSON = "input/experimental_modal_data.json"


# =============================================================================
# NUMERIC FORMATTING
# =============================================================================
ROUND_DECIMALS = 3


def r3(x):
    try:
        return round(float(x), ROUND_DECIMALS)
    except Exception:
        return x


def sci3(x):
    try:
        return f"{float(x):.3e}"
    except Exception:
        return str(x)


def deep_round(obj, decimals=ROUND_DECIMALS):
    if isinstance(obj, dict):
        return {k: deep_round(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_round(v, decimals) for v in obj]
    if isinstance(obj, tuple):
        return [deep_round(v, decimals) for v in obj]
    if isinstance(obj, np.ndarray):
        return [deep_round(v, decimals) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), decimals)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


# =============================================================================
# HELPERS
# =============================================================================
def story_column_layout_to_text(layout, nStory):
    parts = []
    for story in range(1, nStory + 1):
        cols = layout.get(story, [1, 2, 3, 4])
        parts.append(f"{story}:{','.join(str(c) for c in cols)}")
    return "; ".join(parts)


# =============================================================================
# HELPER: COLUMN ORIENTATION TEXT FOR REPORTS
# =============================================================================
def column_orientation_layout_to_text(layout, nStory):
    parts = []
    for story in range(1, nStory + 1):
        col_text = []
        for c in (1, 2, 3, 4):
            col_text.append(f"{c}={layout.get(story, {}).get(c, 'weak')}")
        parts.append(f"{story}:{','.join(col_text)}")
    return "; ".join(parts)


# =============================================================================
# HELPER: ADDITIONAL MASS PLACEMENT TEXT FOR REPORTS
# =============================================================================
def additional_masses_to_text(layout, nStory):
    parts = []
    for story in range(1, nStory + 1):
        vals = layout.get(story, [0.0, 0.0, 0.0, 0.0, 0.0])
        parts.append(f"{story}:{','.join(str(r3(v)) for v in vals)}")
    return "; ".join(parts)


def normalize_mode_maxabs(vec):
    v = np.asarray(vec, dtype=float)
    m = np.max(np.abs(v))
    if m <= 0.0:
        return v.copy()
    return v / m


def align_mode_sign(phi_num, phi_exp):
    a = np.asarray(phi_num, dtype=float)
    b = np.asarray(phi_exp, dtype=float)
    if np.dot(a, b) < 0.0:
        return -a
    return a


def safe_percent_error(pred, target):
    if abs(target) < 1e-14:
        return np.nan
    return 100.0 * (pred - target) / target


def write_json(path, data, round_values=True):
    data_out = deep_round(data) if round_values else data
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_out, f, indent=2)


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_experimental_modal_data(json_path, nStory, require_mode_shapes=True):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Experimental modal data file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "frequencies_hz" not in data:
        raise ValueError("Experimental JSON must contain 'frequencies_hz'.")

    freqs = data["frequencies_hz"]
    if not isinstance(freqs, list) or len(freqs) < 1:
        raise ValueError("'frequencies_hz' must be a list with at least 1 value.")

    if "mode_shapes_ux" not in data:
        if require_mode_shapes:
            raise ValueError(
                "Experimental JSON must contain 'mode_shapes_ux' when 'Use mode shapes in calibration' is checked. "
                "Uncheck that option to calibrate using frequencies only."
            )
        return {
            "frequencies_hz": np.array([float(x) for x in freqs], dtype=float),
            "mode_shapes_ux": [],
            "raw_data": data,
            "mode_shapes_available": False,
        }

    mode_shapes = data["mode_shapes_ux"]
    if not isinstance(mode_shapes, dict):
        if require_mode_shapes:
            raise ValueError("'mode_shapes_ux' must be a dictionary.")
        return {
            "frequencies_hz": np.array([float(x) for x in freqs], dtype=float),
            "mode_shapes_ux": [],
            "raw_data": data,
            "mode_shapes_available": False,
        }

    modes = []
    for mode_id in range(1, len(freqs) + 1):
        key = str(mode_id)
        if key not in mode_shapes:
            if require_mode_shapes:
                raise ValueError(f"'mode_shapes_ux' must contain mode '{key}'.")
            return {
                "frequencies_hz": np.array([float(x) for x in freqs], dtype=float),
                "mode_shapes_ux": [],
                "raw_data": data,
                "mode_shapes_available": False,
            }

        vals = mode_shapes[key]
        if len(vals) != nStory:
            if require_mode_shapes:
                raise ValueError(
                    f"Experimental mode shape {key} in '{json_path}' has {len(vals)} story values, "
                    f"but the current model has {nStory} stories. "
                    "Update input/experimental_modal_data.json so each mode shape has one value per story, "
                    "or uncheck 'Use mode shapes in calibration' to calibrate using frequencies only."
                )
            return {
                "frequencies_hz": np.array([float(x) for x in freqs], dtype=float),
                "mode_shapes_ux": [],
                "raw_data": data,
                "mode_shapes_available": False,
            }

        modes.append(np.array([float(x) for x in vals], dtype=float))

    return {
        "frequencies_hz": np.array([float(x) for x in freqs], dtype=float),
        "mode_shapes_ux": modes,
        "raw_data": data,
        "mode_shapes_available": True,
    }

# =============================================================================
# GUI
# =============================================================================
def launch_input_window():
    result = {}
    calibration_state = {
        "available": False,
        "input_signature": None,
        "calibrated_params": None,
        "calibrated_modal": None,
        "uncalibrated_response": None,
    }

    root = tk.Tk()
    root.title("OpenSeesPy Model + Automatic Modal Calibration")
    root.geometry("1240x860")
    root.minsize(1200, 1000)
    root.configure(bg="#eef2f7")

    style = ttk.Style()
    style.theme_use("clam")

    style.configure("Main.TFrame", background="#eef2f7")
    style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"),
                    background="#eef2f7", foreground="#1f2d3d")
    style.configure("Subtitle.TLabel", font=("Segoe UI", 10),
                    background="#eef2f7", foreground="#4a5568")
    style.configure("Section.TLabelframe", background="white",
                    borderwidth=1, relief="solid")
    style.configure("Section.TLabelframe.Label", font=("Segoe UI", 11, "bold"),
                    foreground="#1f2d3d", background="white")
    style.configure("FieldLabel.TLabel", font=("Segoe UI", 10),
                    background="white", foreground="#222222")
    style.configure("Hint.TLabel", font=("Segoe UI", 9),
                    background="white", foreground="#667085")
    style.configure("Small.TLabel", font=("Segoe UI", 9),
                    background="white", foreground="#222222")
    style.configure("Run.TButton", font=("Segoe UI", 10, "bold"), padding=8)
    style.configure("Cancel.TButton", font=("Segoe UI", 10), padding=8)


    style.configure("Removed.TCombobox", fieldbackground="#ffe5e5")
    style.map("Removed.TCombobox",
              fieldbackground=[("disabled", "#ffe5e5")])
    
    outer = ttk.Frame(root, style="Main.TFrame", padding=14)
    outer.pack(fill="both", expand=True)

    ttk.Label(
        outer,
        text="3D Frame Model Inputs + Automatic Modal Calibration",
        style="Title.TLabel"
    ).pack(anchor="w", pady=(0, 2))

    ttk.Label(
        outer,
        text=(
            "Experimental modal targets are loaded automatically from "
            f"'{EXPERIMENTAL_MODAL_JSON}'. "
            "Story-dependent height, mass, columns-present checkboxes, column orientation, and mass-placement rows update from the Number of Stories field."
        ),
        style="Subtitle.TLabel"
    ).pack(anchor="w", pady=(0, 12))

    notebook = ttk.Notebook(outer)
    notebook.pack(fill="both", expand=True)

    tab1 = ttk.Frame(notebook)
    tab2 = ttk.Frame(notebook)
    tab3 = ttk.Frame(notebook)

    notebook.add(tab1, text="Basic Model")
    notebook.add(tab2, text="Mass + Analysis")
    notebook.add(tab3, text="Calibration")

    def make_scrollable_tab(parent):
        canvas = tk.Canvas(parent, bg="#eef2f7", highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        body = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=body, anchor="nw")

        def _configure_body(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _configure_canvas(event):
            canvas.itemconfigure(window_id, width=event.width)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        body.bind("<Configure>", _configure_body)
        canvas.bind("<Configure>", _configure_canvas)
        canvas.bind("<Enter>", lambda event: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda event: canvas.unbind_all("<MouseWheel>"))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return body

    tab1_body = make_scrollable_tab(tab1)
    tab2_body = make_scrollable_tab(tab2)

    entries = {}
    bool_vars = {}

    story_height_vars = {}
    floor_mass_vars = {}
    column_present_vars = {}
    column_orient_vars = {}
    column_orient_widgets = {}
    additional_mass_vars = {}

    nStory_var = tk.IntVar(value=3)
    story_rebuild_job = {"id": None}
    story_rebuild_in_progress = {"value": False}
    story_tables_ready = {"value": False}
    story_table_count = {"value": None}

    def add_field(parent, row, label_text, key, default_value, hint_text, width=28):
        ttk.Label(parent, text=label_text, style="FieldLabel.TLabel").grid(
            row=row, column=0, sticky="w", padx=(10, 8), pady=(8, 1)
        )
        ent = ttk.Entry(parent, width=width, font=("Segoe UI", 10))
        ent.insert(0, str(default_value))
        ent.grid(row=row, column=1, sticky="ew", padx=(0, 10), pady=(8, 1))
        ttk.Label(parent, text=hint_text, style="Hint.TLabel",
                  wraplength=480, justify="left").grid(
            row=row + 1, column=0, columnspan=2, sticky="w",
            padx=(10, 10), pady=(0, 6)
        )
        entries[key] = ent
        return row + 2

    def add_check(parent, row, label_text, key, default_value, hint_text):
        ttk.Label(parent, text=label_text, style="FieldLabel.TLabel").grid(
            row=row, column=0, sticky="w", padx=(10, 8), pady=(8, 1)
        )
        var = tk.BooleanVar(value=default_value)
        ttk.Checkbutton(parent, variable=var).grid(
            row=row, column=1, sticky="w", padx=(0, 10), pady=(8, 1)
        )
        ttk.Label(parent, text=hint_text, style="Hint.TLabel",
                  wraplength=480, justify="left").grid(
            row=row + 1, column=0, columnspan=2, sticky="w",
            padx=(10, 10), pady=(0, 6)
        )
        bool_vars[key] = var
        return row + 2

    def get_int_story_count():
        try:
            n = int(nStory_var.get())
        except Exception:
            n = 3
        return max(1, min(20, n))

    def set_entry_text(entry, text):
        entry.delete(0, tk.END)
        entry.insert(0, text)

    def clear_frame(frame):
        for child in frame.winfo_children():
            child.destroy()

    def _read_plan_ratio():
        """Returns Lx/Ly ratio from the geometry boxes for drawing only."""
        try:
            lx = float(entries.get("Lx").get())
            ly = float(entries.get("Ly").get())
            if lx > 0.0 and ly > 0.0:
                return lx / ly
        except Exception:
            pass
        return 0.245 / 0.23

    def draw_corner_legend(canvas):
        """Draws a clear plan-view sketch for C1-C4 mass-placement labels."""
        canvas.delete("all")
        # Use actual displayed canvas size where available.
        w = max(int(canvas.cget("width")), canvas.winfo_width())
        h = max(int(canvas.cget("height")), canvas.winfo_height())
        ratio = max(0.85, min(1.15, _read_plan_ratio()))

        # Compact, near-square plan based on Lx/Ly so it does not look overly rectangular.
        title_y = 18
        plan_w = min(w * 0.46, 210.0)
        plan_h = plan_w / ratio
        plan_h = min(plan_h, h * 0.46, 150.0)
        plan_w = plan_h * ratio

        left = 0.5 * (w - plan_w)
        right = left + plan_w
        top = 56
        bottom = top + plan_h
        cx, cy = 0.5 * (left + right), 0.5 * (top + bottom)

        # Slab outline
        canvas.create_rectangle(left, top, right, bottom, width=2, outline="#1f2d3d")
        canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="#1f2d3d", outline="#1f2d3d")
        canvas.create_text(cx, cy - 15, text="Extra center mass", font=("Segoe UI", 8), fill="#1f2d3d")

        # Labels are placed outside the slab with fixed offsets to avoid overlap.
        corners = [
            (left,  bottom, "C1", "",    -12, 10, "e"),
            (right, bottom, "C2", "",    12, 10, "w"),
            (right, top,    "C3", "",   12, -10, "w"),
            (left,  top,    "C4", "",   -12, -10, "e"),
        ]
        for x, y, label, coord, dx, dy, anchor in corners:
            canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill="#ffffff", outline="#1f2d3d", width=2)
            canvas.create_text(x + dx, y + dy - 7, text=label, font=("Segoe UI", 10, "bold"), fill="#1f2d3d", anchor=anchor)
            canvas.create_text(x + dx, y + dy + 8, text=coord, font=("Segoe UI", 10), fill="#4a5568", anchor=anchor)

        # Axes and shaking direction, kept below the plan.
        axis_y = min(h - 15, bottom + 42)

        canvas.create_line(left - 58, axis_y, left, axis_y, arrow=tk.LAST, width=2, fill="#1f2d3d")
        canvas.create_text(left + 8, axis_y, text="X / direction of shaking", anchor="w", font=("Segoe UI", 10), fill="#1f2d3d")

        canvas.create_line(left - 58, axis_y, left - 58, axis_y - 58, arrow=tk.LAST, width=2, fill="#1f2d3d")
        canvas.create_text(left - 58, axis_y - 70, text="Y", font=("Segoe UI", 10), fill="#1f2d3d")

        canvas.create_text(
            w / 2, title_y,
            text="Plan view: same C1-C4 labels are used for extra masses",
            font=("Segoe UI", 8, "bold"), fill="#1f2d3d"
        )

    def draw_column_3d_sketch(canvas):
        """Draws a compact color-coded 3D frame sketch showing column numbering C1-C4."""
        canvas.delete("all")
        # Use the actual displayed canvas size, not only the requested size.
        # This keeps the frame anchored at the real bottom when the GUI stretches the canvas.
        w = max(int(canvas.cget("width")), canvas.winfo_width())
        h = max(int(canvas.cget("height")), canvas.winfo_height())
        ratio = max(0.85, min(1.15, _read_plan_ratio()))
        n_show = max(1, min(20, get_int_story_count()))

        # Bottom-anchored perspective geometry. The base of the model is placed
        # very close to the bottom of the canvas, and story height is reduced
        # automatically so higher-story models grow upward without leaving the box.
        top_margin = 44
        bottom_margin = 70
        available_h = max(120, h - top_margin - bottom_margin)
        story_h = min(50, available_h / max(n_show, 1))
        story_h = max(8, story_h)

        # Keep the sketch compact in X so the building does not look too stretched.
        base_w = min(w * 0.35, 225 * ratio)
        base_w = max(82, base_w)
        depth_x = min(60, max(40, base_w * 0.48))
        depth_y = min(44, max(30, depth_x * 0.72))

        total_w = base_w + depth_x
        x0 = max(36, (w - total_w) / 2.0 - 14)
        y0 = h - bottom_margin

        c1 = (x0, y0)
        c2 = (x0 + base_w, y0)
        c4 = (x0 + depth_x, y0 - depth_y)
        c3 = (x0 + base_w + depth_x, y0 - depth_y)

        def up(p, k):
            return (p[0], p[1] - story_h * k)

        column_colors = {
            "C1": "#2563eb",  # blue
            "C2": "#dc2626",  # red
            "C3": "#16a34a",  # green
            "C4": "#9333ea",  # purple
        }

        # Light slab/floor panels, similar in spirit to post-processing views.
        for k in range(0, n_show + 1):
            p1, p2, p3, p4 = up(c1, k), up(c2, k), up(c3, k), up(c4, k)
            fill = "#eef4ff" if k == n_show else "#f8fafc"
            outline = "#64748b" if k in (0, n_show) else "#cbd5e1"
            canvas.create_polygon(
                p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1],
                fill=fill, outline=outline, width=2 if k in (0, n_show) else 1
            )

        # Colored columns and visible nodes.
        # If a column is unchecked in a story, that column segment is shown as a light dashed line.
        col_defs = [
            ("C1", 1, c1),
            ("C2", 2, c2),
            ("C3", 3, c3),
            ("C4", 4, c4),
        ]
        for label, col_id, base_pt in col_defs:
            color = column_colors[label]
            for story in range(1, n_show + 1):
                present = True
                try:
                    present = column_present_vars[story][col_id].get()
                except Exception:
                    present = True

                p_bot = up(base_pt, story - 1)
                p_top = up(base_pt, story)

                if present:
                    try:
                        orient = column_orient_vars[story][col_id].get()
                    except Exception:
                        orient = "Weak axis"

                    col_width = 7 if orient == "Strong axis" else 4

                    canvas.create_line(
                        p_bot[0], p_bot[1], p_top[0], p_top[1],
                        width=col_width, fill=color
                    )
                else:
                    canvas.create_line(
                        p_bot[0], p_bot[1], p_top[0], p_top[1],
                        width=2, fill="#cbd5e1", dash=(3, 3)
                    )

            for k in range(0, n_show + 1):
                x, y = up(base_pt, k)
                canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#ffffff", outline=color, width=2)

        # Column labels are placed around the middle height of the frame,
        # instead of at the roof, so they remain readable for tall models.
        mid_k = max(0.5, n_show * 0.15)
        mid = {label: up(pt, mid_k) for label, col_id, pt in col_defs}
        label_specs = [
            ("C1", "C 1", -10, -8, "e"),
            ("C2", "C2",  -20, -8, "w"),
            ("C3", "C3",  10, -8, "w"),
            ("C4", "C4", -10, -8, "e"),
        ]
        for key, lab, dx, dy, anchor in label_specs:
            x, y = mid[key]
            y_text = max(48, min(h - 46, y + dy))
            canvas.create_text(x + dx, y_text, text=lab, font=("Segoe UI", 10, "bold"), fill=column_colors[key], anchor=anchor)

        # Axes: Y arrow is parallel to the building's displayed Y/depth direction
        # (the same direction from C1 to C4 in this perspective sketch).
        ax0 = (w - 90, h - 20)
        # Y is exactly parallel to the building depth direction C1 -> C4.
        y_len = 44.0
        y_norm = math.sqrt(depth_x**2 + depth_y**2)
        y_arrow = (depth_x / y_norm * y_len, -depth_y / y_norm * y_len)
        canvas.create_line(ax0[0], ax0[1], ax0[0] + 44, ax0[1], arrow=tk.LAST, width=2, fill="#1f2d3d")
        canvas.create_text(ax0[0] + 52, ax0[1], text="X", anchor="w", font=("Segoe UI", 9), fill="#1f2d3d")
        canvas.create_line(ax0[0], ax0[1], ax0[0] + y_arrow[0], ax0[1] + y_arrow[1], arrow=tk.LAST, width=2, fill="#1f2d3d")
        canvas.create_text(ax0[0] + y_arrow[0] + 6, ax0[1] + y_arrow[1] - 2, text="Y", anchor="w", font=("Segoe UI", 9), fill="#1f2d3d")
        canvas.create_line(ax0[0], ax0[1], ax0[0], ax0[1] - 44, arrow=tk.LAST, width=2, fill="#1f2d3d")
        canvas.create_text(ax0[0], ax0[1] - 54, text="Z", anchor="center", font=("Segoe UI", 9), fill="#1f2d3d")

    def update_column_presence_gui(story=None, col=None):
        """
        Updates the GUI after a column-present checkbox is changed:
        1. disables/enables the matching orientation combobox
        2. redraws the 3D sketch
        """
        stories = [story] if story is not None else list(column_present_vars.keys())

        for st in stories:
            for c in (1, 2, 3, 4):
                if col is not None and c != col:
                    continue

                present = column_present_vars.get(st, {}).get(c, None)
                widget = column_orient_widgets.get(st, {}).get(c, None)

                if present.get():
                    widget.configure(state="readonly", style="TCombobox")
                else:
                    widget.configure(style="Removed.TCombobox")
                    widget.configure(state="disabled")

        try:
            draw_column_3d_sketch(frame_canvas)
        except Exception:
            pass

    def rebuild_story_tables(update_modes=False):
        if story_rebuild_in_progress["value"]:
            return
        story_rebuild_in_progress["value"] = True
        try:
            nStory = get_int_story_count()
            nStory_var.set(nStory)

            old_heights = {k: v.get() for k, v in story_height_vars.items()}
            old_masses = {k: v.get() for k, v in floor_mass_vars.items()}
            old_present = {
                st: {c: column_present_vars.get(st, {}).get(c, tk.BooleanVar(value=True)).get()
                     for c in (1, 2, 3, 4)}
                for st in range(1, max(column_present_vars.keys(), default=0) + 1)
            }
            old_orient = {
                st: {c: column_orient_vars.get(st, {}).get(c, tk.StringVar(value="Weak axis")).get()
                     for c in (1, 2, 3, 4)}
                for st in range(1, max(column_orient_vars.keys(), default=0) + 1)
            }
            old_fracs = {
                st: [additional_mass_vars.get(st, {}).get(i, tk.StringVar(value="0.0")).get() for i in range(5)]
                for st in range(1, max(additional_mass_vars.keys(), default=0) + 1)
            }

            story_height_vars.clear()
            floor_mass_vars.clear()
            column_present_vars.clear()
            column_orient_vars.clear()
            column_orient_widgets.clear()
            additional_mass_vars.clear()

            clear_frame(story_table_frame)
            clear_frame(orientation_table_frame)
            clear_frame(floor_mass_table_frame)

            # Story height / total mass / columns-present table
            ttk.Label(story_table_frame, text="Story", style="Small.TLabel").grid(row=0, column=0, padx=6, pady=4)
            ttk.Label(story_table_frame, text="Height (m)", style="Small.TLabel").grid(row=0, column=1, padx=6, pady=4)
            ttk.Label(story_table_frame, text="Self-weight floor mass (kg)", style="Small.TLabel").grid(row=0, column=2, padx=6, pady=4)
            ttk.Label(story_table_frame, text="Columns present", style="Small.TLabel").grid(row=0, column=3, columnspan=4, padx=6, pady=4)
            corner_headers = {
                1: "C1",
                2: "C2",
                3: "C3",
                4: "C4",
            }
            for c in (1, 2, 3, 4):
                ttk.Label(story_table_frame, text=corner_headers[c], style="Small.TLabel", justify="center").grid(row=1, column=2 + c, padx=6, pady=2)

            for story in range(1, nStory + 1):
                row = story + 1
                ttk.Label(story_table_frame, text=str(story), style="Small.TLabel").grid(row=row, column=0, padx=6, pady=3)
                h_default = old_heights.get(story, old_heights.get(story - 1, "0.24"))
                m_default = old_masses.get(story, old_masses.get(story - 1, "0.27"))
                story_height_vars[story] = tk.StringVar(value=h_default)
                floor_mass_vars[story] = tk.StringVar(value=m_default)
                ttk.Entry(story_table_frame, textvariable=story_height_vars[story], width=13).grid(row=row, column=1, padx=6, pady=3)
                ttk.Entry(story_table_frame, textvariable=floor_mass_vars[story], width=16).grid(row=row, column=2, padx=6, pady=3)
                column_present_vars[story] = {}
                for c in (1, 2, 3, 4):
                    present_default = old_present.get(story, old_present.get(story - 1, {})).get(c, True)
                    column_present_vars[story][c] = tk.BooleanVar(value=present_default)
                    chk = ttk.Checkbutton(
                        story_table_frame,
                        variable=column_present_vars[story][c],
                        command=lambda st=story, cc=c: update_column_presence_gui(st, cc)
                    )
                    chk.grid(row=row, column=2 + c, padx=6, pady=3)

            # Column orientation table
            ttk.Label(orientation_table_frame, text="Story", style="Small.TLabel").grid(row=0, column=0, padx=6, pady=4)
            corner_headers = {
                1: "C1\n(0,0)",
                2: "C2\n(Lx,0)",
                3: "C3\n(Lx,Ly)",
                4: "C4\n(0,Ly)",
            }
            for c in (1, 2, 3, 4):
                ttk.Label(orientation_table_frame, text=corner_headers[c], style="Small.TLabel", justify="center").grid(row=0, column=c, padx=6, pady=4)

            for story in range(1, nStory + 1):
                ttk.Label(orientation_table_frame, text=str(story), style="Small.TLabel").grid(row=story, column=0, padx=6, pady=3)
                column_orient_vars[story] = {}
                for c in (1, 2, 3, 4):
                    default = old_orient.get(story, old_orient.get(story - 1, {})).get(c, "Weak axis")
                    if default not in ("Weak axis", "Strong axis"):
                        default = "Weak axis"
                    column_orient_vars[story][c] = tk.StringVar(value=default)
                    combo = ttk.Combobox(
                        orientation_table_frame,
                        textvariable=column_orient_vars[story][c],
                        values=("Weak axis", "Strong axis"),
                        state="readonly",
                        width=12
                    )
                    combo.grid(row=story, column=c, padx=6, pady=3)
                    combo.bind("<<ComboboxSelected>>", refresh_basic_sketch)

                    if story not in column_orient_widgets:
                        column_orient_widgets[story] = {}
                    column_orient_widgets[story][c] = combo

            # Floor mass placement table
            headers = ["Story", "Extra center\nkg", "Extra C1\nkg", "Extra C2\nkg", "Extra C3\nkg", "Extra C4\nkg"]
            for col, header in enumerate(headers):
                ttk.Label(floor_mass_table_frame, text=header, style="Small.TLabel", justify="center").grid(row=0, column=col, padx=5, pady=4)

            for story in range(1, nStory + 1):
                ttk.Label(floor_mass_table_frame, text=str(story), style="Small.TLabel").grid(row=story, column=0, padx=5, pady=3)
                vals = old_fracs.get(story, old_fracs.get(story - 1, ["0.0", "0.0", "0.0", "0.0", "0.0"]))
                if len(vals) != 5:
                    vals = ["0.0", "0.0", "0.0", "0.0", "0.0"]
                additional_mass_vars[story] = {}
                for i, val in enumerate(vals):
                    additional_mass_vars[story][i] = tk.StringVar(value=val)
                    ent = ttk.Entry(floor_mass_table_frame, textvariable=additional_mass_vars[story][i], width=9)
                    ent.grid(row=story, column=i + 1, padx=5, pady=3)

            # Keep student inputs consistent when the number of stories changes.
            # This is applied only when the story count changes, not every time Run is clicked,
            # so manual user edits to the mode counts are not overwritten unnecessarily.
            if update_modes:
                try:
                    if "numModes" in entries:
                        target_modes = max(2, min(4, nStory))
                        set_entry_text(entries["numModes"], str(target_modes))
                    if "nCalibModes" in entries:
                        current_num_modes = int(entries["numModes"].get()) if "numModes" in entries else max(2, min(4, nStory))
                        target_calib_modes = max(1, min(3, current_num_modes, nStory))
                        set_entry_text(entries["nCalibModes"], str(target_calib_modes))
                except Exception:
                    pass

            update_column_presence_gui()
            story_table_count["value"] = nStory

        finally:
            story_rebuild_in_progress["value"] = False

    def schedule_story_table_rebuild(*args):
        if not story_tables_ready["value"] or story_rebuild_in_progress["value"]:
            return
        if story_rebuild_job["id"] is not None:
            try:
                root.after_cancel(story_rebuild_job["id"])
            except Exception:
                pass
        story_rebuild_job["id"] = root.after(150, lambda: rebuild_story_tables(update_modes=True))

    def set_all_column_orientation(value):
        for story in column_orient_vars:
            for c in column_orient_vars[story]:
                try:
                    if not column_present_vars[story][c].get():
                        continue
                except Exception:
                    pass
                column_orient_vars[story][c].set(value)

    # -------------------------------------------------------------------------
    # Tab 1: Basic model, compact layout
    # -------------------------------------------------------------------------
    tab1_top = ttk.Frame(tab1_body)
    tab1_top.pack(fill="x", expand=False, pady=8, padx=8)

    tab1_left = ttk.Frame(tab1_top, width=760)
    tab1_left.pack(side="left", fill="both", expand=True, padx=(0, 8), anchor="n")

    tab1_right = ttk.Frame(tab1_top, width=360)
    tab1_right.pack(side="left", fill="y", expand=False, padx=(8, 0), anchor="n")
    tab1_right.pack_propagate(False)

    geom = ttk.LabelFrame(tab1_left, text="Geometry", style="Section.TLabelframe", padding=(10, 8))
    geom.pack(fill="x", expand=False, pady=(0, 8), padx=0, anchor="n")
    geom.columnconfigure(0, weight=0)
    geom.columnconfigure(1, weight=1)

    r = 0
    r = add_field(geom, r, "Building length in X (m)", "Lx", "0.245", "Plan dimension along X.", width=36)
    r = add_field(geom, r, "Building width in Y (m)", "Ly", "0.23", "Plan dimension along Y.", width=36)

    ttk.Label(geom, text="Number of stories", style="FieldLabel.TLabel").grid(
        row=r, column=0, sticky="w", padx=(10, 8), pady=(8, 1)
    )
    story_spin = ttk.Spinbox(geom, from_=1, to=20, textvariable=nStory_var, width=8, command=schedule_story_table_rebuild)
    story_spin.grid(row=r, column=1, sticky="w", padx=(0, 10), pady=(8, 1))
    ttk.Label(
        geom,
        text="Automatically updates story rows and safe mode counts.",
        style="Hint.TLabel",
        wraplength=620,
        justify="left"
    ).grid(row=r + 1, column=0, columnspan=2, sticky="w", padx=(10, 10), pady=(0, 6))
    r += 2

    ttk.Label(
        geom,
        text="Columns present are selected below using C1-C4 checkboxes.",
        style="Hint.TLabel",
        wraplength=620,
        justify="left"
    ).grid(row=r, column=0, columnspan=2, sticky="w", padx=(10, 10), pady=(0, 6))

    sec = ttk.LabelFrame(tab1_left, text="Sections", style="Section.TLabelframe", padding=(10, 8))
    sec.pack(fill="x", expand=False, pady=(0, 0), padx=0, anchor="n")
    sec.columnconfigure(0, weight=0)
    sec.columnconfigure(1, weight=1)

    r = 0
    r = add_field(sec, r, "Column thickness (m)", "t_column", "0.001", "Net thickness of the column section.", width=36)
    r = add_field(sec, r, "Column net width (m)", "b_column", "0.006", "Net column width at the hole center.", width=36)
    r = add_field(sec, r, "Beam thickness / width (m)", "b_beam", "0.001", "Smaller beam cross-section dimension.", width=36)
    r = add_field(sec, r, "Beam net height (m)", "h_beam", "0.008", "Net beam height at the hole center.", width=36)

    frame_sketch_box = ttk.LabelFrame(tab1_right, text="3D Column Numbering Sketch", style="Section.TLabelframe", padding=(10, 8))
    frame_sketch_box.pack(fill="both", expand=True, pady=0, padx=0, anchor="n")
    frame_canvas = tk.Canvas(frame_sketch_box, width=350, height=300, bg="white", highlightthickness=0)
    frame_canvas.pack(anchor="center", fill="both", expand=True, padx=4, pady=4)
    draw_column_3d_sketch(frame_canvas)

    def refresh_basic_sketch(*args):
        try:
            draw_column_3d_sketch(frame_canvas)
        except Exception:
            pass

    try:
        entries["Lx"].bind("<FocusOut>", refresh_basic_sketch)
        entries["Ly"].bind("<FocusOut>", refresh_basic_sketch)
        nStory_var.trace_add("write", lambda *args: refresh_basic_sketch())
        frame_canvas.bind("<Configure>", refresh_basic_sketch)
    except Exception:
        pass

    tab1_bottom = ttk.Frame(tab1_body)
    tab1_bottom.pack(fill="x", expand=False, pady=(0, 8), padx=8)

    story_box = ttk.LabelFrame(tab1_bottom, text="Story Height + Self-Weight Floor Mass + Columns Present", style="Section.TLabelframe", padding=(10, 8))
    story_box.pack(side="left", fill="both", expand=True, padx=(0, 6), anchor="n")
    story_table_frame = ttk.Frame(story_box)
    story_table_frame.pack(anchor="w", padx=4, pady=4)

    orient_box = ttk.LabelFrame(tab1_bottom, text="Column Orientation for X Shaking", style="Section.TLabelframe", padding=(10, 8))
    orient_box.pack(side="left", fill="both", expand=True, padx=(6, 0), anchor="n")
    orient_controls = ttk.Frame(orient_box)
    orient_controls.pack(fill="x", pady=(0, 4))
    ttk.Button(orient_controls, text="Set all Weak axis", command=lambda: set_all_column_orientation("Weak axis")).pack(side="left", padx=(4, 6))
    ttk.Button(orient_controls, text="Set all Strong axis", command=lambda: set_all_column_orientation("Strong axis")).pack(side="left", padx=(0, 6))
    orientation_table_frame = ttk.Frame(orient_box)
    orientation_table_frame.pack(anchor="w", padx=4, pady=4)

    # -------------------------------------------------------------------------
    # Tab 2: Material/mass placement and analysis, side-by-side layout retained
    # -------------------------------------------------------------------------
    tab2_top = ttk.Frame(tab2_body)
    tab2_top.pack(fill="x", expand=False, pady=8, padx=8)

    tab2_left = ttk.Frame(tab2_top)
    tab2_left.pack(side="left", fill="both", expand=True, padx=(0, 6), anchor="n")

    tab2_right = ttk.Frame(tab2_top)
    tab2_right.pack(side="left", fill="both", expand=True, padx=(6, 0), anchor="n")

    mat = ttk.LabelFrame(tab2_left, text="Material", style="Section.TLabelframe", padding=(10, 8))
    mat.pack(fill="x", expand=False, pady=0, padx=0, anchor="n")
    mat.columnconfigure(0, weight=0)
    mat.columnconfigure(1, weight=1)

    r = 0
    r = add_field(mat, r, "Elastic modulus E (Pa)", "E", "200e9", "Young's modulus.")
    r = add_field(mat, r, "Poisson ratio ν (-)", "nu", "0.33", "Used to compute shear modulus.")

    ttk.Label(
        mat,
        text="Corner legend for additional mass placement: the same C1-C4 labels are used in the additional-mass table below.",
        style="Hint.TLabel",
        wraplength=480,
        justify="left"
    ).grid(row=r, column=0, columnspan=2, sticky="w", padx=(10, 10), pady=(2, 6))
    r += 1

    mass_legend_canvas = tk.Canvas(mat, width=420, height=220, bg="white", highlightthickness=0)
    mass_legend_canvas.grid(row=r, column=0, columnspan=2, sticky="w", padx=(10, 10), pady=(0, 6))
    draw_corner_legend(mass_legend_canvas)
    mass_legend_canvas.bind("<Configure>", lambda event: draw_corner_legend(mass_legend_canvas))
    try:
        entries["Lx"].bind("<FocusOut>", lambda event: draw_corner_legend(mass_legend_canvas), add="+")
        entries["Ly"].bind("<FocusOut>", lambda event: draw_corner_legend(mass_legend_canvas), add="+")
    except Exception:
        pass
    r += 1

    ana = ttk.LabelFrame(tab2_right, text="Analysis", style="Section.TLabelframe", padding=(10, 8))
    ana.pack(fill="x", expand=False, pady=0, padx=0, anchor="n")
    ana.columnconfigure(0, weight=0)
    ana.columnconfigure(1, weight=1)

    r = 0
    r = add_field(ana, r, "Number of modes in model", "numModes", "4", "Total number of numerical modes to extract.")
    r = add_field(ana, r, "Damping ratio ζ (-)", "zeta", "0.005", "Target Rayleigh damping ratio.")
    r = add_field(ana, r, "Ground-motion scale factor", "gmFactor", "9.81", "Multiplier applied to the input record.")
    r = add_field(ana, r, "Ground-motion file", "gmFile", "input/sine_1Hz_accel.txt", "Path to the acceleration time-history file.")
    r = add_field(ana, r, "Ground-motion time step dt (s)", "dtGM", "0.01", "Sampling time step of the ground-motion record.")
    r = add_check(ana, r, "Run transient after calibration", "run_transient", True, "If unchecked, only modal analysis and calibration are performed.")

    mass_box = ttk.LabelFrame(tab2_body, text="Additional Mass Placement (kg)", style="Section.TLabelframe", padding=(10, 8))
    mass_box.pack(fill="x", expand=False, pady=(0, 8), padx=8, anchor="n")
    floor_mass_table_frame = ttk.Frame(mass_box)
    floor_mass_table_frame.pack(anchor="w", padx=4, pady=4)

    # -------------------------------------------------------------------------
    # Tab 3
    # -------------------------------------------------------------------------
    calib_container = ttk.Frame(tab3)
    calib_container.pack(fill="x", expand=False, pady=8, padx=8)

    calib_left = ttk.LabelFrame(calib_container, text="Calibration Settings", style="Section.TLabelframe", padding=(10, 8))
    calib_left.pack(side="left", fill="both", expand=True, padx=(0, 6), anchor="n")

    calib_right = ttk.LabelFrame(calib_container, text="Optimizer Bounds", style="Section.TLabelframe", padding=(10, 8))
    calib_right.pack(side="left", fill="both", expand=True, padx=(6, 0), anchor="n")

    calib_left.columnconfigure(0, weight=0)
    calib_left.columnconfigure(1, weight=1)
    calib_right.columnconfigure(0, weight=0)
    calib_right.columnconfigure(1, weight=1)

    r_left = 0
    r_left = add_check(calib_left, r_left, "Enable automatic calibration", "enable_calibration", True,
                       "If unchecked, the model uses the original parameters only.")
    r_left = add_check(calib_left, r_left, "Use mode shapes in calibration", "use_mode_shapes", False,
                       "If the experimental mode-shape file has a different number of story values, the code will explain the mismatch clearly.")

    ttk.Label(calib_left, text="Mass calibration target", style="FieldLabel.TLabel").grid(
        row=r_left, column=0, sticky="w", padx=(10, 8), pady=(8, 1)
    )
    mass_calibration_scope_var = tk.StringVar(value="Self-weight mass only")
    mass_calibration_scope_combo = ttk.Combobox(
        calib_left,
        textvariable=mass_calibration_scope_var,
        values=("Self-weight mass only", "Total mass including additional masses"),
        state="readonly",
        width=34
    )
    mass_calibration_scope_combo.grid(row=r_left, column=1, sticky="w", padx=(0, 10), pady=(8, 1))
    ttk.Label(
        calib_left,
        text=(
            "Choose whether calibration scales only the self-weight masses from the Basic Model tab, "
            "or scales both self-weight and additional masses for each floor."
        ),
        style="Hint.TLabel",
        wraplength=480,
        justify="left"
    ).grid(row=r_left + 1, column=0, columnspan=2, sticky="w", padx=(10, 10), pady=(0, 6))
    r_left += 2

    r_left = add_field(calib_left, r_left, "Number of modes used in calibration", "nCalibModes", "3",
                       "How many modes from the experimental file should be used.")
    r_left = add_field(calib_left, r_left, "Frequency tolerance (%)", "freq_tol_percent", "5.0",
                       "Used for reporting only.")
    r_left = add_field(calib_left, r_left, "Frequency weight", "w_freq", "1.0",
                       "Weight of frequency mismatch in the objective function.")
    r_left = add_field(calib_left, r_left, "Mode-shape weight", "w_mode", "0.35",
                       "Weight of mode-shape mismatch in the objective function.")

    r_right = 0
    r_right = add_field(calib_right, r_right, "Lower bound for E scale", "E_scale_lb", "0.70",
                        "Calibrated E = original E × E_scale.")
    r_right = add_field(calib_right, r_right, "Upper bound for E scale", "E_scale_ub", "1.30",
                        "Calibrated E = original E × E_scale.")
    r_right = add_field(calib_right, r_right, "Lower bound for mass scales", "m_scale_lb", "0.70",
                        "Lower bound for each story mass scale. The selected mass calibration target controls whether this scale affects only self-weight mass or total mass.")
    r_right = add_field(calib_right, r_right, "Upper bound for mass scales", "m_scale_ub", "1.30",
                        "Upper bound for each story mass scale. The selected mass calibration target controls whether this scale affects only self-weight mass or total mass.")
    r_right = add_field(calib_right, r_right, "Maximum optimizer evaluations", "max_nfev", "200",
                        "Larger values may improve fit, especially when total mass is calibrated, but increase runtime.")
    r_right = add_check(calib_right, r_right, "Show uncalibrated response on run plot", "show_uncalibrated_response", True,
                        "After using Calibrate, overlays the stored uncalibrated roof response on the calibrated run plot.")

    # Build initial story-dependent tables after their frames exist.
    rebuild_story_tables(update_modes=False)
    story_tables_ready["value"] = True
    nStory_var.trace_add("write", schedule_story_table_rebuild)
    story_spin.bind("<Return>", lambda event: rebuild_story_tables(update_modes=True))
    story_spin.bind("<FocusOut>", lambda event: rebuild_story_tables(update_modes=True))

    btn_frame = ttk.Frame(outer, style="Main.TFrame")
    btn_frame.pack(fill="x", pady=(6, 2), anchor="w")

    def table_to_story_column_layout(nStory):
        out = {}
        for story in range(1, nStory + 1):
            cols = [c for c in (1, 2, 3, 4) if column_present_vars[story][c].get()]
            if not cols:
                raise ValueError(f"Story {story} must have at least one column present.")
            out[story] = cols
        return out

    def table_to_column_orientation(nStory):
        out = {}
        for story in range(1, nStory + 1):
            out[story] = {}
            for c in (1, 2, 3, 4):
                out[story][c] = "strong" if column_orient_vars[story][c].get() == "Strong axis" else "weak"
        return out

    def table_to_additional_masses(nStory):
        """
        Reads additional masses in kg from the Mass + Analysis tab.

        These masses are extra masses only. The self-weight floor mass from
        the Basic Model tab is always assigned at the floor center and is the
        only mass changed by calibration.
        """
        out = {}
        for story in range(1, nStory + 1):
            vals = []
            for i in range(5):
                vals.append(float(additional_mass_vars[story][i].get()))
            if any(v < 0.0 for v in vals):
                raise ValueError(f"Additional masses in story {story} cannot be negative.")
            out[story] = vals
        return out

    def collect_gui_inputs():
        out = {}
        requested_story_count = get_int_story_count()
        rebuild_story_tables(update_modes=(story_table_count["value"] != requested_story_count))
        out["Lx"] = float(entries["Lx"].get())
        out["Ly"] = float(entries["Ly"].get())

        out["nStory"] = get_int_story_count()
        out["story_heights"] = [float(story_height_vars[st].get()) for st in range(1, out["nStory"] + 1)]
        out["floor_masses"] = [float(floor_mass_vars[st].get()) for st in range(1, out["nStory"] + 1)]

        out["t_column"] = float(entries["t_column"].get())
        out["b_column"] = float(entries["b_column"].get())
        out["b_beam"] = float(entries["b_beam"].get())
        out["h_beam"] = float(entries["h_beam"].get())

        out["E"] = float(entries["E"].get())
        out["nu"] = float(entries["nu"].get())

        out["numModes"] = int(entries["numModes"].get())
        out["zeta"] = float(entries["zeta"].get())
        out["gmFactor"] = float(entries["gmFactor"].get())
        out["gmFile"] = entries["gmFile"].get().strip()
        out["dtGM"] = float(entries["dtGM"].get())

        out["show_info"] = False
        out["run_transient"] = bool_vars["run_transient"].get()

        out["enable_calibration"] = bool_vars["enable_calibration"].get()
        out["use_mode_shapes"] = bool_vars["use_mode_shapes"].get()
        if mass_calibration_scope_var.get() == "Total mass including additional masses":
            out["mass_calibration_scope"] = "total_mass"
        else:
            out["mass_calibration_scope"] = "self_weight_only"
        out["nCalibModes"] = int(entries["nCalibModes"].get())
        out["freq_tol_percent"] = float(entries["freq_tol_percent"].get())
        out["w_freq"] = float(entries["w_freq"].get())
        out["w_mode"] = float(entries["w_mode"].get())
        out["E_scale_lb"] = float(entries["E_scale_lb"].get())
        out["E_scale_ub"] = float(entries["E_scale_ub"].get())
        out["m_scale_lb"] = float(entries["m_scale_lb"].get())
        out["m_scale_ub"] = float(entries["m_scale_ub"].get())
        out["max_nfev"] = int(entries["max_nfev"].get())
        out["show_uncalibrated_response"] = bool_vars["show_uncalibrated_response"].get()

        if out["Lx"] <= 0 or out["Ly"] <= 0:
            raise ValueError("Building length and width must be positive.")
        if out["nStory"] < 1:
            raise ValueError("Please enter at least one story.")
        if any(h <= 0 for h in out["story_heights"]):
            raise ValueError("All story heights must be positive.")
        if any(m <= 0 for m in out["floor_masses"]):
            raise ValueError("All self-weight floor masses must be positive.")
        if out["t_column"] <= 0 or out["b_column"] <= 0:
            raise ValueError("Column dimensions must be positive.")
        if out["b_beam"] <= 0 or out["h_beam"] <= 0:
            raise ValueError("Beam dimensions must be positive.")
        if out["E"] <= 0:
            raise ValueError("Elastic modulus E must be positive.")
        if not (0.0 <= out["nu"] < 0.5):
            raise ValueError("Poisson ratio must be between 0.0 and 0.5.")
        if out["numModes"] < 2:
            raise ValueError("Set numModes >= 2.")
        if out["zeta"] < 0:
            raise ValueError("Damping ratio cannot be negative.")
        if out["dtGM"] <= 0:
            raise ValueError("Ground-motion time step must be positive.")
        if out["E_scale_lb"] <= 0 or out["E_scale_ub"] <= 0:
            raise ValueError("E scale bounds must be positive.")
        if out["m_scale_lb"] <= 0 or out["m_scale_ub"] <= 0:
            raise ValueError("Mass scale bounds must be positive.")
        if out["E_scale_lb"] >= out["E_scale_ub"]:
            raise ValueError("E scale lower bound must be smaller than upper bound.")
        if out["m_scale_lb"] >= out["m_scale_ub"]:
            raise ValueError("Mass scale lower bound must be smaller than upper bound.")
        if out["max_nfev"] <= 0:
            raise ValueError("Maximum optimizer evaluations must be positive.")
        if out["nCalibModes"] <= 0:
            raise ValueError("Number of calibration modes must be positive.")
        if out["nCalibModes"] > out["numModes"]:
            raise ValueError("Number of calibration modes cannot exceed total extracted modes.")
        if out["nCalibModes"] > out["nStory"]:
            raise ValueError(
                "Number of calibration modes should not exceed the number of stories for the current UX mode-shape calibration. "
                f"Current model has {out['nStory']} stories. Use nCalibModes <= {out['nStory']}."
            )

        out["story_column_layout"] = table_to_story_column_layout(out["nStory"])
        out["column_orientation_layout"] = table_to_column_orientation(out["nStory"])
        out["additional_masses"] = table_to_additional_masses(out["nStory"])
        return out

    def make_calibration_signature(params):
        keys = [
            "Lx", "Ly", "story_heights", "t_column", "b_column",
            "b_beam", "h_beam", "E", "nu", "floor_masses",
            "numModes", "zeta", "gmFactor", "gmFile", "dtGM",
            "enable_calibration", "use_mode_shapes", "mass_calibration_scope", "nCalibModes",
            "freq_tol_percent", "w_freq", "w_mode", "E_scale_lb",
            "E_scale_ub", "m_scale_lb", "m_scale_ub", "max_nfev",
            "story_column_layout",
            "column_orientation_layout", "additional_masses"
        ]
        return json.dumps({k: deep_round(params.get(k), 8) for k in keys}, sort_keys=True)

    def save_overlay_response(path, response):
        if response is None:
            return
        np.savez(
            path,
            t_hist=response["t_hist"],
            u_hist=response["u_hist"],
            a_hist=response["a_hist"],
        )

    def calibrate_clicked():
        try:
            params_for_calib = collect_gui_inputs()
            os.makedirs("input", exist_ok=True)
            os.makedirs("output", exist_ok=True)

            show_info = params_for_calib["show_info"]
            original_params = copy.deepcopy(params_for_calib)
            write_json("output/original_inputs.json", original_params)

            exp_data = prepare_experimental_modal_data(params_for_calib)
            write_json("output/experimental_modal_data_loaded.json", exp_data["raw_data"])

            print("\nRunning original modal analysis...")
            modal_before = extract_modal_results(original_params, normalize_modes=True, show_info=show_info)
            export_modal_files(modal_before, "original")

            if params_for_calib["enable_calibration"]:
                print("\nRunning automatic calibration...")
                calib_result, calibrated_params = run_calibration(original_params, exp_data, show_info=show_info)
                print("\nCalibration finished.")
                print("Success:", calib_result.success)
                print("Message:", calib_result.message)

                calibration_warning = ""
                if not calib_result.success:
                    # SciPy reports success=False when the maximum number of function
                    # evaluations is reached, even though it still returns the best
                    # parameter set found so far. For this GUI workflow, that is not a
                    # fatal calibration error. We keep the best result, run the calibrated
                    # modal analysis, and clearly warn the user instead of stopping.
                    msg = str(calib_result.message)
                    if calib_result.status == 0 or "maximum number of function evaluations" in msg.lower():
                        calibration_warning = (
                            "\n\nNote: the optimizer reached the maximum number of evaluations. "
                            "The best calibrated parameters found so far were saved and used. "
                            "If you want a tighter convergence, increase 'Maximum optimizer evaluations'."
                        )
                        print("Warning:", calibration_warning)
                    else:
                        raise RuntimeError(
                            "There is a problem during calibration.\n"
                            + msg
                        )

                print("\nRunning calibrated modal analysis...")
                modal_after = extract_modal_results(calibrated_params, normalize_modes=True, show_info=show_info)
                export_modal_files(modal_after, "calibrated")
            else:
                calib_result = None
                calibrated_params = original_params
                modal_after = modal_before

            write_json("output/calibrated_inputs.json", calibrated_params)

            report = make_modal_comparison_report(
                exp_data, modal_before, modal_after,
                original_params, calibrated_params, calib_result
            )
            write_json("output/modal_comparison_report.json", report)
            write_text("output/modal_comparison_report.txt", report_to_text(report))
            save_calibration_summary_figure(
                exp_data, modal_before, modal_after,
                original_params, calibrated_params,
                save_path="output/calibration_summary.png"
            )

            uncalibrated_response = None
            if params_for_calib["show_uncalibrated_response"]:
                print("\nCalculating and storing uncalibrated transient response...")
                modal_original_dynamic = extract_modal_results(original_params, normalize_modes=True, show_info=show_info)
                uncalibrated_response = run_transient_analysis_collect_data(
                    original_params, modal_original_dynamic,
                    show_info=show_info,
                    recorder_prefix="original_"
                )
                save_overlay_response("output/original_transient_response_overlay.npz", uncalibrated_response)

            calibration_state["available"] = True
            calibration_state["input_signature"] = make_calibration_signature(params_for_calib)
            calibration_state["calibrated_params"] = copy.deepcopy(calibrated_params)
            calibration_state["calibrated_modal"] = modal_after
            calibration_state["uncalibrated_response"] = uncalibrated_response

            messagebox.showinfo(
                "Calibration complete",
                "Calibration process done successfully.\n"
                "Now click Run Analysis to run the transient analysis\n"
                "with the calibrated model."
                + (calibration_warning if 'calibration_warning' in locals() else "")
            )


        except Exception as e:
            messagebox.showerror(
                "Calibration failed",
                "There is a problem during calibration.\n\n" + str(e)
            )

    def run_clicked():
        try:
            run_params = collect_gui_inputs()
            run_params["gui_action"] = "run"
            run_params["use_precalibrated"] = False
            run_params["overlay_uncalibrated_response"] = None

            current_signature = make_calibration_signature(run_params)
            if not calibration_state["available"]:
                answer = messagebox.askyesno(
                    "Run without calibration?",
                    "Calibration has not been performed yet.\n\n"
                    "Are you sure you want to run the analysis\n"
                    "without calibration?"
                )
                if not answer:
                    return

            if calibration_state["available"]:
                if calibration_state["input_signature"] == current_signature:
                    run_params["use_precalibrated"] = True
                    run_params["precalibrated_params"] = copy.deepcopy(calibration_state["calibrated_params"])
                    if run_params["show_uncalibrated_response"]:
                        run_params["overlay_uncalibrated_response"] = calibration_state["uncalibrated_response"]
                else:
                    answer = messagebox.askyesno(
                        "Inputs changed after calibration",
                        "Some inputs changed after you clicked Calibrate.\n\n"
                        "Run with the current uncalibrated inputs instead?\n\n"
                        "Choose No to go back and recalibrate first."
                    )
                    if not answer:
                        return

            result.clear()
            result.update(run_params)
            root.destroy()

        except Exception as e:
            messagebox.showerror("Invalid input", str(e))

    def cancel_clicked():
        root.destroy()
        raise SystemExit("Simulation cancelled by user.")

    ttk.Button(btn_frame, text="Calibrate", command=calibrate_clicked,
               style="Run.TButton").pack(side="left", padx=(2, 8))
    ttk.Button(btn_frame, text="Run Analysis", command=run_clicked,
               style="Run.TButton").pack(side="left", padx=(0, 8))
    ttk.Button(btn_frame, text="Cancel", command=cancel_clicked,
               style="Cancel.TButton").pack(side="left")

    root.mainloop()
    return result

# =============================================================================
# MODEL BUILDING
# =============================================================================
def build_model(params, show_info=False):
    ops.wipe()
    ops.model('Basic', '-ndm', 3, '-ndf', 6)

    Lx = params["Lx"]
    Ly = params["Ly"]
    story_heights = params["story_heights"]
    nStory = params["nStory"]
    story_column_layout = params["story_column_layout"]
    column_orientation_layout = params.get(
        "column_orientation_layout",
        {story: {c: "weak" for c in (1, 2, 3, 4)} for story in range(1, nStory + 1)}
    )
    additional_masses = params.get(
        "additional_masses",
        {story: [0.0, 0.0, 0.0, 0.0, 0.0] for story in range(1, nStory + 1)}
    )

    E = params["E"]
    nu = params["nu"]
    G = E / (2.0 * (1.0 + nu))

    t_column = params["t_column"]
    b_column = params["b_column"]
    b_beam = params["b_beam"]
    h_beam = params["h_beam"]

    A = t_column * b_column
    Iy = b_column * t_column**3 / 12.0
    Iz = t_column * b_column**3 / 12.0
    J = (1.0 / 3.0) * b_column * t_column**3

    Ab = b_beam * h_beam
    Iyb = h_beam * b_beam**3 / 12.0
    Izb = b_beam * h_beam**3 / 12.0
    Jb = (1.0 / 3.0) * h_beam * b_beam**3

    floor_masses = params["floor_masses"]
    total_dynamic_masses = [
        floor_masses[story - 1] + sum(additional_masses.get(story, [0.0, 0.0, 0.0, 0.0, 0.0]))
        for story in range(1, nStory + 1)
    ]
    floor_weights = [m * 9.81 for m in total_dynamic_masses]
    floor_Jm = [floor_masses[i] * (Lx**2 + Ly**2) / 12.0 for i in range(nStory)]

    z_levels = [0.0]
    for h in story_heights:
        z_levels.append(z_levels[-1] + h)

    base_nodes = [1, 2, 3, 4]
    base_coords = [
        (0.0, 0.0, 0.0),
        (Lx,  0.0, 0.0),
        (Lx,  Ly,  0.0),
        (0.0, Ly,  0.0),
    ]

    for nd, (x, y, z) in zip(base_nodes, base_coords):
        ops.node(nd, x, y, z)

    story_node_tags = {}
    master_nodes = []
    mass_assignment = {}
    gravity_corner_masses_by_story = {}

    for story in range(1, nStory + 1):
        z = z_levels[story]
        nds = [story * 10 + 1, story * 10 + 2, story * 10 + 3, story * 10 + 4]
        crds = [
            (0.0, 0.0, z),
            (Lx,  0.0, z),
            (Lx,  Ly,  z),
            (0.0, Ly,  z),
        ]

        for nd, (x, y, zc) in zip(nds, crds):
            ops.node(nd, x, y, zc)

        story_node_tags[story] = nds

        master = 1000 + story
        ops.node(master, Lx / 2.0, Ly / 2.0, z)
        master_nodes.append(master)

    for nd in base_nodes:
        ops.fix(nd, 1, 1, 1, 1, 1, 1)

    for master in master_nodes:
        ops.fix(master, 0, 0, 1, 1, 1, 0)

    for story in range(1, nStory + 1):
        ops.rigidDiaphragm(3, 1000 + story, *story_node_tags[story])

    # -------------------------------------------------------------------------
    # Mass placement
    # -------------------------------------------------------------------------
    # The floor mass entered in the Basic Model tab is treated as the
    # self-weight/structural mass of each floor. It is always assigned at the
    # diaphragm master node at the center of the slab and is never redistributed
    # to the corners.
    #
    # The table in the Mass + Analysis tab is now for additional masses in kg.
    # These additional masses can be placed at the center or at C1-C4. They are
    # included in modal/transient analysis, but they are not changed by the
    # calibration optimizer.
    for story in range(1, nStory + 1):
        self_mass = floor_masses[story - 1]
        extra = additional_masses.get(story, [0.0, 0.0, 0.0, 0.0, 0.0])
        extra_center_mass = extra[0]
        extra_corner_masses = extra[1:]

        master = 1000 + story

        # Keep the same floor-center rotational inertia idea for the structural
        # self-weight mass. Extra center mass is treated as a point mass at the
        # diaphragm center; extra corner masses create torsional inertia through
        # their offset from the master node and the rigid-diaphragm constraint.
        center_translational_mass = self_mass + extra_center_mass
        center_Jm = self_mass * (Lx**2 + Ly**2) / 12.0

        ops.mass(
            master,
            center_translational_mass, center_translational_mass, 0.0,
            0.0, 0.0, center_Jm
        )

        for nd, m_corner in zip(story_node_tags[story], extra_corner_masses):
            ops.mass(
                nd,
                m_corner, m_corner, 0.0,
                0.0, 0.0, 0.0
            )

        gravity_corner_masses = [
            (self_mass + extra_center_mass) / 4.0 + m_extra
            for m_extra in extra_corner_masses
        ]
        gravity_corner_masses_by_story[story] = gravity_corner_masses

        mass_assignment[story] = {
            "self_weight_mass_center_kg": self_mass,
            "additional_masses_center_c1_c2_c3_c4_kg": extra[:],
            "total_dynamic_mass_kg": self_mass + sum(extra),
            "center_node": master,
            "center_translational_mass_kg": center_translational_mass,
            "corner_nodes": story_node_tags[story][:],
            "additional_corner_masses_kg": extra_corner_masses,
            "gravity_corner_masses_kg": gravity_corner_masses,
            "center_rotational_mass_Jz": center_Jm,
        }

    # Existing transformations are retained.
    # Tag 1 is the original vertical-column orientation. For a vertical column,
    # vecxz=(1,0,0) makes global X response use column Iy, i.e. weak-axis response
    # for the default rectangular section t_column x b_column.
    ops.geomTransf('Linear', 1, 1.0, 0.0, 0.0)

    # Beam transformations retained exactly as in the original code.
    ops.geomTransf('Linear', 2, 0.0, 1.0, 0.0)
    ops.geomTransf('Linear', 3, 1.0, 0.0, 0.0)

    # New transformation for vertical columns with strong-axis stiffness in X.
    # For a vertical element, vecxz=(0,1,0) makes global X response use column Iz.
    ops.geomTransf('Linear', 4, 0.0, 1.0, 0.0)

    col_data = []
    col_tag = 1

    for story in range(1, nStory + 1):
        if story == 1:
            lower_nodes = base_nodes
            upper_nodes = story_node_tags[1]
        else:
            lower_nodes = story_node_tags[story - 1]
            upper_nodes = story_node_tags[story]

        existing_cols = story_column_layout.get(story, [1, 2, 3, 4])

        for col_id in existing_cols:
            idx = col_id - 1
            iNode = lower_nodes[idx]
            jNode = upper_nodes[idx]
            orientation = column_orientation_layout.get(story, {}).get(col_id, "weak")
            col_data.append((col_tag, iNode, jNode, story, col_id, orientation))
            col_tag += 1

    for tag, iNode, jNode, _, _, orientation in col_data:
        transfTag = 1 if orientation == "weak" else 4
        ops.element('elasticBeamColumn', tag, iNode, jNode, A, E, G, J, Iy, Iz, transfTag)

    beam_data = []
    for story in range(1, nStory + 1):
        n1, n2, n3, n4 = story_node_tags[story]
        base_tag = story * 100
        beam_data.extend([
            (base_tag + 1, n1, n2, 2, story, "X-bottom"),
            (base_tag + 2, n4, n3, 2, story, "X-top"),
            (base_tag + 3, n1, n4, 3, story, "Y-left"),
            (base_tag + 4, n2, n3, 3, story, "Y-right"),
        ])

    for tag, iNode, jNode, transfTag, _, _ in beam_data:
        ops.element('elasticBeamColumn', tag, iNode, jNode, Ab, E, G, Jb, Iyb, Izb, transfTag)

    vis_elems = [(i, j) for _, i, j, _, _, _ in col_data] + [(i, j) for _, i, j, _, _, _ in beam_data]
    vis_nodes = base_nodes[:]
    for story in range(1, nStory + 1):
        vis_nodes.extend(story_node_tags[story])

    node_xyz = {nd: ops.nodeCoord(nd) for nd in vis_nodes}

    if show_info:
        print("\n=== MODEL SUMMARY ===")
        print("Stories:", nStory)
        print("Story heights:", story_heights)
        print("Self-weight floor masses:", floor_masses)
        print("Story column layout:", story_column_layout_to_text(story_column_layout, nStory))
        print("Column orientation layout:", column_orientation_layout_to_text(column_orientation_layout, nStory))
        print("Additional masses:", additional_masses_to_text(additional_masses, nStory))
        print("Mass assignment:", deep_round(mass_assignment))

    return {
        "z_levels": z_levels,
        "floor_weights": floor_weights,
        "floor_Jm": floor_Jm,
        "gravity_corner_masses_by_story": gravity_corner_masses_by_story,
        "base_nodes": base_nodes,
        "story_node_tags": story_node_tags,
        "master_nodes": master_nodes,
        "col_data": col_data,
        "beam_data": beam_data,
        "vis_elems": vis_elems,
        "vis_nodes": vis_nodes,
        "node_xyz": node_xyz,
        "mass_assignment": mass_assignment,
    }


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================
def run_gravity_analysis(ctx, show_info=False):
    story_node_tags = ctx["story_node_tags"]
    gravity_corner_masses_by_story = ctx.get("gravity_corner_masses_by_story", {})
    floor_weights = ctx.get("floor_weights", [])
    nStory = len(story_node_tags)

    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)

    # Gravity loads follow the same physical mass definition used for dynamics.
    # The self-weight mass and any extra center mass are shared equally by the
    # four slab corner nodes for vertical loading. Extra C1-C4 masses add their
    # own vertical load directly to the matching corner node.
    for story in range(1, nStory + 1):
        corner_masses = gravity_corner_masses_by_story.get(story, None)

        # Fallback for older saved input dictionaries.
        if corner_masses is None:
            Pcol = floor_weights[story - 1] / 4.0
            for nd in story_node_tags[story]:
                ops.load(nd, 0.0, 0.0, -Pcol, 0.0, 0.0, 0.0)
            continue

        for nd, m_corner in zip(story_node_tags[story], corner_masses):
            ops.load(nd, 0.0, 0.0, -m_corner * 9.81, 0.0, 0.0, 0.0)

    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('NormDispIncr', 1.0e-12, 50, 0)
    ops.algorithm('Newton')
    ops.integrator('LoadControl', 1.0)
    ops.analysis('Static')

    ok = ops.analyze(1)
    if ok != 0:
        raise RuntimeError("Static gravity analysis failed.")

    if show_info:
        print("Gravity analysis completed successfully.")

    ops.loadConst('-time', 0.0)


def run_eigen_with_fallback(num_modes, show_info=False):
    """
    Run OpenSees eigen analysis robustly.

    The default OpenSees eigensolver is ARPACK. It is fast, but it can fail for
    small models, repeated/clustered eigenvalues, or when too many modes are
    requested relative to the number of inertial DOFs. If ARPACK fails, this
    function retries with fullGenLapack, which is slower but more robust for the
    small student models used here.
    """
    try:
        return np.array(ops.eigen(num_modes), dtype=float)
    except Exception as arpack_error:
        if show_info:
            print("Default ARPACK eigen solver failed. Retrying with fullGenLapack...")
            print("ARPACK error:", arpack_error)

        lapack_errors = []
        for solver_name in ("-fullGenLapack", "fullGenLapack"):
            try:
                return np.array(ops.eigen(solver_name, num_modes), dtype=float)
            except Exception as e:
                lapack_errors.append(f"{solver_name}: {e}")

        raise RuntimeError(
            "OpenSees eigen analysis failed with both the default ARPACK solver and fullGenLapack. "
            "For a 2-story model, try setting 'Number of modes in model' = 2 and "
            "'Number of modes used in calibration' = 2. Details: "
            + " | ".join(lapack_errors)
        ) from arpack_error


def extract_modal_results(params, normalize_modes=True, show_info=False):
    ctx = build_model(params, show_info=show_info)
    run_gravity_analysis(ctx, show_info=show_info)

    lam = run_eigen_with_fallback(params["numModes"], show_info=show_info)
    if len(lam) < params["numModes"]:
        raise RuntimeError(
            f"OpenSees returned only {len(lam)} eigenvalues, but {params['numModes']} modes were requested. "
            "Reduce 'Number of modes in model'."
        )
    if not np.all(np.isfinite(lam)) or np.any(lam <= 0.0):
        raise RuntimeError(
            "OpenSees returned non-positive or non-finite eigenvalues. "
            "Check that all active stories have enough columns, positive mass, and stable boundary conditions."
        )

    omegas = np.sqrt(lam)
    freqs = omegas / (2.0 * math.pi)
    periods = 2.0 * math.pi / omegas

    master_nodes = ctx["master_nodes"]

    mode_shapes_ux_master = []
    for mode in range(1, params["numModes"] + 1):
        phi = np.array([ops.nodeEigenvector(node, mode, 1) for node in master_nodes], dtype=float)
        if normalize_modes:
            phi = normalize_mode_maxabs(phi)
        mode_shapes_ux_master.append(phi)

    return {
        "lam": lam,
        "omegas": omegas,
        "freqs": freqs,
        "periods": periods,
        "mode_shapes_ux_master": mode_shapes_ux_master,
        "ctx": ctx,
    }


def plot_modal_results_calibrated_only(modal_data, title_prefix="Calibrated", block=True):
    freqs = modal_data["freqs"]
    periods = modal_data["periods"]
    numModes = len(freqs)

    fig_modes = plt.figure(figsize=(4 * numModes, 5))
    try:
        fig_modes.canvas.manager.window.move(100, 800)
    except Exception:
        pass

    axes = [fig_modes.add_subplot(1, numModes, i + 1, projection='3d')
            for i in range(numModes)]

    for i, mode in enumerate(range(1, numModes + 1)):
        opsv.plot_mode_shape(mode, ax=axes[i], az_el=(-70, 25))

        axes[i].grid(False)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_zticks([])

        for coll in list(axes[i].collections):
            coll.remove()

        axes[i].set_title(
            f"{title_prefix} Mode {mode}\nT={periods[i]:.4f} s, f={freqs[i]:.3f} Hz"
        )

    plt.tight_layout()
    plt.show(block=block)


def export_modal_files(modal_data, output_prefix):
    lam = modal_data["lam"]
    omegas = modal_data["omegas"]
    freqs = modal_data["freqs"]
    periods = modal_data["periods"]
    ctx = modal_data["ctx"]
    master_nodes = ctx["master_nodes"]

    with open(f"output/{output_prefix}_periods.out", "w", encoding="utf-8") as f:
        f.write("Mode  Lambda(rad^2/s^2)  Omega(rad/s)  Frequency(Hz)  Period(s)\n")
        for i in range(len(freqs)):
            f.write(
                f"{i+1}  {r3(lam[i])}  {r3(omegas[i])}  {r3(freqs[i])}  {r3(periods[i])}\n"
            )

    with open(f"output/{output_prefix}_mode_shapes_normalized.out", "w", encoding="utf-8") as f:
        f.write("Mode  StoryMasterNode  UX_normalized\n")
        for mode in range(1, len(freqs) + 1):
            phi = modal_data["mode_shapes_ux_master"][mode - 1]
            for node, ux in zip(master_nodes, phi):
                f.write(f"{mode}  {node}  {r3(ux)}\n")


# =============================================================================
# CALIBRATION
# =============================================================================
def prepare_experimental_modal_data(params):
    loaded = load_experimental_modal_data(
        EXPERIMENTAL_MODAL_JSON,
        params["nStory"],
        require_mode_shapes=params["use_mode_shapes"]
    )

    if params["nCalibModes"] > len(loaded["frequencies_hz"]):
        raise ValueError(
            f"Experimental file contains only {len(loaded['frequencies_hz'])} frequencies, "
            f"but nCalibModes = {params['nCalibModes']}."
        )

    exp_freqs = loaded["frequencies_hz"][:params["nCalibModes"]]
    exp_modes = []
    mode_shapes_available = bool(loaded.get("mode_shapes_available", False))

    if params["use_mode_shapes"]:
        exp_modes = [normalize_mode_maxabs(v) for v in loaded["mode_shapes_ux"][:params["nCalibModes"]]]

    return {
        "freqs": exp_freqs,
        "modes": exp_modes,
        "use_mode_shapes": params["use_mode_shapes"],
        "mode_shapes_available": mode_shapes_available,
        "source_file": EXPERIMENTAL_MODAL_JSON,
        "raw_data": loaded["raw_data"],
        "n_modes_used": params["nCalibModes"]
    }

def apply_calibration_vector(base_params, x):
    p = copy.deepcopy(base_params)
    E_scale = float(x[0])
    mass_scales = np.array(x[1:], dtype=float)
    scope = base_params.get("mass_calibration_scope", "self_weight_only")

    p["E"] = base_params["E"] * E_scale

    # Each story gets one mass scale factor. The option selected in the
    # Calibration tab decides what that scale factor affects.
    #
    # self_weight_only:
    #   The optimizer changes only the structural/self-weight floor mass from
    #   the Basic Model tab. Additional masses remain exactly as entered.
    #
    # total_mass:
    #   The optimizer applies the same story mass scale to the self-weight mass
    #   and to all additional masses on that floor. This changes the total
    #   dynamic mass while keeping the additional-mass placement pattern.
    p["floor_masses"] = [
        base_params["floor_masses"][i] * float(mass_scales[i])
        for i in range(base_params["nStory"])
    ]

    if scope == "total_mass":
        additional = copy.deepcopy(base_params.get("additional_masses", {}))
        for story in range(1, base_params["nStory"] + 1):
            scale = float(mass_scales[story - 1])
            vals = additional.get(story, [0.0, 0.0, 0.0, 0.0, 0.0])
            additional[story] = [float(v) * scale for v in vals]
        p["additional_masses"] = additional

    return p

def modal_residuals(x, base_params, exp_data, w_freq=1.0, w_mode=0.35, show_info=False):
    try:
        p = apply_calibration_vector(base_params, x)
        modal = extract_modal_results(p, normalize_modes=True, show_info=False)

        n_use = exp_data["n_modes_used"]

        num_freqs = np.asarray(modal["freqs"][:n_use], dtype=float)
        exp_freqs = np.asarray(exp_data["freqs"][:n_use], dtype=float)

        res = []

        for fn, ft in zip(num_freqs, exp_freqs):
            res.append(w_freq * (fn - ft) / ft)

        if exp_data["use_mode_shapes"]:
            for i in range(n_use):
                phi_num = np.asarray(modal["mode_shapes_ux_master"][i], dtype=float)
                phi_exp = np.asarray(exp_data["modes"][i], dtype=float)
                a = align_mode_sign(phi_num, phi_exp)
                mode_diff = w_mode * (a - phi_exp)
                res.extend(list(mode_diff))

        return np.array(res, dtype=float)

    except Exception:
        if show_info:
            print("Calibration trial failed. Penalizing this point.")
            traceback.print_exc()
        n_mode_terms = exp_data["n_modes_used"] * base_params["nStory"] if exp_data["use_mode_shapes"] else 0
        return np.ones(exp_data["n_modes_used"] + n_mode_terms, dtype=float) * 1.0e3


def run_calibration(base_params, exp_data, show_info=False):
    nStory = base_params["nStory"]

    x0 = np.array([1.0] + [1.0] * nStory, dtype=float)
    lb = np.array([base_params["E_scale_lb"]] + [base_params["m_scale_lb"]] * nStory, dtype=float)
    ub = np.array([base_params["E_scale_ub"]] + [base_params["m_scale_ub"]] * nStory, dtype=float)

    loss_name = 'soft_l1'

    result = least_squares(
        modal_residuals,
        x0,
        bounds=(lb, ub),
        args=(base_params, exp_data, base_params["w_freq"], base_params["w_mode"], show_info),
        method='trf',
        loss=loss_name,
        max_nfev=base_params["max_nfev"],
        verbose=2 if show_info else 0
    )

    calib_params = apply_calibration_vector(base_params, result.x)
    return result, calib_params


def make_modal_comparison_report(exp_data, modal_before, modal_after, base_params, final_params, calib_result=None):
    n_use = exp_data["n_modes_used"]

    target_freqs = np.asarray(exp_data["freqs"][:n_use], dtype=float)
    before_freqs = np.asarray(modal_before["freqs"][:n_use], dtype=float)
    after_freqs = np.asarray(modal_after["freqs"][:n_use], dtype=float)

    report = {
        "experimental_data_source_file": exp_data.get("source_file", ""),
        "experimental_modal_data_raw": exp_data.get("raw_data", {}),
        "n_modes_used_in_calibration": n_use,
        "experimental_frequencies_hz": target_freqs.tolist(),
        "before_calibration_frequencies_hz": before_freqs.tolist(),
        "after_calibration_frequencies_hz": after_freqs.tolist(),
        "before_calibration_frequency_error_percent": [
            safe_percent_error(before_freqs[i], target_freqs[i]) for i in range(n_use)
        ],
        "after_calibration_frequency_error_percent": [
            safe_percent_error(after_freqs[i], target_freqs[i]) for i in range(n_use)
        ],
        "frequency_tolerance_percent": base_params["freq_tol_percent"],
        "before_calibration_within_tolerance": all(
            abs(safe_percent_error(before_freqs[i], target_freqs[i])) <= base_params["freq_tol_percent"]
            for i in range(n_use)
        ),
        "after_calibration_within_tolerance": all(
            abs(safe_percent_error(after_freqs[i], target_freqs[i])) <= base_params["freq_tol_percent"]
            for i in range(n_use)
        ),
        "original_E": base_params["E"],
        "calibrated_E": final_params["E"],
        "mass_calibration_scope": base_params.get("mass_calibration_scope", "self_weight_only"),
        "original_self_weight_floor_masses": base_params["floor_masses"],
        "calibrated_self_weight_floor_masses": final_params["floor_masses"],
        "column_orientation_layout": base_params.get("column_orientation_layout", {}),
        "original_additional_masses": base_params.get("additional_masses", {}),
        "calibrated_additional_masses": final_params.get("additional_masses", {}),
        "additional_masses": base_params.get("additional_masses", {}),
        "used_mode_shapes_in_calibration": bool(exp_data["use_mode_shapes"]),
    }

    mode_shape_comparison = []
    mode_shapes_can_compare = bool(exp_data.get("use_mode_shapes", False)) and len(exp_data.get("modes", [])) >= n_use

    if mode_shapes_can_compare:
        for i in range(n_use):
            phi_exp = np.asarray(exp_data["modes"][i], dtype=float)
            phi_before = align_mode_sign(np.asarray(modal_before["mode_shapes_ux_master"][i], dtype=float), phi_exp)
            phi_after = align_mode_sign(np.asarray(modal_after["mode_shapes_ux_master"][i], dtype=float), phi_exp)

            mode_shape_comparison.append({
                "mode": i + 1,
                "experimental_normalized": phi_exp.tolist(),
                "before_calibration_normalized": phi_before.tolist(),
                "after_calibration_normalized": phi_after.tolist(),
                "before_l2_mismatch": float(np.linalg.norm(phi_before - phi_exp)),
                "after_l2_mismatch": float(np.linalg.norm(phi_after - phi_exp)),
            })

    report["mode_shape_comparison_available"] = mode_shapes_can_compare
    report["mode_shape_comparison"] = mode_shape_comparison

    if calib_result is not None:
        report["optimizer"] = {
            "success": bool(calib_result.success),
            "status": int(calib_result.status),
            "message": str(calib_result.message),
            "nfev": int(calib_result.nfev),
            "cost": float(calib_result.cost),
            "x": np.asarray(calib_result.x, dtype=float).tolist(),
        }

    return report


def report_to_text(report):
    n_use = report["n_modes_used_in_calibration"]

    lines = []
    lines.append("MODAL CALIBRATION REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Experimental data source file: {report.get('experimental_data_source_file', '')}")
    lines.append(f"Number of modes used in calibration: {n_use}")
    lines.append("")

    lines.append("Experimental frequencies (Hz):")
    for i, f in enumerate(report["experimental_frequencies_hz"], start=1):
        lines.append(f"  Mode {i}: {r3(f)}")
    lines.append("")

    lines.append("Before calibration:")
    for i, (f, e) in enumerate(zip(
        report["before_calibration_frequencies_hz"],
        report["before_calibration_frequency_error_percent"]
    ), start=1):
        lines.append(f"  Mode {i}: f = {r3(f)} Hz, error = {r3(e)} %")
    lines.append("")

    lines.append("After calibration:")
    for i, (f, e) in enumerate(zip(
        report["after_calibration_frequencies_hz"],
        report["after_calibration_frequency_error_percent"]
    ), start=1):
        lines.append(f"  Mode {i}: f = {r3(f)} Hz, error = {r3(e)} %")
    lines.append("")

    lines.append(f"Tolerance used: {r3(report['frequency_tolerance_percent'])} %")
    lines.append(f"Before within tolerance: {report['before_calibration_within_tolerance']}")
    lines.append(f"After within tolerance:  {report['after_calibration_within_tolerance']}")
    lines.append(f"Used mode shapes in calibration: {report['used_mode_shapes_in_calibration']}")
    lines.append("")

    scope_text = "Total mass including additional masses" if report.get("mass_calibration_scope") == "total_mass" else "Self-weight mass only"
    lines.append("Parameter update:")
    lines.append(f"  Mass calibration target = {scope_text}")
    lines.append(f"  Original E   = {sci3(report['original_E'])}")
    lines.append(f"  Calibrated E = {sci3(report['calibrated_E'])}")
    lines.append(f"  Original self-weight masses   = {[r3(v) for v in report['original_self_weight_floor_masses']]}")
    lines.append(f"  Calibrated self-weight masses = {[r3(v) for v in report['calibrated_self_weight_floor_masses']]}")
    lines.append("")

    if report.get("column_orientation_layout"):
        nStory = len(report["original_self_weight_floor_masses"])
        lines.append("Column orientation layout:")
        lines.append(f"  {column_orientation_layout_to_text(report['column_orientation_layout'], nStory)}")
        lines.append("")

    if report.get("original_additional_masses"):
        nStory = len(report["original_self_weight_floor_masses"])
        lines.append("Original additional masses [center,c1,c2,c3,c4] kg:")
        lines.append(f"  {additional_masses_to_text(report['original_additional_masses'], nStory)}")
        lines.append("Calibrated additional masses [center,c1,c2,c3,c4] kg:")
        lines.append(f"  {additional_masses_to_text(report['calibrated_additional_masses'], nStory)}")
        lines.append("")

    if report.get("mode_shape_comparison_available", False):
        lines.append("Mode-shape comparison (normalized, max abs = 1):")
        for item in report["mode_shape_comparison"]:
            lines.append(f"  Mode {item['mode']}:")
            lines.append(f"    Experimental          = {[r3(v) for v in item['experimental_normalized']]}")
            lines.append(f"    Before calibration    = {[r3(v) for v in item['before_calibration_normalized']]}")
            lines.append(f"    After calibration     = {[r3(v) for v in item['after_calibration_normalized']]}")
            lines.append(f"    Before L2 mismatch    = {r3(item['before_l2_mismatch'])}")
            lines.append(f"    After L2 mismatch     = {r3(item['after_l2_mismatch'])}")
        lines.append("")
    else:
        lines.append("Mode-shape comparison was not produced because mode shapes were not used or were not available for this story count.")
        lines.append("")

    if "optimizer" in report:
        lines.append("Optimizer summary:")
        lines.append(f"  Success = {report['optimizer']['success']}")
        lines.append(f"  Status  = {report['optimizer']['status']}")
        lines.append(f"  Message = {report['optimizer']['message']}")
        lines.append(f"  nfev    = {report['optimizer']['nfev']}")
        lines.append(f"  Cost    = {r3(report['optimizer']['cost'])}")
        lines.append(f"  x       = {[r3(v) for v in report['optimizer']['x']]}")
        lines.append("")

    return "\n".join(lines)


def save_calibration_summary_figure(exp_data, modal_before, modal_after, original_params, calibrated_params,
                                    save_path="output/calibration_summary.png"):
    n_use = exp_data["n_modes_used"]
    stories = np.arange(1, original_params["nStory"] + 1)
    modes = np.arange(1, n_use + 1)

    # Use a non-interactive Agg Figure here so saving the calibration
    # summary does not briefly open a blank Matplotlib window during Calibrate.
    fig = Figure(figsize=(16, 11))
    canvas = FigureCanvas(fig)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(modes, exp_data["freqs"][:n_use], marker='o', linewidth=2.2, color='red', label="Target")
    ax1.plot(modes, modal_before["freqs"][:n_use], marker='s', linewidth=2.0, color='black', linestyle='--', label="Original")
    ax1.plot(modes, modal_after["freqs"][:n_use], marker='^', linewidth=2.0, color='green', linestyle='-.', label="Calibrated")
    ax1.set_title("Frequencies", fontsize=14)
    ax1.set_xlabel("Mode", fontsize=12)
    ax1.set_ylabel("Frequency (Hz)", fontsize=12)
    ax1.set_xticks(modes)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    ax2 = fig.add_subplot(2, 2, 2)
    x = np.arange(original_params["nStory"])
    w = 0.36

    def total_floor_masses(params):
        additional = params.get("additional_masses", {})
        return [
            params["floor_masses"][i] + sum(additional.get(i + 1, [0.0, 0.0, 0.0, 0.0, 0.0]))
            for i in range(params["nStory"])
        ]

    if original_params.get("mass_calibration_scope") == "total_mass":
        original_bar_masses = total_floor_masses(original_params)
        calibrated_bar_masses = total_floor_masses(calibrated_params)
        mass_plot_title = "Total Floor Masses"
    else:
        original_bar_masses = original_params["floor_masses"]
        calibrated_bar_masses = calibrated_params["floor_masses"]
        mass_plot_title = "Self-Weight Floor Masses"

    ax2.bar(x - w/2, original_bar_masses, width=w, color='black', label="Original")
    ax2.bar(x + w/2, calibrated_bar_masses, width=w, color='green', label="Calibrated")
    ax2.set_title(mass_plot_title, fontsize=14)
    ax2.set_xlabel("Story", fontsize=12)
    ax2.set_ylabel("Mass (kg)", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"M{i}" for i in range(1, original_params["nStory"] + 1)])
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.legend(fontsize=11)

    ax3 = fig.add_subplot(2, 2, 3)

    mode_shapes_can_plot = bool(exp_data.get("use_mode_shapes", False)) and len(exp_data.get("modes", [])) >= n_use

    if mode_shapes_can_plot:
        original_styles = ['-', '--', ':', '-.']
        target_styles = ['-', '--', ':', '-.']
        calibrated_styles = ['-', '--', ':', '-.']

        for i in range(n_use):
            phi_exp = np.asarray(exp_data["modes"][i], dtype=float)
            phi_before = align_mode_sign(np.asarray(modal_before["mode_shapes_ux_master"][i], dtype=float), phi_exp)
            phi_after = align_mode_sign(np.asarray(modal_after["mode_shapes_ux_master"][i], dtype=float), phi_exp)

            ax3.plot(phi_before, stories,
                     color='black', linestyle=original_styles[i % len(original_styles)],
                     marker='s', linewidth=1.8, label=f"Original M{i+1}")
            ax3.plot(phi_exp, stories,
                     color='red', linestyle=target_styles[i % len(target_styles)],
                     marker='o', linewidth=1.8, label=f"Target M{i+1}")
            ax3.plot(phi_after, stories,
                     color='green', linestyle=calibrated_styles[i % len(calibrated_styles)],
                     marker='^', linewidth=1.8, label=f"Calibrated M{i+1}")

        ax3.set_title("Normalized Mode Shapes (UX, max abs = 1)", fontsize=14)
        ax3.set_xlabel("Normalized amplitude", fontsize=12)
        ax3.set_ylabel("Story", fontsize=12)
        ax3.set_yticks(stories)
        ax3.grid(True, alpha=0.3)
        ax3.legend(ncol=3, fontsize=9)
    else:
        ax3.axis("off")
        ax3.set_title("Mode Shapes", fontsize=14)
        ax3.text(0.02, 0.95,
                 "Mode shapes were not used or are not available\nfor the selected number of stories.\n\nFrequencies are still used for calibration.",
                 va="top", ha="left", fontsize=12)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")

    txt = [
        "Calibration summary",
        "",
        f"Experimental file: {EXPERIMENTAL_MODAL_JSON}",
        f"Modes used in calibration: {n_use}",
        "",
        f"Original E:   {sci3(original_params['E'])}",
        f"Calibrated E: {sci3(calibrated_params['E'])}",
        "",
        "Mass calibration target:",
        "Total mass incl. additional" if original_params.get("mass_calibration_scope") == "total_mass" else "Self-weight mass only",
        "",
        f"Original self-weight masses:   {[r3(v) for v in original_params['floor_masses']]}",
        f"Calibrated self-weight masses: {[r3(v) for v in calibrated_params['floor_masses']]}",
        "",
        "Column orientation:",
        column_orientation_layout_to_text(original_params.get('column_orientation_layout', {}), original_params['nStory']),
        "",
        "Original additional masses [center,c1,c2,c3,c4] kg:",
        additional_masses_to_text(original_params.get('additional_masses', {}), original_params['nStory']),
        "",
        "Calibrated additional masses [center,c1,c2,c3,c4] kg:",
        additional_masses_to_text(calibrated_params.get('additional_masses', {}), calibrated_params['nStory']),
        "",
        "Frequency errors before (%):",
        f"{[r3(safe_percent_error(modal_before['freqs'][i], exp_data['freqs'][i])) for i in range(n_use)]}",
        "",
        "Frequency errors after (%):",
        f"{[r3(safe_percent_error(modal_after['freqs'][i], exp_data['freqs'][i])) for i in range(n_use)]}",
    ]
    ax4.text(0.02, 0.98, "\n".join(txt), va="top", ha="left", fontsize=10)

    fig.tight_layout()
    canvas.draw()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")


# =============================================================================
# TRANSIENT ANALYSIS
# =============================================================================
def set_rayleigh_damping_from_modal(modal_data, zeta, numModes):
    if numModes < 2:
        raise ValueError("At least 2 modes are required to compute Rayleigh damping.")

    lam = modal_data["lam"]
    w1 = math.sqrt(lam[0])
    w2 = math.sqrt(lam[1])

    alphaM = 2.0 * zeta * w1 * w2 / (w1 + w2)
    betaKinit = 2.0 * zeta / (w1 + w2)

    ops.rayleigh(alphaM, 0.0, betaKinit, 0.0)
    return alphaM, betaKinit


def setup_dynamic_excitation(params):
    gmFile = params["gmFile"]
    dtGM = params["dtGM"]
    gmFactor = params["gmFactor"]

    if not os.path.exists(gmFile):
        raise FileNotFoundError(f"Ground-motion file not found: {gmFile}")

    ops.timeSeries('Path', 20, '-dt', dtGM, '-filePath', gmFile, '-factor', gmFactor)
    ops.pattern('UniformExcitation', 20, 1, '-accel', 20)


def setup_recorders(ctx, prefix=""):
    master_nodes = ctx["master_nodes"]
    roof_master = master_nodes[-1]

    ops.recorder('Node', '-file', f'output/{prefix}time_floor_disp_X.out',
                 '-time', '-node', *master_nodes, '-dof', 1, 'disp')

    ops.recorder('Node', '-file', f'output/{prefix}time_floor_accel_X.out',
                 '-time', '-node', *master_nodes, '-dof', 1, 'accel')

    ops.recorder('Node', '-file', f'output/{prefix}time_roof_disp_X.out',
                 '-time', '-node', roof_master, '-dof', 1, 'disp')

    ops.recorder('Node', '-file', f'output/{prefix}time_roof_accel_X.out',
                 '-time', '-node', roof_master, '-dof', 1, 'accel')

    return roof_master


def get_deformed_xyz(node_tag, sfac=1.0):
    x, y, z = ops.nodeCoord(node_tag)
    ux = ops.nodeDisp(node_tag, 1)
    uy = ops.nodeDisp(node_tag, 2)
    uz = ops.nodeDisp(node_tag, 3)
    return x + sfac * ux, y + sfac * uy, z + sfac * uz


def run_transient_analysis_collect_data(params, modal_data, show_info=False, recorder_prefix=""):
    ctx = modal_data["ctx"]
    zeta = params["zeta"]
    gmFile = params["gmFile"]
    dtGM = params["dtGM"]

    alphaM, betaKinit = set_rayleigh_damping_from_modal(modal_data, zeta, params["numModes"])

    if show_info:
        print("Rayleigh damping:")
        print("alphaM    =", alphaM)
        print("betaKinit =", betaKinit)

    setup_dynamic_excitation(params)
    roof_master = setup_recorders(ctx, prefix=recorder_prefix)

    ops.wipeAnalysis()
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('NormDispIncr', 1.0e-10, 100, 0)
    ops.algorithm('Newton')
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')

    with open(gmFile, 'r', encoding="utf-8") as f:
        npts = sum(1 for line in f if line.strip())

    if npts < 2:
        raise ValueError("Ground motion file must contain at least 2 acceleration points.")

    nSteps = npts - 1
    t_hist = []
    u_hist = []
    a_hist = []
    ok = 0

    for i in range(nSteps):
        ok = ops.analyze(1, dtGM)

        if ok != 0:
            print(f"Transient analysis failed at step {i+1}, time = {ops.getTime()}")
            break

        t_hist.append(ops.getTime())
        u_hist.append(ops.nodeDisp(roof_master, 1))
        a_hist.append(ops.nodeAccel(roof_master, 1))

    if ok == 0:
        print("Transient analysis completed successfully.")
        print("Final roof displacement X =", r3(ops.nodeDisp(roof_master, 1)), "m")
    else:
        print("Transient analysis failed before completion.")

    return {
        "t_hist": np.array(t_hist, dtype=float),
        "u_hist": np.array(u_hist, dtype=float),
        "a_hist": np.array(a_hist, dtype=float),
    }


def run_transient_analysis_with_visualization(params, modal_data, show_info=False, overlay_response=None):
    ctx = modal_data["ctx"]
    zeta = params["zeta"]
    gmFile = params["gmFile"]
    dtGM = params["dtGM"]

    alphaM, betaKinit = set_rayleigh_damping_from_modal(modal_data, zeta, params["numModes"])

    if show_info:
        print("Rayleigh damping:")
        print("alphaM    =", alphaM)
        print("betaKinit =", betaKinit)

    setup_dynamic_excitation(params)
    roof_master = setup_recorders(ctx)

    plt.ion()

    sfac_anim = 20
    plot_every = 10
    anim_every = 10

    vis_elems = ctx["vis_elems"]
    vis_nodes = ctx["vis_nodes"]
    node_xyz = ctx["node_xyz"]

    with open(gmFile, 'r', encoding="utf-8") as f:
        npts = sum(1 for line in f if line.strip())

    if npts < 2:
        raise ValueError("Ground motion file must contain at least 2 acceleration points.")

    Tmax = (npts - 1) * dtGM
    nSteps = npts - 1

    # Figure 1 only: response plots + calibrated 3D animation.
    # Mode shapes are plotted later in a separate Figure 2.
    fig = plt.figure(figsize=(16, 7))
    try:
        fig.canvas.manager.window.move(100, 10)
    except Exception:
        pass

    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.4])

    ax1 = fig.add_subplot(gs[0, 0])
    line1, = ax1.plot([], [], color='red', linewidth=1.7, label='Calibrated')
    ax1.set_title("Roof Displacement", fontsize=12)
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Displacement (m)", fontsize=12)
    ax1.set_xlim(0.0, Tmax)
    ax1.grid(False)
    if overlay_response is not None:
        ax1.plot(
            overlay_response["t_hist"], overlay_response["u_hist"],
            color='black', linestyle='--', linewidth=0.7,
            label='Uncalibrated'
        )
        ax1.legend(fontsize=9, loc="upper right")

    ax2 = fig.add_subplot(gs[1, 0])
    line2, = ax2.plot([], [], color='red', linewidth=1.7, label='Calibrated')
    ax2.set_title("Roof Acceleration", fontsize=12)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Acceleration (m/s²)", fontsize=12)
    ax2.set_xlim(0.0, Tmax)
    ax2.grid(False)
    if overlay_response is not None:
        ax2.plot(
            overlay_response["t_hist"], overlay_response["a_hist"],
            color='black', linestyle='--', linewidth=0.7,
            label='Uncalibrated'
        )
        ax2.legend(fontsize=9, loc="upper right")

    ax_anim = fig.add_subplot(gs[:, 1], projection='3d')
    ax_anim.set_title("Calibrated Numerical Model 3D Response (OpenSees)")
    ax_anim.set_xlabel("X")
    ax_anim.set_ylabel("Y")
    ax_anim.set_zlabel("Z")

    ax_anim.grid(False)
    ax_anim.xaxis.pane.fill = False
    ax_anim.yaxis.pane.fill = False
    ax_anim.zaxis.pane.fill = False
    ax_anim.set_xticks([])
    ax_anim.set_yticks([])
    ax_anim.set_zticks([])
    ax_anim.view_init(elev=25, azim=-70)

    all_x = [node_xyz[n][0] for n in vis_nodes]
    all_y = [node_xyz[n][1] for n in vis_nodes]
    all_z = [node_xyz[n][2] for n in vis_nodes]

    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    zmin, zmax = min(all_z), max(all_z)

    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    zmid = 0.5 * (zmin + zmax)

    half = 0.55 * max(xmax - xmin, ymax - ymin, zmax - zmin)

    ax_anim.set_xlim(xmid - half, xmid + half)
    ax_anim.set_ylim(ymid - half, ymid + half)
    ax_anim.set_zlim(zmid - half, zmid + half)
    ax_anim.set_box_aspect((1, 1, 1))

    for n1, n2 in vis_elems:
        x1, y1, z1 = node_xyz[n1]
        x2, y2, z2 = node_xyz[n2]
        ax_anim.plot([x1, x2], [y1, y2], [z1, z2],
                     linestyle='--', linewidth=1.0, color='0.6')

    defo_lines = []
    for _ in vis_elems:
        ln, = ax_anim.plot([], [], [], color='navy', linewidth=3.0)
        defo_lines.append(ln)

    def update_live_model():
        for k, (n1, n2) in enumerate(vis_elems):
            x1, y1, z1 = get_deformed_xyz(n1, sfac_anim)
            x2, y2, z2 = get_deformed_xyz(n2, sfac_anim)
            defo_lines[k].set_data([x1, x2], [y1, y2])
            defo_lines[k].set_3d_properties([z1, z2])

    fig.tight_layout()

    ops.wipeAnalysis()
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('NormDispIncr', 1.0e-10, 100, 0)
    ops.algorithm('Newton')
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')

    t_hist = []
    u_hist = []
    a_hist = []

    ok = 0

    update_live_model()
    fig.canvas.draw()
    fig.canvas.flush_events()

    t0_wall = time.perf_counter()

    for i in range(nSteps):
        ok = ops.analyze(1, dtGM)

        if ok != 0:
            print(f"Transient analysis failed at step {i+1}, time = {ops.getTime()}")
            break

        t = ops.getTime()
        u = ops.nodeDisp(roof_master, 1)
        a = ops.nodeAccel(roof_master, 1)

        t_hist.append(t)
        u_hist.append(u)
        a_hist.append(a)

        if i % plot_every == 0:
            line1.set_data(t_hist, u_hist)
            ax1.relim()
            ax1.autoscale_view(scalex=False, scaley=True)

            line2.set_data(t_hist, a_hist)
            ax2.relim()
            ax2.autoscale_view(scalex=False, scaley=True)

        if i % anim_every == 0:
            update_live_model()

        if i % min(plot_every, anim_every) == 0:
            fig.canvas.draw()
            fig.canvas.flush_events()

        target_wall_time = t0_wall + t
        sleep_time = target_wall_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)

    if ok == 0:
        print("Transient analysis completed successfully.")
        print("Final roof displacement X =", r3(ops.nodeDisp(roof_master, 1)), "m")
    else:
        print("Transient analysis failed before completion.")

    plt.ioff()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================
def main():
    params = launch_input_window()

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    show_info = params["show_info"]

    if show_info:
        print("Current working directory:", os.getcwd())
        print("Excitation file exists:", os.path.exists(params["gmFile"]))
        print("Experimental modal JSON exists:", os.path.exists(EXPERIMENTAL_MODAL_JSON))

    use_precalibrated = bool(params.get("use_precalibrated", False))
    overlay_response = params.get("overlay_uncalibrated_response", None)

    run_params = copy.deepcopy(params)
    run_params.pop("gui_action", None)
    run_params.pop("use_precalibrated", None)
    run_params.pop("precalibrated_params", None)
    run_params.pop("overlay_uncalibrated_response", None)

    write_json("output/current_run_inputs.json", run_params)

    if use_precalibrated:
        print("\nUsing calibrated parameters stored by the Calibrate button.")
        final_params = copy.deepcopy(params["precalibrated_params"])
        write_json("output/calibrated_inputs_used_for_run.json", final_params)
    else:
        print("\nNo stored calibration was selected. Running the current input model.")
        final_params = run_params

    print("\nPreparing model for analysis...")
    final_modal = extract_modal_results(final_params, normalize_modes=True, show_info=show_info)
    export_modal_files(final_modal, "run_model")

    if params["run_transient"]:
        mode_title = "Calibrated" if use_precalibrated else "Run Model"
        print("\nShowing mode shapes...")
        plot_modal_results_calibrated_only(
            final_modal, title_prefix=mode_title, block=False
        )

        print("\nRunning transient analysis...")
        run_transient_analysis_with_visualization(
            final_params, final_modal, show_info=show_info,
            overlay_response=overlay_response
        )
    else:
        plot_modal_results_calibrated_only(final_modal, title_prefix="Run Model")

    ops.wipe()
    print("\nDone. Exported files are in the 'output' folder.")
    print("Main summary figure: output/calibration_summary.png")


if __name__ == "__main__":
    main()
