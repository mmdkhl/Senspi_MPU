# -*- coding: utf-8 -*-
"""
Main GUI window for OpenSeesPy Model + Automatic Modal Calibration.

Extracted from launch_input_window() in DigitalTwin_V8.py.
"""

import os
import copy
import json
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

from ..io.loaders import EXPERIMENTAL_MODAL_JSON, write_json
from ..utils.formatters import deep_round
from ..calibration.calibrator import (
    prepare_experimental_modal_data,
    run_calibration,
)
from ..analysis.modal import extract_modal_results, export_modal_files
from ..analysis.transient import run_transient_analysis_collect_data
from ..reporting.calibration_report import (
    make_modal_comparison_report,
    report_to_text,
    save_calibration_summary_figure,
)
from ..io.loaders import write_text
from .widgets.canvas_widget import draw_corner_legend, draw_column_3d_sketch


def launch_input_window():
    """
    Launch the Tkinter input window and return the collected parameters.

    Returns
    -------
    dict : user-specified model and analysis parameters
    """
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
            "Story-dependent height, mass, columns-present checkboxes, column orientation, "
            "and mass-placement rows update from the Number of Stories field."
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
        try:
            lx = float(entries.get("Lx").get())
            ly = float(entries.get("Ly").get())
            if lx > 0.0 and ly > 0.0:
                return lx / ly
        except Exception:
            pass
        return 0.245 / 0.23

    def _draw_corner_legend(canvas):
        draw_corner_legend(canvas, lx_ly_ratio=_read_plan_ratio())

    def _draw_column_3d_sketch(canvas):
        draw_column_3d_sketch(
            canvas,
            n_story=get_int_story_count(),
            column_present_vars=column_present_vars,
            column_orient_vars=column_orient_vars,
            lx_ly_ratio=_read_plan_ratio(),
        )

    def update_column_presence_gui(story=None, col=None):
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
            _draw_column_3d_sketch(frame_canvas)
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

            ttk.Label(story_table_frame, text="Story", style="Small.TLabel").grid(row=0, column=0, padx=6, pady=4)
            ttk.Label(story_table_frame, text="Height (m)", style="Small.TLabel").grid(row=0, column=1, padx=6, pady=4)
            ttk.Label(story_table_frame, text="Self-weight floor mass (kg)", style="Small.TLabel").grid(row=0, column=2, padx=6, pady=4)
            ttk.Label(story_table_frame, text="Columns present", style="Small.TLabel").grid(row=0, column=3, columnspan=4, padx=6, pady=4)
            corner_headers = {1: "C1", 2: "C2", 3: "C3", 4: "C4"}
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

            ttk.Label(orientation_table_frame, text="Story", style="Small.TLabel").grid(row=0, column=0, padx=6, pady=4)
            orient_corner_headers = {
                1: "C1\n(0,0)",
                2: "C2\n(Lx,0)",
                3: "C3\n(Lx,Ly)",
                4: "C4\n(0,Ly)",
            }
            for c in (1, 2, 3, 4):
                ttk.Label(orientation_table_frame, text=orient_corner_headers[c], style="Small.TLabel", justify="center").grid(row=0, column=c, padx=6, pady=4)

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
    # Tab 1: Basic model
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
    story_spin = ttk.Spinbox(geom, from_=1, to=20, textvariable=nStory_var, width=8,
                              command=schedule_story_table_rebuild)
    story_spin.grid(row=r, column=1, sticky="w", padx=(0, 10), pady=(8, 1))
    ttk.Label(
        geom,
        text="Automatically updates story rows and safe mode counts.",
        style="Hint.TLabel", wraplength=620, justify="left"
    ).grid(row=r + 1, column=0, columnspan=2, sticky="w", padx=(10, 10), pady=(0, 6))
    r += 2

    ttk.Label(
        geom,
        text="Columns present are selected below using C1-C4 checkboxes.",
        style="Hint.TLabel", wraplength=620, justify="left"
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

    frame_sketch_box = ttk.LabelFrame(tab1_right, text="3D Column Numbering Sketch",
                                       style="Section.TLabelframe", padding=(10, 8))
    frame_sketch_box.pack(fill="both", expand=True, pady=0, padx=0, anchor="n")
    frame_canvas = tk.Canvas(frame_sketch_box, width=350, height=300, bg="white", highlightthickness=0)
    frame_canvas.pack(anchor="center", fill="both", expand=True, padx=4, pady=4)
    _draw_column_3d_sketch(frame_canvas)

    def refresh_basic_sketch(*args):
        try:
            _draw_column_3d_sketch(frame_canvas)
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

    story_box = ttk.LabelFrame(tab1_bottom,
                                text="Story Height + Self-Weight Floor Mass + Columns Present",
                                style="Section.TLabelframe", padding=(10, 8))
    story_box.pack(side="left", fill="both", expand=True, padx=(0, 6), anchor="n")
    story_table_frame = ttk.Frame(story_box)
    story_table_frame.pack(anchor="w", padx=4, pady=4)

    orient_box = ttk.LabelFrame(tab1_bottom, text="Column Orientation for X Shaking",
                                 style="Section.TLabelframe", padding=(10, 8))
    orient_box.pack(side="left", fill="both", expand=True, padx=(6, 0), anchor="n")
    orient_controls = ttk.Frame(orient_box)
    orient_controls.pack(fill="x", pady=(0, 4))
    ttk.Button(orient_controls, text="Set all Weak axis",
               command=lambda: set_all_column_orientation("Weak axis")).pack(side="left", padx=(4, 6))
    ttk.Button(orient_controls, text="Set all Strong axis",
               command=lambda: set_all_column_orientation("Strong axis")).pack(side="left", padx=(0, 6))
    orientation_table_frame = ttk.Frame(orient_box)
    orientation_table_frame.pack(anchor="w", padx=4, pady=4)

    # -------------------------------------------------------------------------
    # Tab 2: Material/mass placement and analysis
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
        style="Hint.TLabel", wraplength=480, justify="left"
    ).grid(row=r, column=0, columnspan=2, sticky="w", padx=(10, 10), pady=(2, 6))
    r += 1

    mass_legend_canvas = tk.Canvas(mat, width=420, height=220, bg="white", highlightthickness=0)
    mass_legend_canvas.grid(row=r, column=0, columnspan=2, sticky="w", padx=(10, 10), pady=(0, 6))
    _draw_corner_legend(mass_legend_canvas)
    mass_legend_canvas.bind("<Configure>", lambda event: _draw_corner_legend(mass_legend_canvas))
    try:
        entries["Lx"].bind("<FocusOut>", lambda event: _draw_corner_legend(mass_legend_canvas), add="+")
        entries["Ly"].bind("<FocusOut>", lambda event: _draw_corner_legend(mass_legend_canvas), add="+")
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
    r = add_check(ana, r, "Run transient after calibration", "run_transient", True,
                  "If unchecked, only modal analysis and calibration are performed.")

    mass_box = ttk.LabelFrame(tab2_body, text="Additional Mass Placement (kg)",
                               style="Section.TLabelframe", padding=(10, 8))
    mass_box.pack(fill="x", expand=False, pady=(0, 8), padx=8, anchor="n")
    floor_mass_table_frame = ttk.Frame(mass_box)
    floor_mass_table_frame.pack(anchor="w", padx=4, pady=4)

    # -------------------------------------------------------------------------
    # Tab 3: Calibration
    # -------------------------------------------------------------------------
    calib_container = ttk.Frame(tab3)
    calib_container.pack(fill="x", expand=False, pady=8, padx=8)

    calib_left = ttk.LabelFrame(calib_container, text="Calibration Settings",
                                  style="Section.TLabelframe", padding=(10, 8))
    calib_left.pack(side="left", fill="both", expand=True, padx=(0, 6), anchor="n")

    calib_right = ttk.LabelFrame(calib_container, text="Optimizer Bounds",
                                   style="Section.TLabelframe", padding=(10, 8))
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
        style="Hint.TLabel", wraplength=480, justify="left"
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
                        "Lower bound for each story mass scale.")
    r_right = add_field(calib_right, r_right, "Upper bound for mass scales", "m_scale_ub", "1.30",
                        "Upper bound for each story mass scale.")
    r_right = add_field(calib_right, r_right, "Maximum optimizer evaluations", "max_nfev", "200",
                        "Larger values may improve fit but increase runtime.")
    r_right = add_check(calib_right, r_right, "Show uncalibrated response on run plot", "show_uncalibrated_response", True,
                        "After using Calibrate, overlays the stored uncalibrated roof response on the calibrated run plot.")

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
                            "There is a problem during calibration.\n" + msg
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
