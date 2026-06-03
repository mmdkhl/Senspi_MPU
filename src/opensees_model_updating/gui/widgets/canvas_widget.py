# -*- coding: utf-8 -*-
"""
GUI canvas drawing utilities.

Contains draw_corner_legend() and draw_column_3d_sketch() from DigitalTwin_V8.py.
"""

import math
import tkinter as tk


def draw_corner_legend(canvas, lx_ly_ratio=None):
    """
    Draw a clear plan-view sketch for C1-C4 mass-placement labels.

    Parameters
    ----------
    canvas : tk.Canvas
        Target canvas widget.
    lx_ly_ratio : float or None
        Lx/Ly ratio for plan shape. Defaults to 0.245/0.23.
    """
    canvas.delete("all")
    w = max(int(canvas.cget("width")), canvas.winfo_width())
    h = max(int(canvas.cget("height")), canvas.winfo_height())

    if lx_ly_ratio is None:
        ratio = 0.245 / 0.23
    else:
        ratio = lx_ly_ratio

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

    canvas.create_rectangle(left, top, right, bottom, width=2, outline="#1f2d3d")
    canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="#1f2d3d", outline="#1f2d3d")
    canvas.create_text(cx, cy - 15, text="Extra center mass", font=("Segoe UI", 8), fill="#1f2d3d")

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


def draw_column_3d_sketch(canvas, n_story=3, column_present_vars=None,
                           column_orient_vars=None, lx_ly_ratio=None):
    """
    Draw a compact color-coded 3D frame sketch showing column numbering C1-C4.

    Parameters
    ----------
    canvas : tk.Canvas
        Target canvas widget.
    n_story : int
        Number of stories.
    column_present_vars : dict or None
        {story: {col_id: tk.BooleanVar}} for column presence.
    column_orient_vars : dict or None
        {story: {col_id: tk.StringVar}} for column orientation.
    lx_ly_ratio : float or None
        Lx/Ly ratio for plan shape.
    """
    canvas.delete("all")
    w = max(int(canvas.cget("width")), canvas.winfo_width())
    h = max(int(canvas.cget("height")), canvas.winfo_height())

    if lx_ly_ratio is None:
        ratio = 0.245 / 0.23
    else:
        ratio = max(0.85, min(1.15, lx_ly_ratio))

    n_show = max(1, min(20, n_story))

    top_margin = 44
    bottom_margin = 70
    available_h = max(120, h - top_margin - bottom_margin)
    story_h = min(50, available_h / max(n_show, 1))
    story_h = max(8, story_h)

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
        "C1": "#2563eb",
        "C2": "#dc2626",
        "C3": "#16a34a",
        "C4": "#9333ea",
    }

    for k in range(0, n_show + 1):
        p1, p2, p3, p4 = up(c1, k), up(c2, k), up(c3, k), up(c4, k)
        fill = "#eef4ff" if k == n_show else "#f8fafc"
        outline = "#64748b" if k in (0, n_show) else "#cbd5e1"
        canvas.create_polygon(
            p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1],
            fill=fill, outline=outline, width=2 if k in (0, n_show) else 1
        )

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
                if column_present_vars is not None:
                    present = column_present_vars[story][col_id].get()
            except Exception:
                present = True

            p_bot = up(base_pt, story - 1)
            p_top = up(base_pt, story)

            if present:
                try:
                    orient = column_orient_vars[story][col_id].get() if column_orient_vars else "Weak axis"
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
        canvas.create_text(x + dx, y_text, text=lab, font=("Segoe UI", 10, "bold"),
                           fill=column_colors[key], anchor=anchor)

    ax0 = (w - 90, h - 20)
    y_len = 44.0
    y_norm = math.sqrt(depth_x**2 + depth_y**2)
    y_arrow = (depth_x / y_norm * y_len, -depth_y / y_norm * y_len)
    canvas.create_line(ax0[0], ax0[1], ax0[0] + 44, ax0[1], arrow=tk.LAST, width=2, fill="#1f2d3d")
    canvas.create_text(ax0[0] + 52, ax0[1], text="X", anchor="w", font=("Segoe UI", 9), fill="#1f2d3d")
    canvas.create_line(ax0[0], ax0[1], ax0[0] + y_arrow[0], ax0[1] + y_arrow[1], arrow=tk.LAST, width=2, fill="#1f2d3d")
    canvas.create_text(ax0[0] + y_arrow[0] + 6, ax0[1] + y_arrow[1] - 2, text="Y", anchor="w",
                       font=("Segoe UI", 9), fill="#1f2d3d")
    canvas.create_line(ax0[0], ax0[1], ax0[0], ax0[1] - 44, arrow=tk.LAST, width=2, fill="#1f2d3d")
    canvas.create_text(ax0[0], ax0[1] - 54, text="Z", anchor="center", font=("Segoe UI", 9), fill="#1f2d3d")
