
import trimesh
import numpy as np

# ---------------- USER-ADJUSTABLE PARAMETERS (all in mm) ----------------
BREAKOUT_HOLE_X = 58.0     # VERIFY: spacing between breakout mounting holes, X
BREAKOUT_HOLE_Y = 49.0     # VERIFY: spacing between breakout mounting holes, Y
BREAKOUT_HOLE_DIA = 2.9    # M2.5 clearance

OLED_HOLE_SPACING = 22.0   # CONFIRMED: OLED's 4 mounting holes, 22x22mm square pattern
OLED_PILOT_HOLE_DIA = 1.7  # self-tapping pilot hole diameter for M2 screw into printed plastic
OLED_SCREW_LENGTH = 6.0    # M2 x 6mm screws
OLED_PCB_THICKNESS = 1.6   # ASSUMPTION: typical OLED PCB thickness -- verify with calipers
# v4: reduced from 5.0mm -> 4.0mm. The real screen (25x14mm, user-measured)
# is WIDER than the 22mm hole spacing, so the window now has to reach past
# the hole X-positions. A 5mm boss would collide with a window that size;
# 4mm keeps a positive clearance margin (see clearance check below) while
# still leaving (4.0 - 1.7) / 2 = 1.15mm of wall around the M2 pilot hole,
# which is workable for self-tapping into PLA/PETG.
OLED_BOSS_DIA = 4.0        # diameter of each downward boss the screw taps into

# How deep the screw engages into the bracket's boss material: total screw
# length, minus the OLED PCB it has to pass through first, minus a small
# safety margin so the screw tip doesn't poke through the bracket's top face.
OLED_BOSS_ENGAGE_DEPTH = OLED_SCREW_LENGTH - OLED_PCB_THICKNESS - 0.5  # ~3.9mm

# --- Window (screen cutout) geometry -- v4, user-measured screen size ---
# v3 used a photo-estimated 20x12mm window. The user then measured the
# actual screen directly: 25mm x 14mm. That's now the window size below.
# Because 25mm is WIDER than the 22mm hole spacing, the window necessarily
# extends past the hole X-positions -- there's no way around that if the
# window must expose the full screen. What matters is that it still clears
# the BOSSES (the printed screw posts), not the holes themselves, since the
# boss diameter can be sized down independently of the hole spacing.
#
# Clearance check (worst case: diagonal distance from each boss center at
# (OLED_HOLE_SPACING/2, OLED_HOLE_SPACING/2) = (11, 11) to the nearest
# window corner):
#   window corner = (WINDOW_WIDTH/2, WINDOW_Y_OFFSET + WINDOW_HEIGHT/2)
#     = (12.5, 1.5 + 7) = (12.5, 8.5)
#   distance = hypot(11-12.5, 11-8.5) = hypot(1.5, 2.5) = 2.92mm
#   clearance = distance - OLED_BOSS_DIA/2 (2.0mm) = 0.92mm  (must be > 0)
# ~0.9mm of margin -- tight but workable. If your slicer/printer needs more
# margin, shrink OLED_BOSS_DIA further (a 3mm boss gives ~1.4mm clearance,
# still fine for an M2 self-tapping screw) rather than shrinking the window.
WINDOW_WIDTH = 25.0        # mm, user-measured screen width
WINDOW_HEIGHT = 14.0       # mm, user-measured screen height
WINDOW_Y_OFFSET = 1.5      # mm, shifts window toward the header/+Y edge (still a
                           # photo-based estimate -- not corrected by the user, so
                           # kept as-is; verify with a dry-fit before printing)

PLATE_MARGIN = 6.0         # extra plate material beyond the breakout holes
PLATE_THICKNESS = 3.0
# --------------------------------------------------------------------

plate_length = BREAKOUT_HOLE_X + 2 * PLATE_MARGIN
plate_width = BREAKOUT_HOLE_Y + 2 * PLATE_MARGIN

plate = trimesh.creation.box(extents=[plate_length, plate_width, PLATE_THICKNESS])


def cyl(dia, height, x, y, z=0.0):
    c = trimesh.creation.cylinder(radius=dia / 2, height=height, sections=48)
    c.apply_translation([x, y, z])
    return c


# corner mounting holes (through the plate) for attaching to the breakout board
breakout_holes = [
    cyl(BREAKOUT_HOLE_DIA, PLATE_THICKNESS + 2, sx * BREAKOUT_HOLE_X / 2, sy * BREAKOUT_HOLE_Y / 2)
    for sx in (-1, 1) for sy in (-1, 1)
]
body = trimesh.boolean.difference([plate] + breakout_holes)

# Window cutout over the OLED screen -- v3: rectangular, off-center (shifted
# toward the header/+Y edge), sized to the real screen proportions instead
# of a centered square. See clearance math in the WINDOW_* comments above.
window = trimesh.creation.box(extents=[WINDOW_WIDTH, WINDOW_HEIGHT, PLATE_THICKNESS + 2])
window.apply_translation([0, WINDOW_Y_OFFSET, 0])
body = trimesh.boolean.difference([body, window])

# 4 downward-facing bosses on the underside of the bracket, at the OLED's
# hole spacing, for the M2x6 screws coming up from below through the OLED.
bosses = [
    cyl(OLED_BOSS_DIA, OLED_BOSS_ENGAGE_DEPTH, sx * OLED_HOLE_SPACING / 2, sy * OLED_HOLE_SPACING / 2,
        z=-(PLATE_THICKNESS / 2 + OLED_BOSS_ENGAGE_DEPTH / 2))
    for sx in (-1, 1) for sy in (-1, 1)
]
body = trimesh.util.concatenate([body] + bosses)

# self-tapping pilot holes through each boss
pilot_holes = [
    cyl(OLED_PILOT_HOLE_DIA, OLED_BOSS_ENGAGE_DEPTH + 2, sx * OLED_HOLE_SPACING / 2, sy * OLED_HOLE_SPACING / 2,
        z=-(PLATE_THICKNESS / 2 + OLED_BOSS_ENGAGE_DEPTH / 2))
    for sx in (-1, 1) for sy in (-1, 1)
]
body = trimesh.boolean.difference([body] + pilot_holes)

body.remove_unreferenced_vertices()
print("watertight:", body.is_watertight)
print("volume (mm^3):", round(body.volume, 1))
print("bounds (mm):", body.bounds.tolist())
print("window opening (mm):", WINDOW_WIDTH, "x", WINDOW_HEIGHT, "  y-offset:", WINDOW_Y_OFFSET)
print("boss engagement depth (mm):", round(OLED_BOSS_ENGAGE_DEPTH, 2))

# clearance sanity check (printed, not just commented)
boss_center = OLED_HOLE_SPACING / 2
corner = (WINDOW_WIDTH / 2, WINDOW_Y_OFFSET + WINDOW_HEIGHT / 2)
dist = ((boss_center - corner[0]) ** 2 + (boss_center - corner[1]) ** 2) ** 0.5
print("boss-to-window clearance (mm):", round(dist - OLED_BOSS_DIA / 2, 2), "(must be > 0)")

out_path = "oled_bracket.stl"
body.export(out_path)
print("wrote", out_path)
