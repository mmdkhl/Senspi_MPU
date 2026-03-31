"""
3D 1-bay, N-story aluminum frame in OpenSeesPy
Units: N, m, sec

Modeling notes
- Floors act as rigid diaphragms in the XY plane.
- Floor masses are lumped at diaphragm master nodes.
- Dynamic excitation is applied in global X using UniformExcitation.
"""

%matplotlib qt

import os
import math
import time
import matplotlib.pyplot as plt
import opsvis as opsv
import openseespy.opensees as ops


# =============================================================================
# 0. INPUTS
# =============================================================================
# Turn summary printing on/off
show_info = False

# Files
gmFile = "input/sine_1Hz_accel.txt"

# Geometry
Lx, Ly = 0.245, 0.23
story_heights = [0.24, 0.24, 0.24]   # one value per story
nStory = len(story_heights)

# Material
E = 200e9
nu = 0.33

# Net Sections at hole center
# Columns
t_column = 0.001
b_column = 0.006   # net width = total width - hole diameter = 10 mm - 4 mm = 6 mm

# Beams
b_beam = 0.001
h_beam = 0.008   # net height = total height - hole diameter = 12 mm - 4 mm = 8 mm


# Floor masses (kg), one value per story
floor_masses = [0.3, 0.5, 0.3]


# Modal analysis
numModes = 4

# Rayleigh damping
zeta = 0.005   # 0.5%

# Dynamic excitation
dtGM = 0.01
gmFactor = 9.81

# Visualization
sfac_anim = 20
plot_every = 10
anim_every = 10


# =============================================================================
# 1. BASIC CHECKS + SUMMARY
# =============================================================================
if len(floor_masses) != nStory:
    raise ValueError("floor_masses must have the same length as story_heights.")

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

if show_info:
    print("Current working directory:", os.getcwd())
    print("Excitation file exists:", os.path.exists(gmFile))
    print("Number of stories =", nStory)
    print("Story heights =", story_heights)
    print("Floor masses =", floor_masses)


# =============================================================================
# 2. MODEL
# =============================================================================
ops.wipe()
ops.model('Basic', '-ndm', 3, '-ndf', 6)


# =============================================================================
# 3. GEOMETRY
# =============================================================================
z_levels = [0.0]
for h in story_heights:
    z_levels.append(z_levels[-1] + h)


# =============================================================================
# 4. MATERIAL
# =============================================================================
G = E / (2.0 * (1.0 + nu))


# =============================================================================
# 5. SECTIONS
# =============================================================================
# Columns
A = t_column * b_column
Iy = b_column * t_column**3 / 12.0
Iz = t_column * b_column**3 / 12.0
J = (1.0 / 3.0) * b_column * t_column**3

# Beams
Ab = b_beam * h_beam
Iyb = h_beam * b_beam**3 / 12.0
Izb = b_beam * h_beam**3 / 12.0
Jb = (1.0 / 3.0) * h_beam * b_beam**3

if show_info:
    print("Column A  =", A)
    print("Column Iy =", Iy)
    print("Column Iz =", Iz)
    print("Column J  =", J)
    print("Beam Ab  =", Ab)
    print("Beam Iyb =", Iyb)
    print("Beam Izb =", Izb)
    print("Beam Jb  =", Jb)


# =============================================================================
# 6. FLOOR MASS
# =============================================================================
floor_weights = [m * 9.81 for m in floor_masses]
floor_Jm = [m * (Lx**2 + Ly**2) / 12.0 for m in floor_masses]

if show_info:
    for k in range(nStory):
        print(f"Floor {k+1} mass (kg) =", floor_masses[k])
        print(f"Floor {k+1} weight (N) =", floor_weights[k])
        print(f"Floor {k+1} Jm (kg m^2) =", floor_Jm[k])


# =============================================================================
# 7. NODES
# =============================================================================
# Base
base_nodes = [1, 2, 3, 4]
base_coords = [
    (0.0, 0.0, 0.0),
    (Lx,  0.0, 0.0),
    (Lx,  Ly,  0.0),
    (0.0, Ly,  0.0),
]

for nd, (x, y, z) in zip(base_nodes, base_coords):
    ops.node(nd, x, y, z)

# Story nodes and master nodes
story_node_tags = {}
master_nodes = []

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


# =============================================================================
# 8. BOUNDARY CONDITIONS
# =============================================================================
# Fully fixed base
for nd in base_nodes:
    ops.fix(nd, 1, 1, 1, 1, 1, 1)

# Master nodes keep in-plane DOFs (UX, UY, RZ)
for master in master_nodes:
    ops.fix(master, 0, 0, 1, 1, 1, 0)


# =============================================================================
# 9. RIGID DIAPHRAGMS
# =============================================================================
# Floors are rigid in the XY plane (perpendicular direction = Z)
for story in range(1, nStory + 1):
    ops.rigidDiaphragm(3, 1000 + story, *story_node_tags[story])


# =============================================================================
# 10. MASSES
# =============================================================================
# Lumped translational masses in X/Y and rotational inertia about Z
for story in range(1, nStory + 1):
    ops.mass(1000 + story,
             floor_masses[story - 1], floor_masses[story - 1], 0.0,
             0.0, 0.0, floor_Jm[story - 1])


# =============================================================================
# 11. GEOMETRIC TRANSFORMATIONS
# =============================================================================
ops.geomTransf('Linear', 1, 1.0, 0.0, 0.0)  # columns
ops.geomTransf('Linear', 2, 0.0, 1.0, 0.0)  # beams along X
ops.geomTransf('Linear', 3, 1.0, 0.0, 0.0)  # beams along Y


# =============================================================================
# 12. ELEMENTS
# =============================================================================
col_data = []
col_tag = 1

# Columns from base to first story
for i in range(4):
    col_data.append((col_tag, base_nodes[i], story_node_tags[1][i]))
    col_tag += 1

# Columns between upper stories
for story in range(1, nStory):
    lower_nodes = story_node_tags[story]
    upper_nodes = story_node_tags[story + 1]
    for i in range(4):
        col_data.append((col_tag, lower_nodes[i], upper_nodes[i]))
        col_tag += 1

for tag, iNode, jNode in col_data:
    ops.element('elasticBeamColumn', tag, iNode, jNode, A, E, G, J, Iy, Iz, 1)

beam_data = []
for story in range(1, nStory + 1):
    n1, n2, n3, n4 = story_node_tags[story]
    base_tag = story * 100
    beam_data.extend([
        (base_tag + 1, n1, n2, 2),
        (base_tag + 2, n4, n3, 2),
        (base_tag + 3, n1, n4, 3),
        (base_tag + 4, n2, n3, 3),
    ])

for tag, iNode, jNode, transfTag in beam_data:
    ops.element('elasticBeamColumn', tag, iNode, jNode, Ab, E, G, Jb, Iyb, Izb, transfTag)


# =============================================================================
# 13. GRAVITY LOAD
# =============================================================================
ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)

for story in range(1, nStory + 1):
    Pcol = floor_weights[story - 1] / 4.0
    for nd in story_node_tags[story]:
        ops.load(nd, 0.0, 0.0, -Pcol, 0.0, 0.0, 0.0)


# =============================================================================
# 14. STATIC ANALYSIS
# =============================================================================
ops.constraints('Transformation')
ops.numberer('RCM')
ops.system('BandGeneral')
ops.test('NormDispIncr', 1.0e-12, 50, 0)
ops.algorithm('Newton')
ops.integrator('LoadControl', 1.0)
ops.analysis('Static')

ok = ops.analyze(1)

if ok == 0 and show_info:
    ops.reactions()
    print("Gravity analysis completed successfully.")
    print("Roof corner vertical displacement =", ops.nodeDisp(story_node_tags[nStory][0], 3), "m")
    print("Base reaction at node 1 in Z =", ops.nodeReaction(1, 3), "N")
elif ok != 0:
    print("Static analysis failed.")

ops.loadConst('-time', 0.0)


# =============================================================================
# 15. MODAL ANALYSIS
# =============================================================================
lam = ops.eigen('-fullGenLapack', numModes)

fig_modes = plt.figure(figsize=(16, 5))
fig_modes.canvas.manager.window.move(100, 800)
axes = [fig_modes.add_subplot(1, numModes, i + 1, projection='3d') for i in range(numModes)]

for i, mode in enumerate(range(1, numModes + 1)):
    opsv.plot_mode_shape(mode, ax=axes[i], az_el=(-70, 25))

    axes[i].grid(False)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].set_zticks([])

    # Remove markers/collections added by opsvis
    for coll in list(axes[i].collections):
        coll.remove()

    omega = math.sqrt(lam[i])
    freq = omega / (2.0 * math.pi)
    T = 2.0 * math.pi / omega
    axes[i].set_title(f"Mode {mode}\nT={T:.4f} s, f={freq:.3f} Hz")

plt.tight_layout()
plt.show()

# Save modal results
with open("output/periods.out", "w") as f:
    f.write("Mode  Lambda(rad^2/s^2)  Omega(rad/s)  Frequency(Hz)  Period(s)\n")
    for i in range(numModes):
        omega = math.sqrt(lam[i])
        freq = omega / (2.0 * math.pi)
        T = 2.0 * math.pi / omega
        f.write(f"{i+1}  {lam[i]}  {omega}  {freq}  {T}\n")

with open("output/mode_shapes.out", "w") as f:
    f.write("Mode  Node  UX  UY  UZ  RX  RY  RZ\n")
    for mode in range(1, numModes + 1):
        for node in master_nodes:
            ux = ops.nodeEigenvector(node, mode, 1)
            uy = ops.nodeEigenvector(node, mode, 2)
            uz = ops.nodeEigenvector(node, mode, 3)
            rx = ops.nodeEigenvector(node, mode, 4)
            ry = ops.nodeEigenvector(node, mode, 5)
            rz = ops.nodeEigenvector(node, mode, 6)
            f.write(f"{mode}  {node}  {ux}  {uy}  {uz}  {rx}  {ry}  {rz}\n")


# =============================================================================
# 16. RAYLEIGH DAMPING
# =============================================================================
# 0.5% damping based on the first two modes
w1 = math.sqrt(lam[0])
w2 = math.sqrt(lam[1])

alphaM = 2.0 * zeta * w1 * w2 / (w1 + w2)
betaKinit = 2.0 * zeta / (w1 + w2)

ops.rayleigh(alphaM, 0.0, betaKinit, 0.0)

if show_info:
    print("Rayleigh damping:")
    print("alphaM    =", alphaM)
    print("betaKinit =", betaKinit)


# =============================================================================
# 17. DYNAMIC EXCITATION
# =============================================================================
# Base acceleration input in global X
ops.timeSeries('Path', 20, '-dt', dtGM, '-filePath', gmFile, '-factor', gmFactor)
ops.pattern('UniformExcitation', 20, 1, '-accel', 20)


# =============================================================================
# 18. RECORDERS
# =============================================================================
roof_master = master_nodes[-1]

# Relative response histories in global X
ops.recorder('Node', '-file', 'output/time_floor_disp_X.out',
             '-time', '-node', *master_nodes, '-dof', 1, 'disp')

ops.recorder('Node', '-file', 'output/time_floor_accel_X.out',
             '-time', '-node', *master_nodes, '-dof', 1, 'accel')

ops.recorder('Node', '-file', 'output/time_roof_disp_X.out',
             '-time', '-node', roof_master, '-dof', 1, 'disp')

ops.recorder('Node', '-file', 'output/time_roof_accel_X.out',
             '-time', '-node', roof_master, '-dof', 1, 'accel')


# =============================================================================
# 19. VISUALIZATION DATA + HELPER FUNCTIONS
# =============================================================================
plt.ion()  # interactive mode for live updating

# Elements to draw in the 3D view:
# - all columns
# - all beams
vis_elems = [(i, j) for _, i, j in col_data] + [(i, j) for _, i, j, _ in beam_data]

# Structural nodes to visualize
vis_nodes = base_nodes[:]
for story in range(1, nStory + 1):
    vis_nodes.extend(story_node_tags[story])

# Original nodal coordinates (undeformed geometry)
node_xyz = {nd: ops.nodeCoord(nd) for nd in vis_nodes}

# Scale factor only for visualization of deformation
# (does not affect analysis results)

def get_deformed_xyz(node_tag, sfac=1.0):
    """
    Return deformed coordinates of a node.
    Original coordinates are shifted by the current nodal displacements.
    """
    x, y, z = ops.nodeCoord(node_tag)
    ux = ops.nodeDisp(node_tag, 1)
    uy = ops.nodeDisp(node_tag, 2)
    uz = ops.nodeDisp(node_tag, 3)
    return x + sfac * ux, y + sfac * uy, z + sfac * uz

def update_live_model():
    """
    Update the plotted deformed frame using current displacements
    from the transient analysis step.
    """
    for k, (n1, n2) in enumerate(vis_elems):
        x1, y1, z1 = get_deformed_xyz(n1, sfac_anim)
        x2, y2, z2 = get_deformed_xyz(n2, sfac_anim)
        defo_lines[k].set_data([x1, x2], [y1, y2])
        defo_lines[k].set_3d_properties([z1, z2])


# =============================================================================
# 20. TRANSIENT ANALYSIS + SINGLE WINDOW VISUALIZATION
#
# Layout:
# - left top    : roof displacement time history
# - left bottom : roof acceleration time history
# - right       : live 3D structural response
# =============================================================================
ops.wipeAnalysis()
ops.constraints('Transformation')
ops.numberer('RCM')
ops.system('BandGeneral')
ops.test('NormDispIncr', 1.0e-10, 100, 0)
ops.algorithm('Newton')
ops.integrator('Newmark', 0.5, 0.25)
ops.analysis('Transient')

# Total analysis time and number of time steps (calculate automaically)
with open(gmFile, 'r') as f:
    npts = sum(1 for line in f if line.strip())  # number of acceleration points

Tmax = (npts - 1) * dtGM
nSteps = npts - 1


# -------------------------------------------------------------------------
# Create one figure containing all live outputs
# -------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 7))
fig.canvas.manager.window.move(100, 10)

# 2 rows x 2 columns:
# left column  -> two 2D plots
# right column -> one 3D animation spanning both rows
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.4])

# -------------------------------------------------------------------------
# Left-top panel: roof displacement history in global X
# -------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
line1, = ax1.plot([], [], color='red')
ax1.set_title("Roof Displacement",fontsize=12)
ax1.set_xlabel("Time (s)",fontsize=12)
ax1.set_ylabel("Displacement (m)",fontsize=12)
ax1.set_xlim(0.0, Tmax)
ax1.grid(False)   # no grid background

# -------------------------------------------------------------------------
# Left-bottom panel: roof acceleration history in global X
# -------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[1, 0])
line2, = ax2.plot([], [], color='red')
ax2.set_title("Roof Acceleration",fontsize=12)
ax2.set_xlabel("Time (s)",fontsize=12)
ax2.set_ylabel("Acceleration (m/s²)",fontsize=12)
ax2.set_xlim(0.0, Tmax)
ax2.grid(False)   # no grid background

# -------------------------------------------------------------------------
# Right panel: 3D structural model and live deformed shape
# -------------------------------------------------------------------------
ax_anim = fig.add_subplot(gs[:, 1], projection='3d')
ax_anim.set_title("Numerical model 3D Response (OpenSees)")
ax_anim.set_xlabel("X")
ax_anim.set_ylabel("Y")
ax_anim.set_zlabel("Z")

# Clean 3D appearance
ax_anim.grid(False)
ax_anim.xaxis.pane.fill = False
ax_anim.yaxis.pane.fill = False
ax_anim.zaxis.pane.fill = False
ax_anim.set_xticks([])
ax_anim.set_yticks([])
ax_anim.set_zticks([])
ax_anim.view_init(elev=25, azim=-70)

# -------------------------------------------------------------------------
# Set fixed 3D axis limits from undeformed geometry
# This avoids axis resizing during animation
# -------------------------------------------------------------------------
all_x = [node_xyz[n][0] for n in vis_nodes]
all_y = [node_xyz[n][1] for n in vis_nodes]
all_z = [node_xyz[n][2] for n in vis_nodes]

xmin, xmax = min(all_x), max(all_x)
ymin, ymax = min(all_y), max(all_y)
zmin, zmax = min(all_z), max(all_z)

xmid = 0.5 * (xmin + xmax)
ymid = 0.5 * (ymin + ymax)
zmid = 0.5 * (zmin + zmax)

# Use the largest structural dimension to keep the 3D view proportional
half = 0.55 * max(xmax - xmin, ymax - ymin, zmax - zmin)

ax_anim.set_xlim(xmid - half, xmid + half)
ax_anim.set_ylim(ymid - half, ymid + half)
ax_anim.set_zlim(zmid - half, zmid + half)
ax_anim.set_box_aspect((1, 1, 1))

# -------------------------------------------------------------------------
# Plot undeformed structure as a reference
# -------------------------------------------------------------------------
for n1, n2 in vis_elems:
    x1, y1, z1 = node_xyz[n1]
    x2, y2, z2 = node_xyz[n2]
    ax_anim.plot([x1, x2], [y1, y2], [z1, z2],
                 linestyle='--', linewidth=1.0, color='0.6')

# -------------------------------------------------------------------------
# Create empty line objects for the deformed structure
# These will be updated during the transient analysis
# -------------------------------------------------------------------------
defo_lines = []
for _ in vis_elems:
    ln, = ax_anim.plot([], [], [], color='navy', linewidth=3.0)
    defo_lines.append(ln)

fig.tight_layout()

# -------------------------------------------------------------------------
# Lists to store response history for plotting
# -------------------------------------------------------------------------
t_hist = []   # time
u_hist = []   # roof displacement in X
a_hist = []   # roof acceleration in X

# -------------------------------------------------------------------------
# Update frequencies for live visualization
#
# plot_every:
#   update the displacement/acceleration curves every N steps
#
# anim_every:
#   update the 3D deformed shape every N steps
#
# Larger values = faster execution, less frequent refreshing
# Smaller values = smoother display, more plotting cost
# -------------------------------------------------------------------------
ok = 0

# -------------------------------------------------------------------------
# Draw initial undeformed/deformed state before starting time stepping
# -------------------------------------------------------------------------
update_live_model()
fig.canvas.draw()
fig.canvas.flush_events()

# Real-time synchronization
t0_wall = time.perf_counter()


# -------------------------------------------------------------------------
# Transient analysis loop
# -------------------------------------------------------------------------
for i in range(nSteps):
    ok = ops.analyze(1, dtGM)

    if ok != 0:
        print(f"Transient analysis failed at step {i+1}, time = {ops.getTime()}")
        break

    # Current analysis time
    t = ops.getTime()

    # Current roof response in global X
    u = ops.nodeDisp(roof_master, 1)
    a = ops.nodeAccel(roof_master, 1)

    # Store time-history results
    t_hist.append(t)
    u_hist.append(u)
    a_hist.append(a)

    # Update 2D response plots every 'plot_every' steps
    if i % plot_every == 0:
        line1.set_data(t_hist, u_hist)
        ax1.relim()
        ax1.autoscale_view(scalex=False, scaley=True)

        line2.set_data(t_hist, a_hist)
        ax2.relim()
        ax2.autoscale_view(scalex=False, scaley=True)

    # Update 3D deformed frame every 'anim_every' steps
    if i % anim_every == 0:
        update_live_model()

    # Redraw the figure when either plot or animation is refreshed
    if i % min(plot_every, anim_every) == 0:
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Keep playback synchronized with analysis time (real-time display)
    target_wall_time = t0_wall + t
    sleep_time = target_wall_time - time.perf_counter()
    if sleep_time > 0:
        time.sleep(sleep_time)

# =============================================================================
# 21. FINAL RESULTS
# =============================================================================
if ok == 0:
    print("Transient analysis completed successfully.")
    print("Final roof displacement X =", ops.nodeDisp(roof_master, 1), "m")
else:
    print("Transient analysis failed before completion.")

plt.ioff()
plt.show()

# =============================================================================
# END
# =============================================================================
ops.wipe()