"""
3D 1-bay, 3-story aluminum frame in OpenSeesPy
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
# 0. FILES
# =============================================================================
os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

gmFile = "input/sine_1Hz_accel.txt"

print("Current working directory:", os.getcwd())
print("Excitation file exists:", os.path.exists(gmFile))


# =============================================================================
# 1. MODEL
# =============================================================================
ops.wipe()
ops.model('Basic', '-ndm', 3, '-ndf', 6)


# =============================================================================
# 2. GEOMETRY
# =============================================================================
Lx, Ly = 0.245, 0.23
H = 0.225
z1, z2, z3 = H, 2.0 * H, 3.0 * H


# =============================================================================
# 3. MATERIAL
# =============================================================================
E = 21e10
nu = 0.33
G = E / (2.0 * (1.0 + nu))


# =============================================================================
# 4. SECTIONS
# =============================================================================
# Columns
bx, by = 0.001, 0.009
A = bx * by
Iy = by * bx**3 / 12.0
Iz = bx * by**3 / 12.0
J = (1.0 / 3.0) * by * bx**3


print("Column A  =", A)
print("Column Iy =", Iy)
print("Column Iz =", Iz)
print("Column J  =", J)


# Beams
bb, hb = 0.001, 0.011
Ab = bb * hb
Iyb = hb * bb**3 / 12.0
Izb = bb * hb**3 / 12.0
Jb = (1.0 / 3.0) * hb * bb**3

print("Beam Ab  =", Ab)
print("Beam Iyb =", Iyb)
print("Beam Izb =", Izb)
print("Beam Jb  =", Jb)


# =============================================================================
# 5. FLOOR MASS
# =============================================================================

mFloor = 0.25 #total weight of beam,column and wooden slab per floor
Wfloor = mFloor * 9.81
JmFloor = mFloor * (Lx**2 + Ly**2) / 12.0


print("Floor mass (kg) =", mFloor)
print("Floor weight (N) =", Wfloor)
print("Floor Jm (kg m^2) =", JmFloor)


# =============================================================================
# 6. NODES
# =============================================================================
# Base
ops.node(1, 0.0, 0.0, 0.0)
ops.node(2, Lx,  0.0, 0.0)
ops.node(3, Lx,  Ly,  0.0)
ops.node(4, 0.0, Ly,  0.0)

# Story 1
ops.node(11, 0.0, 0.0, z1)
ops.node(12, Lx,  0.0, z1)
ops.node(13, Lx,  Ly,  z1)
ops.node(14, 0.0, Ly,  z1)

# Story 2
ops.node(21, 0.0, 0.0, z2)
ops.node(22, Lx,  0.0, z2)
ops.node(23, Lx,  Ly,  z2)
ops.node(24, 0.0, Ly,  z2)

# Story 3
ops.node(31, 0.0, 0.0, z3)
ops.node(32, Lx,  0.0, z3)
ops.node(33, Lx,  Ly,  z3)
ops.node(34, 0.0, Ly,  z3)

# One diaphragm master node per floor
ops.node(1001, Lx / 2.0, Ly / 2.0, z1)
ops.node(1002, Lx / 2.0, Ly / 2.0, z2)
ops.node(1003, Lx / 2.0, Ly / 2.0, z3)


# =============================================================================
# 7. BOUNDARY CONDITIONS
# =============================================================================
# Fully fixed base
for nd in [1, 2, 3, 4]:
    ops.fix(nd, 1, 1, 1, 1, 1, 1)

# Master nodes keep in-plane DOFs (UX, UY, RZ)
ops.fix(1001, 0, 0, 1, 1, 1, 0)
ops.fix(1002, 0, 0, 1, 1, 1, 0)
ops.fix(1003, 0, 0, 1, 1, 1, 0)


# =============================================================================
# 8. RIGID DIAPHRAGMS
# =============================================================================
# Floors are rigid in the XY plane (perpendicular direction = Z)
ops.rigidDiaphragm(3, 1001, 11, 12, 13, 14)
ops.rigidDiaphragm(3, 1002, 21, 22, 23, 24)
ops.rigidDiaphragm(3, 1003, 31, 32, 33, 34)


# =============================================================================
# 9. MASSES
# =============================================================================
# Lumped translational masses in X/Y and rotational inertia about Z
ops.mass(1001, mFloor, mFloor, 0.0, 0.0, 0.0, JmFloor)
ops.mass(1002, mFloor, mFloor, 0.0, 0.0, 0.0, JmFloor)
ops.mass(1003, mFloor, mFloor, 0.0, 0.0, 0.0, JmFloor)


# =============================================================================
# 10. GEOMETRIC TRANSFORMATIONS
# =============================================================================
ops.geomTransf('Linear', 1, 1.0, 0.0, 0.0)  # columns
ops.geomTransf('Linear', 2, 0.0, 1.0, 0.0)  # beams along X
ops.geomTransf('Linear', 3, 1.0, 0.0, 0.0)  # beams along Y


# =============================================================================
# 11. ELEMENTS
# =============================================================================
col_data = [
    (1, 1, 11), (2, 2, 12), (3, 3, 13), (4, 4, 14),
    (5, 11, 21), (6, 12, 22), (7, 13, 23), (8, 14, 24),
    (9, 21, 31), (10, 22, 32), (11, 23, 33), (12, 24, 34),
]

for tag, iNode, jNode in col_data:
    ops.element('elasticBeamColumn', tag, iNode, jNode, A, E, G, J, Iy, Iz, 1)

beam_data = [
    (101, 11, 12, 2), (102, 14, 13, 2), (103, 11, 14, 3), (104, 12, 13, 3),
    (201, 21, 22, 2), (202, 24, 23, 2), (203, 21, 24, 3), (204, 22, 23, 3),
    (301, 31, 32, 2), (302, 34, 33, 2), (303, 31, 34, 3), (304, 32, 33, 3),
]

for tag, iNode, jNode, transfTag in beam_data:
    ops.element('elasticBeamColumn', tag, iNode, jNode, Ab, E, G, Jb, Iyb, Izb, transfTag)


# =============================================================================
# 12. GRAVITY LOAD
# =============================================================================
Pcol = Wfloor / 4.0  # floor weight distributed equally to 4 corner joints

ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)

for nd in [11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34]:
    ops.load(nd, 0.0, 0.0, -Pcol, 0.0, 0.0, 0.0)


# =============================================================================
# 13. STATIC ANALYSIS
# =============================================================================
ops.constraints('Transformation')
ops.numberer('RCM')
ops.system('BandGeneral')
ops.test('NormDispIncr', 1.0e-12, 50, 0)
ops.algorithm('Newton')
ops.integrator('LoadControl', 1.0)
ops.analysis('Static')

ok = ops.analyze(1)

if ok == 0:
    ops.reactions()
    print("Gravity analysis completed successfully.")
    print("Roof corner vertical displacement =", ops.nodeDisp(31, 3), "m")
    print("Base reaction at node 1 in Z =", ops.nodeReaction(1, 3), "N")
else:
    print("Static analysis failed.")

ops.loadConst('-time', 0.0)


# =============================================================================
# 14. MODAL ANALYSIS
# =============================================================================
numModes = 3
lam = ops.eigen('-fullGenLapack', numModes)

fig_modes = plt.figure(figsize=(16, 5))
axes = [fig_modes.add_subplot(1, 3, i + 1, projection='3d') for i in range(numModes)]

for i, mode in enumerate(range(1, numModes + 1)):
    opsv.plot_mode_shape(mode, ax=axes[i], az_el=(-60, 25))

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

floor_nodes = [1001, 1002, 1003]
with open("output/mode_shapes.out", "w") as f:
    f.write("Mode  Node  UX  UY  UZ  RX  RY  RZ\n")
    for mode in range(1, numModes + 1):
        for node in floor_nodes:
            ux = ops.nodeEigenvector(node, mode, 1)
            uy = ops.nodeEigenvector(node, mode, 2)
            uz = ops.nodeEigenvector(node, mode, 3)
            rx = ops.nodeEigenvector(node, mode, 4)
            ry = ops.nodeEigenvector(node, mode, 5)
            rz = ops.nodeEigenvector(node, mode, 6)
            f.write(f"{mode}  {node}  {ux}  {uy}  {uz}  {rx}  {ry}  {rz}\n")


# =============================================================================
# 15. RAYLEIGH DAMPING
# =============================================================================
# 0.5% damping based on the first two modes
zeta = 0.005
w1 = math.sqrt(lam[0])
w2 = math.sqrt(lam[1])

alphaM = 2.0 * zeta * w1 * w2 / (w1 + w2)
betaKinit = 2.0 * zeta / (w1 + w2)

ops.rayleigh(alphaM, 0.0, betaKinit, 0.0)

print("Rayleigh damping:")
print("alphaM    =", alphaM)
print("betaKinit =", betaKinit)


# =============================================================================
# 16. DYNAMIC EXCITATION
# =============================================================================
# Base acceleration input in global X
dtGM = 0.01

ops.timeSeries('Path', 20, '-dt', dtGM, '-filePath', gmFile, '-factor', 9.81)
ops.pattern('UniformExcitation', 20, 1, '-accel', 20)


# =============================================================================
# 17. RECORDERS
# =============================================================================
# Relative response histories in global X
ops.recorder('Node', '-file', 'output/time_floor_disp_X.out',
             '-time', '-node', 1001, 1002, 1003, '-dof', 1, 'disp')

ops.recorder('Node', '-file', 'output/time_floor_accel_X.out',
             '-time', '-node', 1001, 1002, 1003, '-dof', 1, 'accel')

ops.recorder('Node', '-file', 'output/time_roof_disp_X.out',
             '-time', '-node', 1003, '-dof', 1, 'disp')

ops.recorder('Node', '-file', 'output/time_roof_accel_X.out',
             '-time', '-node', 1003, '-dof', 1, 'accel')


# =============================================================================
# 18. VISUALIZATION DATA + HELPER FUNCTIONS
# =============================================================================
plt.ion()  # interactive mode for live updating

# Elements to draw in the 3D view:
# - all columns
# - all beams
vis_elems = [(i, j) for _, i, j in col_data] + [(i, j) for _, i, j, _ in beam_data]

# Structural nodes to visualize
vis_nodes = [1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34]

# Original nodal coordinates (undeformed geometry)
node_xyz = {nd: ops.nodeCoord(nd) for nd in vis_nodes}

# Scale factor only for visualization of deformation
# (does not affect analysis results)
sfac_anim = 20

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
# 19. TRANSIENT ANALYSIS + SINGLE WINDOW VISUALIZATION
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
fig = plt.figure(figsize=(14, 7))

# 2 rows x 2 columns:
# left column  -> two 2D plots
# right column -> one 3D animation spanning both rows
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.4])

# -------------------------------------------------------------------------
# Left-top panel: roof displacement history in global X
# -------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
line1, = ax1.plot([], [], color='red')
ax1.set_title("Roof Displacement")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Displacement (m)")
ax1.set_xlim(0.0, Tmax)
ax1.grid(False)   # no grid background

# -------------------------------------------------------------------------
# Left-bottom panel: roof acceleration history in global X
# -------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[1, 0])
line2, = ax2.plot([], [], color='red')
ax2.set_title("Roof Acceleration")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Acceleration (m/s²)")
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
ax_anim.view_init(elev=25, azim=-60)

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
plot_every = 10
anim_every = 10

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
    u = ops.nodeDisp(1003, 1)
    a = ops.nodeAccel(1003, 1)

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
# 20. FINAL RESULTS
# =============================================================================
if ok == 0:
    print("Transient analysis completed successfully.")
    print("Final roof displacement X =", ops.nodeDisp(1003, 1), "m")
else:
    print("Transient analysis failed before completion.")

plt.ioff()
plt.show()

# =============================================================================
# END
# =============================================================================
ops.wipe()