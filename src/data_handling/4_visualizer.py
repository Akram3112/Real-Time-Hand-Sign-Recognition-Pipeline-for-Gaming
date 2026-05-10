import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.widgets import Slider, RadioButtons

# ==============================
# LOAD ALL LABELS
# ==============================

FEATURES_DIR = "2_features"

all_data   = {}   # label -> dataframe
all_frames = {}   # label -> numpy array (n_frames, 126)
all_pids   = {}   # label -> person_id array

for fname in sorted(os.listdir(FEATURES_DIR)):
    if not fname.endswith(".csv"):
        continue
    label = fname.replace(".csv", "")
    df    = pd.read_csv(os.path.join(FEATURES_DIR, fname))
    all_pids[label]   = df["person_id"].values
    all_frames[label] = df.drop(columns=["person_id"]).values

LABELS = list(all_frames.keys())

# ==============================
# HAND CONNECTIONS (MediaPipe)
# ==============================

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# ==============================
# GLOBAL AXIS LIMITS (across ALL labels)
# ==============================

_all_vals = np.concatenate([f[f != 0] for f in all_frames.values()])
_pad      = _all_vals.std() * 0.3
LIM       = (_all_vals.min() - _pad, _all_vals.max() + _pad)

# ==============================
# STATE
# ==============================

current_label    = LABELS[0]
current_frame_idx = 0

# ==============================
# DRAW
# ==============================

def draw_hand_2d(ax, pts, color, label):
    """pts: (21,3). Plot X,Y only. Skip if not detected (all zeros)."""
    if not pts.any():
        return
    ax.scatter(pts[:, 0], pts[:, 1], c=color, s=18, zorder=3)
    for a, b in CONNECTIONS:
        ax.plot(
            [pts[a, 0], pts[b, 0]],
            [pts[a, 1], pts[b, 1]],
            c=color, linewidth=1.5
        )

def plot_frame(frame_idx, label):
    ax.clear()
    frame = all_frames[label][frame_idx]
    left  = frame[  :63].reshape(21, 3)
    right = frame[63:  ].reshape(21, 3)

    draw_hand_2d(ax, left,  'steelblue', 'Left')
    draw_hand_2d(ax, right, 'tomato',    'Right')

    ax.set_xlim(*LIM)
    ax.set_ylim(*LIM)
    ax.set_aspect('equal')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.invert_yaxis()   # match image convention (y down)

    pid = all_pids[label][frame_idx]
    ax.set_title(f"[{label}]  Frame {frame_idx} | Person {pid}", pad=8)

    handles = []
    if left.any():
        handles.append(mlines.Line2D([], [], color='steelblue',
                                     marker='o', label='Left'))
    if right.any():
        handles.append(mlines.Line2D([], [], color='tomato',
                                     marker='o', label='Right'))
    ax.legend(handles=handles, loc='upper right')
    fig.canvas.draw_idle()

# ==============================
# FIGURE LAYOUT
# ==============================

fig = plt.figure(figsize=(10, 8))
plt.subplots_adjust(left=0.25, bottom=0.2)

ax = fig.add_subplot(111)

# ── frame slider ────────────────────────────────────────────────────
ax_frame = plt.axes([0.25, 0.08, 0.65, 0.03])
frame_slider = Slider(
    ax_frame, "Frame", 0,
    len(all_frames[current_label]) - 1,
    valinit=0, valstep=1
)

# ── label slider (discrete, one step per label) ──────────────────────
ax_label = plt.axes([0.25, 0.03, 0.65, 0.03])
label_slider = Slider(
    ax_label, "Label", 0, len(LABELS) - 1,
    valinit=0, valstep=1
)
label_slider.valtext.set_text(LABELS[0])   # show name instead of index

# ==============================
# CALLBACKS
# ==============================

def on_frame_change(val):
    global current_frame_idx
    current_frame_idx = int(frame_slider.val)
    # clamp in case new label has fewer frames
    max_f = len(all_frames[current_label]) - 1
    current_frame_idx = min(current_frame_idx, max_f)
    plot_frame(current_frame_idx, current_label)

def on_label_change(val):
    global current_label, current_frame_idx
    current_label = LABELS[int(label_slider.val)]
    label_slider.valtext.set_text(current_label)

    # reset frame slider range for new label
    max_f = len(all_frames[current_label]) - 1
    frame_slider.valmax = max_f
    frame_slider.ax.set_xlim(0, max_f)

    current_frame_idx = 0
    frame_slider.set_val(0)
    plot_frame(0, current_label)

frame_slider.on_changed(on_frame_change)
label_slider.on_changed(on_label_change)

plot_frame(0, current_label)
plt.show()