import numpy as np
from vispy import app, scene
from sandbox.lattice_gas import index_to_cartesian, simulate_lattice

# ----------------------------
# Simulation setup
# ----------------------------
N_rows, N_cols = 100, 100
N_steps = 200
lattice_list, momenta, particle_numbers, spurious_c = simulate_lattice(N_rows, N_cols, N_steps)

# ----------------------------
# VisPy visualization
# ----------------------------
angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, -2*np.pi/3, -np.pi/3])
arrow_length = 0.5
head_size = 0.2  # arrowhead size

canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(aspect=1)
view.camera.set_range(x=(0, N_cols), y=(0, N_rows))

# Precompute coords
coords = {(i,j): index_to_cartesian(i, j) for i in range(N_rows) for j in range(N_cols) }

# Markers for lattice points
points = scene.visuals.Markers(parent=view.scene)
points.set_data(np.array(list(coords.values())), size=1, face_color='red')

# Arrows for particles
arrow_node = scene.visuals.Arrow(
    pos=np.zeros((0, 2), dtype=np.float32),
    arrows=np.zeros((0, 4), dtype=np.float32),
    color='blue',
    arrow_color='blue',
    arrow_size=5.0,
    arrow_type='stealth',
    connect='segments',
    parent=view.scene
)

frame_idx = 0
running = True  # control playback

def update(ev):
    global frame_idx
    if not running:
        return

    frame_idx = (frame_idx + 1) % len(lattice_list)
    draw_frame(frame_idx)

def draw_frame(idx):
    lattice = lattice_list[idx]
    shafts = []
    arrow_pairs = []
    for i in range(lattice.shape[0]):
        for j in range(lattice.shape[1]):
            value = lattice[i, j]
            x, y = coords[(i, j)]
            for k in range(6):
                if (value >> k) & 1:
                    angle = angles[k]
                    dx = arrow_length * np.cos(angle)
                    dy = arrow_length * np.sin(angle)
                    dst = (x + dx, y + dy)

                    shafts.extend([[x, y], dst])
                    arrow_pairs.append([x, y, dst[0], dst[1]])

    if shafts:
        pos = np.array(shafts, dtype=np.float32)
        arrows = np.array(arrow_pairs, dtype=np.float32)
    else:
        pos = np.zeros((0, 2), dtype=np.float32)
        arrows = np.zeros((0, 4), dtype=np.float32)

    arrow_node.set_data(pos=pos, arrows=arrows)
    canvas.update()

# Keyboard controls
@canvas.events.key_press.connect
def on_key(event):
    global running, frame_idx
    if event.key == 'Space':
        running = not running
    elif event.key == 'Right':
        frame_idx = (frame_idx + 1) % len(lattice_list)
        draw_frame(frame_idx)
    elif event.key == 'Left':
        frame_idx = (frame_idx - 1) % len(lattice_list)
        draw_frame(frame_idx)
    elif event.key == 'R':
        frame_idx = 0
        draw_frame(frame_idx)

timer = app.Timer(interval=0.05, connect=update, start=True)

if __name__ == '__main__':
    app.run()