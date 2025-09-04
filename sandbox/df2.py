import numpy as np
from vispy import app, scene
from sandbox.lattice_gas2 import index_to_cartesian, simulate_lattice, generate_lattice

# ----------------------------
# Simulation setup
# ----------------------------
N_rows, N_cols = 100, 100
N_steps = 1000
lattice = generate_lattice(N_rows, N_cols, 0.3)
lattice[:,:,:] = 0
lattice[:,:,0] = 1
lattice[0:10,:,6] = 1
lattice[30:60,50:51,6] = 1
lattice[90:,:,6] = 1
lattice[lattice[:,:,6] == 1,:6] = 0
lattice_list, momenta, particle_numbers, spurious_c = simulate_lattice(lattice, N_steps)

momenta = np.array(momenta)
arrow_length = 0.5   # scale for arrow shafts
arrow_size = 5.0     # arrowhead size

canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(aspect=1)
view.camera.set_range(x=(0, N_cols), y=(0, N_rows))

# Precompute coords
coords = {(i, j): index_to_cartesian(i, j) for i in range(N_rows) for j in range(N_cols)}
boundaries = {(i, j): lattice[i, j, 6] for i in range(N_rows) for j in range(N_cols)}

# Separate boundary vs non-boundary points
boundary_points = [coords[(i, j)] for (i, j) in boundaries if boundaries[(i, j)] == 1]
non_boundary_points = [coords[(i, j)] for (i, j) in boundaries if boundaries[(i, j)] == 0]

# Non-boundary lattice points (black dots)
points = scene.visuals.Markers(parent=view.scene)
points.set_data(np.array(non_boundary_points), size=5, face_color='black')

boundary_markers = scene.visuals.Markers(parent=view.scene)
boundary_markers.set_data(
    np.array(boundary_points),
    size=20,
    face_color='red',
    edge_color='red',
    symbol='x'
)

# Arrows for momenta
arrow_node = scene.visuals.Arrow(
    pos=np.zeros((0, 2), dtype=np.float32),
    arrows=np.zeros((0, 4), dtype=np.float32),
    color='blue',
    arrow_color='blue',
    arrow_size=arrow_size,
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

    frame_idx = (frame_idx + 1) % momenta.shape[0]
    draw_frame(frame_idx)

def draw_frame(idx):
    shafts = []
    arrow_pairs = []

    # go through each lattice site
    for i in range(N_rows):
        for j in range(N_cols):
            x, y = coords[(i, j)]
            dx, dy = momenta[idx, i, j]  # vector
            if dx == 0 and dy == 0:
                continue

            dst = (x + dx * arrow_length, y + dy * arrow_length)

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
        frame_idx = (frame_idx + 1) % momenta.shape[0]
        draw_frame(frame_idx)
    elif event.key == 'Left':
        frame_idx = (frame_idx - 1) % momenta.shape[0]
        draw_frame(frame_idx)
    elif event.key == 'R':
        frame_idx = 0
        draw_frame(frame_idx)

timer = app.Timer(interval=0.05, connect=update, start=True)

if __name__ == '__main__':
    app.run()