import math
import numpy as np
from typing import Tuple
from tqdm import tqdm

def index_to_cartesian(i: int, j: int) -> Tuple[float, float]:
    a = 1
    x = a * j + a / 2 * i
    y = math.sqrt(3) * a / 2 * i
    return x, y

directions = np.array([
    [ 0, 1],
    [ 1, 0],
    [ 1, -1],
    [ 0,-1],
    [-1, 0],
    [-1, 1],
])

def bit_index_to_direction(i: int) -> np.ndarray:
    return np.array([
        np.cos(2 * np.pi * i / 6),
        np.sin(2 * np.pi * i / 6),
    ])

bit_directions = np.array([bit_index_to_direction(i) for i in range(6)])

scattering_rules = {
    (0,0,1,0,0,1): np.array([(1,0,0,1,0,0), (0,1,0,0,1,0)]),
    (0,1,0,0,1,0): np.array([(0,0,1,0,0,1), (1,0,0,1,0,0)]),
    (1,0,0,1,0,0): np.array([(0,0,1,0,0,1), (0,1,0,0,1,0)]),
    (0,1,0,1,0,1): np.array([(1,0,1,0,1,0)]),
    (1,0,1,0,1,0): np.array([(0,1,0,1,0,1)]),
}

reflective_rules = {
    0: 3,
    1: 4,
    2: 5,
    3: 0,
    4: 1,
    5: 2
}

def generate_lattice(N_rows: int, N_cols: int, density: float = 0.5) -> np.ndarray:
    assert 0 <= density <= 1
    # 0-6 dims: 0-5 are for particles with different directions. dim 6 is for reflective boundaries
    lattice = np.zeros((N_rows, N_cols, 6 + 1))
    lattice[:,:,:6] = np.random.choice([0, 1], size=(N_rows, N_cols, 6), p=[1-density, density]).astype(bool)
    return lattice

def apply_collisions(lattice: np.ndarray) -> np.ndarray:
    new_lattice = lattice.copy()
    for input, output in scattering_rules.items():
        matches = np.all(lattice[:,:,:6] == input, axis=-1)
        choices = np.random.randint(0, len(output), size=matches.sum())
        new_values = output[choices]
        new_lattice[:,:,:6][matches] = new_values
    return new_lattice

def move_particles(lattice: np.ndarray) -> np.ndarray:
    new_lattice = np.zeros_like(lattice, dtype=bool)
    new_lattice[:,:,6] = lattice[:,:,6]

    for i, direction in enumerate(directions):
        new_lattice[:,:,i] = np.roll(lattice[:,:,i], shift=direction, axis=(0,1))

    return new_lattice

def resolve_reflections(lattice: np.ndarray) -> np.ndarray:
    new_lattice = lattice.copy()

    for i, direction in enumerate(directions):
        # handle collisions
        reflective_collisions = new_lattice[:,:,i] & new_lattice[:,:,6]
        new_lattice[reflective_collisions,i] = 0
        new_i = reflective_rules[i]
        prev_positions = np.roll(reflective_collisions, shift=-direction, axis=(0,1))
        new_lattice[prev_positions,new_i] = 1

    return new_lattice

def unsqueeze(arr, axis=-1):
    return np.expand_dims(arr, axis)

def compute_momentum(lattice: np.ndarray) -> np.ndarray:
    return (unsqueeze(lattice[:,:,:6]) * bit_directions).sum(axis=-2)

def simulate_lattice(lattice: np.ndarray, N_steps: int):
    lattice_time_series = np.zeros((N_steps * 3 + 1, *lattice.shape), dtype=bool)
    lattice_time_series[0,:,:] = lattice

    momenta = []
    particle_numbers = []
    spurious_c = []

    for t in tqdm(range(N_steps)):
        lattice = apply_collisions(lattice)
        lattice_time_series[3*t+1,:,:] = lattice

        lattice = move_particles(lattice)
        lattice_time_series[3*t+2,:,:] = lattice
        
        lattice = resolve_reflections(lattice)
        lattice_time_series[3*t+3,:,:] = lattice

        momentum = compute_momentum(lattice)
        momenta.append(momentum)
    
    return lattice_time_series, momenta, particle_numbers, spurious_c