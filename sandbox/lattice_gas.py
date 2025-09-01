import random
import math
import numpy as np
from typing import Tuple, List
from tqdm import tqdm

def index_to_cartesian(i: int, j: int) -> Tuple[float, float]:
    a = 1
    x = a * j
    if i % 2 == 1:
        x += a / 2
    y = math.sqrt(3) * a / 2 * i
    return x, y

bit_to_neighbor = {
    1: ((0, 1), (0, 1)),
    2: ((1, 0), (1, 1)),
    4: ((1, -1), (1, 0)),
    8: ((0, -1), (0, -1)),
    16: ((-1, -1), (-1, 0)),
    32: ((-1, 0), (-1,1))
}

def bit_index_to_direction(i: int) -> np.ndarray:
    return np.array([
        np.cos(2 * np.pi * i / 6),
        np.sin(2 * np.pi * i / 6)
    ])

bit_directions = np.array([bit_index_to_direction(i) for i in range(6)])

scattering_rules = {
    0b001001: (0b100100, 0b010010),
    0b010010: (0b001001, 0b100100),
    0b100100: (0b001001, 0b010010),
    0b010101: (0b101010,),
    0b101010: (0b010101,)
}

def generate_lattice(shape: Tuple[int,int]):
    lattice = np.random.randint(0,64, size=shape).astype(np.uint8)
    return lattice

def update_lattice(lattice: np.ndarray) -> np.ndarray:
    N_rows = lattice.shape[0]
    N_cols = lattice.shape[1]
    new_lattice = np.zeros_like(lattice)
    for row in range(N_rows):
        for col in range(N_cols):
            value = lattice[row, col]
            if value in scattering_rules:
                value = random.choice(scattering_rules[value])
            is_even_row = row % 2
            for bit, neighbor in bit_to_neighbor.items():
                drow, dcol = neighbor[is_even_row]
                if value & bit:
                    new_row = (row + drow) % N_rows
                    new_col = (col + dcol) % N_cols
                    new_lattice[new_row, new_col] |= bit
    return new_lattice

def bitfield(n, num_bits) -> List[int]:
    return [int(digit) for digit in f"{n:0{num_bits}b}"]

def bits_to_xy_momenta(value: int) -> np.ndarray:
    bits = np.array(bitfield(value, 6)).reshape(-1,1)
    momenta = bits * bit_directions
    return momenta.sum(axis=0)

def bits_to_spurious(value: int) -> int:
    bits = np.array(bitfield(value, 6)).reshape(-1,1)
    I = bits[1] - bits[4] + bits[2] - bits[5] + bits[3] - bits[0]
    return I

def simulate_lattice(N_rows: int, N_cols: int, N_steps: int):
    lattice = generate_lattice((N_rows, N_cols))
    lattice_time_series = np.zeros((N_steps + 1, *lattice.shape)).astype(np.uint8)
    lattice_time_series[0,:,:] = lattice

    momenta = []
    particle_numbers = []
    spurious_c = []

    for t in tqdm(range(N_steps)):
        lattice = update_lattice(lattice)
        lattice_time_series[t+1,:,:] = lattice
        N_particles = 0
        total_momentum = 0
        spurious = 0
        for row in range(N_rows):
            for col in range(N_cols):
                value = lattice[row, col]
                N_particles += sum(bitfield(value, 6))
                total_momentum += bits_to_xy_momenta(value)
                spurious += bits_to_spurious(value)
        particle_numbers.append(N_particles)
        momenta.append(total_momentum)
        spurious_c.append(spurious)
    
    return lattice_time_series, momenta, particle_numbers, spurious_c