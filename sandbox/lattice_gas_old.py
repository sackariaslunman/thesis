import random
import math
import numpy as np
from typing import Tuple, Dict

def generate_state() -> int:
    return random.randint(0,63)

def index_to_cartesian(i: int, j: int) -> Tuple[float, float]:
    a = 1
    x = a * j
    if i % 2 == 1:
        x += a / 2
    y = math.sqrt(3) * a / 2 * i
    return x, y

def add_indices(indices1: Tuple[int,...], indices2: Tuple[int,...]) -> Tuple[int,...]:
    return tuple(x + y for x, y in zip(indices1, indices2))

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
    # 0b010101: (0b101010,),
    # 0b101010: (0b010101,)
}

def generate_lattice(shape: Tuple[int,int]):
    lattice = { (i,j): generate_state() for j in range(shape[1]) for i in range(shape[0]) }
    return lattice

def update_lattice(lattice: Dict[Tuple[int,int], int], N_rows: int, N_cols: int) -> Dict[Tuple[int,int], int]:
    new_lattice = {key: 0 for key in lattice.keys()}
    for site, value in lattice.items():
        if value in scattering_rules:
            value = random.choice(scattering_rules[value])
        is_even_row = site[0] % 2
        for bit, neighbor in bit_to_neighbor.items():
            if value & bit:
                new_site = add_indices(site, neighbor[is_even_row])
                new_site = (new_site[0] % N_rows, new_site[1] % N_cols)
                new_lattice[new_site] |= bit
    return new_lattice