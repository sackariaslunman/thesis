import numpy as np
import random
import math
from typing import Tuple

def generate_state(n_states: int) -> int:
    return random.randint(0, n_states-1)

def index_to_cartesian(i: int, j: int) -> Tuple[float, float]:
    a = 1
    x = a * i + a / 2 * j
    y = math.sqrt(3) * a / 2 * j
    return x, y

neighbors = np.array([
    [1,0],
    [0,1],
    [-1,1],
    [-1,-0],
    [0,-1],
    [1,-1],
])

def generate_lattice(max_x: int, max_y: int):
    lattice = {}
    origin_site = (0,0)
    unexpanded_sites = set([origin_site])
    discovered_sites = set([origin_site])

    while unexpanded_sites:
        site = unexpanded_sites.pop()
        lattice[site] = generate_state(8)
        new_sites = neighbors + np.array(site)
        for new_site in new_sites:
            new_site_tuple = tuple(new_site.tolist())
            x, y = index_to_cartesian(*new_site)
            if new_site_tuple not in discovered_sites and \
                    abs(x) <= max_x and abs(y) <= max_y:
                unexpanded_sites.add(new_site_tuple)
            discovered_sites.add(new_site_tuple)
    return lattice

lattice = generate_lattice(10,10)