import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from itertools import combinations

class Representation():

    def __init__(self, N_x = 4, N_y = 4, magnon_number = -1, k_max = 6):
        self.n_y = N_y
        self.n_x = N_x
        self.n = N_x * N_y
        
        if magnon_number == -1:
            self.magnon_number = int(self.n/2)
        else:
            self.magnon_number = magnon_number

        self.k_max = k_max

        self.weight_matrix = np.array([ 2**(ii-1) for ii in range(self.n, 0, -1)]).reshape((self.n_y, self.n_x))
        
        self.rotation_map = np.array( [ [ (1 + (-1)**(x + y))/2 for x in range(self.n_x) ] for y in range(self.n_y) ] ).reshape((self.n_y, self.n_x))

        n_x = self.n_x
        n_y = self.n_y

        k = self.k_max
        k_y = int(np.floor(self.k_max * n_x/n_y))

        index_x = np.arange(0, (2*k + 1) * n_x, 1)
        index_y = np.arange(0, (2*k_y + 1) * n_y, 1)

        X, Y = np.meshgrid(index_x, index_y)

        map_index = np.zeros_like(X)

        map_index[X >= int(n_x/2)] += 1

        map_index[Y >= int(n_y/2)] += 2

        self.index_map = map_index

        self.coords_per_id = [ (int(n_y/2)-1, int(n_x/2)-1), (int(n_y/2)-1, int(n_x/2)),
                          (int(n_y/2), int(n_x/2)-1), (int(n_y/2), int(n_x/2)) ]

        def _generate_bitmasks():
            """Generate bitmasks efficiently."""
            for combo in combinations(range(self.n), self.magnon_number):
                val = 0
                for i in combo:
                    val |= 1 << (self.n - 1 - i)
                yield val

        self.spin_configs = list(_generate_bitmasks())
        self.spin_configs = {i: r for r, i in enumerate(self.spin_configs)}

        self.precompute_flips_change()
        self.get_eta_phases()

    def get_target_coord(self, y, x):
        index = self.index_map[y, x]
        return self.coords_per_id[index], index
    
    def map_int_to_config(self, n):
        return np.array([(n >> i) & 1 for i in reversed(range(self.n))], dtype=np.uint8).reshape((self.n_y, self.n_x))
    
    def map_int_to_basis(self, n):
        return self.spin_configs[n]

    def precompute_flips_change(self):
        """
        Precomputed change in the int index of the state with the spin flip term in Heisenberg Ham.
        """
        weight_rolled = np.roll(self.weight_matrix, (-1, 0), axis=(0, 1))
        self.flips_change_up = np.array(self.weight_matrix - weight_rolled, dtype=int)

        weight_rolled = np.roll(self.weight_matrix, (0, -1), axis=(0, 1))
        self.flips_change_side = np.array(self.weight_matrix - weight_rolled, dtype=int)

    def get_eta_phases(self):

        n_x = self.n_x
        n_y = self.n_y

        k = self.k_max
        k_y = int(np.floor(self.k_max * n_x/n_y))

        phase_maps = []

        for y0, x0 in self.coords_per_id:
            x_index = np.arange(0, (2*k + 1) * n_x, 1) - x0
            y_index = np.arange(0, (2*k_y + 1) * n_y, 1) - y0

            X, Y = np.meshgrid(x_index, y_index)

            tan = np.mod(np.arctan2(Y, X) + np.pi, 2 * np.pi)
            tan[y0, x0] = 0

            phase_maps.append(tan)

        def extended_phase(rot, id):
            config_extended = np.concatenate((2*k + 1)*[rot], axis = 1)
            config_extended = np.concatenate((2*k_y + 1)*[config_extended], axis = 0)

            phase = np.sum(phase_maps[id] * config_extended)

            return np.mod(phase, 2 * np.pi)


        spin_confg_items = list(self.spin_configs.items())

        map_int_to_basis = self.map_int_to_basis
        map_int_to_config = self.map_int_to_config
        get_target_coords = self.get_target_coord
        rotation_map = self.rotation_map

        directions = [(0, -1), (-1, 0)]  # x and y directions
        flips = [self.flips_change_side, self.flips_change_up]  # x and y directions

        eta_phases = []

        for config_int, state in tqdm(spin_confg_items):
            config = map_int_to_config(config_int)
            
            for id, dir in enumerate(directions):
                roll_config = np.roll(config, dir, axis=(0, 1))
                possible_mixing = np.mod(config + roll_config, 2)

                config_int_shift = np.array(possible_mixing * flips[id], dtype=float)
                
                # multiplying to account for sign
                config_int_shift *= (-1.)**config

                for int_shift in config_int_shift[config_int_shift != 0].astype(int):
                    basis_index = map_int_to_basis(config_int + int_shift)

                    if basis_index < state:
                        y, x = np.where(config_int_shift == int_shift)
                        y = y[0]
                        x = x[0]
                        sign = (-1.)**config[y, x]

                        flipped_cfg = map_int_to_config(config_int + int_shift)

                        flipped_rot = (flipped_cfg * rotation_map + (1 - flipped_cfg) * (1 - rotation_map))
                        flipped_roll_rot = np.roll(flipped_rot, dir, axis=(0, 1))

                        target, index = get_target_coords(y, x)

                        x = -x + target[1]
                        y = -y + target[0]

                        rot_xy = np.roll(flipped_rot, (y, x), (0, 1))
                        roll_rot_xy = np.roll(flipped_roll_rot, (y, x), (0, 1))
                        
                        eta_phases.append( sign * np.mod( extended_phase(rot_xy, index) + extended_phase(roll_rot_xy, index) , 2 * np.pi))

        print(len(eta_phases))
        np.savetxt(fr'JW-phases/no_sym/eta_phases_{n_x}x{n_y}_Sz={int(n_x*n_y/2-self.magnon_number)}.txt', eta_phases)

        if self.magnon_number == int((self.n_x * self.n_y)/2):
            eta_phases = []

            for config_int, state in tqdm(spin_confg_items):
                config = self.map_int_to_config(config_int)
                
                # finds where we can down-flip spin
                # It is this sum exp(ikj) S^-_j on GS
                for y, x in zip(*np.where(config == 1)):
                    flipped = config.copy()
                    flipped[y, x] = 0

                    flipped_rot = (flipped * rotation_map + (1 - flipped) * (1 - rotation_map))

                    target, index = get_target_coords(y, x)

                    x = -x + target[1]
                    y = -y + target[0]

                    rot_xy = np.roll(flipped_rot, (y, x), (0, 1))
                    
                    eta_phases.append( np.mod( extended_phase(rot_xy, index), 2 * np.pi))

            print(len(eta_phases))
            np.savetxt(fr'JW-phases/no_sym/eta_phases_{n_x}x{n_y}_S-.txt', eta_phases)


if __name__ == '__main__':
    nx = 6
    ny = 4

    k_max = 0
    mag = int((nx * ny)/2)
    r = Representation(nx, ny, magnon_number=mag, k_max=k_max)

    mag = int((nx * ny)/2) - 1
    r = Representation(nx, ny, magnon_number=mag, k_max=k_max)