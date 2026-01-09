import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from itertools import combinations

class Representation():

    def __init__(self, n_x = 4, n_y = 4, magnon_number = -1, k_max = 6):
        self.n_y = n_y
        self.n_x = n_x
        self.n = n_x * n_y
        
        if magnon_number == -1:
            self.magnon_number = int(self.n/2)
        else:
            self.magnon_number = magnon_number

        self.k_max = k_max

        self.weight_matrix = np.array([ 2**(ii-1) for ii in range(self.n, 0, -1)]).reshape((self.n_y, self.n_x))
        
        self.rotation_map = np.array( [ [ (1 + (-1)**(x + y))/2 for x in range(self.n_x) ] for y in range(self.n_y) ] ).reshape((self.n_y, self.n_x))

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

        self.get_representations()
        self.precompute_flips_change()
        self.get_eta_phases()

    def get_target_coord(self, y, x):
        index = self.index_map[y, x]
        return self.coords_per_id[index], index

    def get_representations(self, n_workers=None):
        """
        Generates representations of spin states with parallel execution.
        
        Args:
            n_workers: Number of parallel workers. None = use all CPU cores.
        """
        if n_workers is None:
            n_workers = mp.cpu_count()
        
        # Pre-compute translations (even only)
        xs = np.arange(self.n_x)
        ys = np.arange(self.n_y)
        temp_translations = [[[-int(y), -int(x)] for x in xs] for y in ys]
        translations = []
        
        for trans in temp_translations:
            for t in trans:
                if np.sum(t) % 2 == 0:
                    translations.append(t)
        
        # Pre-compute weight matrices and store as instance variable
        self.weight_matrices = [
            (np.roll(self.weight_matrix, np.array(t), axis=(0, 1)).ravel(), t)
            for t in translations
        ]
        
        # Pre-compute bitmasks for all combinations
        all_states = list(self._generate_bitmasks())
        self.size = len(all_states)
        
        # Split work into chunks for parallel processing
        chunk_size = max(1, len(all_states) // (n_workers * 4))
        
        # Initialize results storage
        self.representatives = []
        self.norms = {}
        self.get_representative = {}
        
        # Process in parallel
        print(f"Processing {len(all_states)} states with {n_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit chunks
            futures = []
            for i in range(0, len(all_states), chunk_size):
                chunk = all_states[i:i + chunk_size]
                futures.append(executor.submit(self._process_state_batch, chunk))
            
            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=len(futures)):
                batch_results = future.result()
                self._merge_results(batch_results)
        
        self.representatives.sort()
        self.representatives.reverse()

        # Create final representative enumeration
        self.representatives = {r: i for i, r in enumerate(self.representatives)}
        
        print(f'Total number of found representatives: {len(self.representatives)}')
        print(f'Percent of repr in total states: {round(100*len(self.representatives)/self.size, 2)}%')
        print(f'Estimated matrix reduction: {round(self.size/len(self.representatives), 1)}:1')
    
    def _generate_bitmasks(self):
        """Generate bitmasks efficiently."""
        for combo in combinations(range(self.n), self.magnon_number):
            val = 0
            for i in combo:
                val |= 1 << (self.n - 1 - i)
            yield val
    
    def _process_state_batch(self, states):
        """Process a batch of states."""
        results = []
        
        for state_int in states:
            # Convert to config using the proper mapping
            state_cfg = self.map_int_to_config(state_int).ravel()
            
            set_states = set()
            all_states = []
            is_representative = True
            
            for w, trans in self.weight_matrices:
                rolled_state_int = int(np.dot(w, state_cfg))
                
                if rolled_state_int < state_int:
                    is_representative = False
                    break
                
                set_states.add(rolled_state_int)
                all_states.append((rolled_state_int, trans))
            
            if is_representative:
                results.append({
                    'state': state_int,
                    'norm': len(set_states),
                    'all_states': all_states
                })
        
        return results
    
    def _merge_results(self, batch_results):
        """Merge results from a batch into main storage."""
        for result in batch_results:
            state = result['state']
            norm = result['norm']
            all_states = result['all_states']
            
            self.representatives.append(state)
            self.norms[state] = norm
            
            for _state, trans in all_states:
                if _state in self.get_representative:
                    self.get_representative[_state][1].append(trans)
                else:
                    self.get_representative[_state] = [state, [trans]]
    
    def map_int_to_config(self, n):
        return np.array([(n >> i) & 1 for i in reversed(range(self.n))], dtype=np.uint8).reshape((self.n_y, self.n_x))
    
    def map_int_to_basis(self, n):
        return self.representatives[n]

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

        # parameter used to artificially enlarge the supercell
        # uses PBC to change system [nx, ny] into [ (2*k + 1) * nx, (2*ky + 1) * ny ]
        # after all results for k = 0 and k != 0 are to epsilon the same
        k = self.k_max
        k_y = int(np.floor(self.k_max * n_x/n_y))

        # stores 4 different maps of phases theta_ir
        # 4 comes from ad hoc choice of "shifted supercell"
        phase_maps = []

        for y0, x0 in self.coords_per_id:
            x_index = np.arange(0, (2*k + 1) * n_x, 1) - x0
            y_index = np.arange(0, (2*k_y + 1) * n_y, 1) - y0

            X, Y = np.meshgrid(x_index, y_index)

            tan = np.mod(np.arctan2(Y, X) + np.pi, 2 * np.pi)
            # in numpy the first coordinate is y-axis and the second is x-axis, thats why we have (y0, x0)
            tan[y0, x0] = 0

            # creates the phase map theta_ir for each possible "center" of shifted supercells
            # i = (x_index, y_index)

            ## uncomment if want to check phase colormaps
            # plt.imshow(tan, cmap = 'plasma', aspect = 'auto')
            # plt.colorbar()
            # plt.show()

            phase_maps.append(tan)

        def extended_phase(rot, id):
            # extends the config to desired enlargened size (for k = 0 it's the same as just rot)
            config_extended = np.concatenate((2*k + 1)*[rot], axis = 1)
            config_extended = np.concatenate((2*k_y + 1)*[config_extended], axis = 0)

            phase = np.sum(phase_maps[id] * config_extended)

            return np.mod(phase, 2 * np.pi)

        # Precompute mappings
        spin_confg_items = list(self.representatives.items())

        map_int_to_basis = self.map_int_to_basis
        map_int_to_config = self.map_int_to_config
        rotation_map = self.rotation_map
        get_target_coords = self.get_target_coord

        directions = [(0, -1), (-1, 0)]  # x and y directions
        flips = [self.flips_change_side, self.flips_change_up]  # x and y directions
        # I see that it gives the correct result for 4x4, but I don't know why we need this phase
        thetas = [0, np.pi/2]  # x and y directions
        #thetas = [0, 0]  # x and y directions

        eta_phases = []

        for config_int, state in tqdm(spin_confg_items):
            config = map_int_to_config(config_int)
            
            for id, dir in enumerate(directions):
                # shifts the config to get i -> j
                roll_config = np.roll(config, dir, axis=(0, 1))
                # finds where a NN spin flipping can occur
                possible_mixing = np.mod(config + roll_config, 2)

                # checks how configuration changes upon the spin flip
                config_int_shift = np.array(possible_mixing * flips[id], dtype=float)
                
                # multiplying to account for sign
                config_int_shift *= (-1.)**config

                for int_shift in config_int_shift[config_int_shift != 0].astype(int):
                    # finds the momentum representative of the flipped state
                    repr_b, _ = self.get_representative[config_int + int_shift]
                    # maps to basis
                    basis_index = map_int_to_basis(repr_b)

                    if basis_index < state:
                        # finds where the hopping occured (it is always non degenerate!)
                        y, x = np.where(config_int_shift == int_shift)
                        y = y[0]
                        x = x[0]
                        # sign to check whether it was S+S- or S-S+, as it:
                        # 1) conjugetes the exponent (-1. sign in the phase)
                        # 2) changes whether we calculate phase on the flipped or original state (see pdf with Hamiltonian)
                        sign = 1. if config[y, x] == 0 else -1

                        if sign == 1:
                            flipped_cfg = map_int_to_config(config_int + int_shift)
                        else:
                            # it's only flipped by name, it's the original state
                            flipped_cfg = config.copy()

                        # maps harcore bosons into rotated hardcore bosons
                        flipped_rot = (flipped_cfg * rotation_map + (1 - flipped_cfg) * (1 - rotation_map))
                        # it's needed to evaluate theta_jr |psi> = theta_ir T_(j-i) |psi>
                        flipped_roll_rot = np.roll(flipped_rot, dir, axis=(0, 1))

                        # checks what is this center for shifted supercell
                        target, index = get_target_coords(y, x)

                        # in x we have target[1] as in numpy the coordinates are y-first
                        x = -x + target[1]
                        y = -y + target[0]

                        # shifts (with PBC) the configuration, so the place where the flipping occures is at the center (target variable)
                        rot_xy = np.roll(flipped_rot, (y, x), (0, 1))
                        roll_rot_xy = np.roll(flipped_roll_rot, (y, x), (0, 1))
                        
                        # calculates theta_ri + theta_rj = theta_ir + theta_jr + 2pi, which we remove with taking mod 2pi everything
                        # again, sign for conjugate
                        eta_phases.append( sign * np.mod( thetas[id] + extended_phase(rot_xy, index) + extended_phase(roll_rot_xy, index), 2 * np.pi))

        print(len(eta_phases))
        np.savetxt(fr'JW-phases/eta_phases_{n_x}x{n_y}_Sz={int(n_x*n_y/2-self.magnon_number)}.txt', eta_phases)


if __name__ == '__main__':
    nx = 4
    ny = 4

    k_max = 0
    mag = int((nx * ny)/2)
    r = Representation(nx, ny, magnon_number=mag, k_max=k_max)

    mag = int((nx * ny)/2) - 1
    r = Representation(nx, ny, magnon_number=mag, k_max=k_max)