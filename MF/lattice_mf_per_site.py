from numpy import roll, array, mod, dot, uint8, sum, ones
import numpy as np
from itertools import combinations
from time import time
import os
import matplotlib.pyplot as plt

from scipy.special import binom

from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, lil_matrix

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from tqdm import tqdm

class SpinConfiguration:
    
    def __init__(self, nx = 6, ny = 6, magnon_number = -1, delta = 1., lamb = 1., k = np.array([0., 0.]),
                 alpha = 1/2, lowest_eignstates = 1, print_data = False):
        """
        k is in units of pi
        """
        self.debug = 0
        
        self.print = print_data
        self.n_lowest = lowest_eignstates

        self.nx = nx
        self.ny = ny
        self.n = nx * ny

        if magnon_number == -1:
            self.magnon_number = int(self.n/2)
        else:
            self.magnon_number = magnon_number

        self.delta = delta
        self.lamb = lamb
        self.k = k

        self.alpha = alpha
                                        
        self.weight_matrix = array([ 2**(ii-1) for ii in range(self.n, 0, -1)]).reshape((self.ny, self.nx))
        
        self.precompute_flips_change()
        
        self.rotation_map = array( [ [ (1+(-1)**(x+y))/2 for x in range(self.nx) ] for y in range(self.ny) ] ).reshape((self.ny, self.nx))
        
        self.t_matrices = None
        self.n_in_basis = None

        xs = np.arange(self.nx)
        ys = np.arange(self.ny)
        all_tr = [ [ (-int(y), -int(x)) for x in xs ] for y in ys ]

        # Pre-compute translations (even only)
        translations = []
        
        for trans in all_tr:
            for t in trans:
                if np.sum(t) % 2 == 0:
                    translations.append(t)

        # Pre-compute weight matrices and store as instance variable
        self.weight_matrices = [
            ((np.roll(self.weight_matrix, np.array(t), axis=(0, 1))).ravel(), t)
            for t in translations
        ]
        
        self.eign_en = []
        self.eignstates = []
        
        if self.magnon_number > self.n:
            raise ValueError('Magnon number should be smaller than the system size')
        
        self.size = int(binom(self.n, self.magnon_number))

        self.get_representations()

        self.get_hamiltonian()
        
        start = time()

        self.get_eigva_eigve()
        
        if print_data:
            print(f'Finding eigvals and eigvecs took: {round(time()-start, 5)} s')
    

    def get_hamiltonian(self, k = None, delta = None, lamb = None, mf_n = 0, t = 0):
        
        if k is None:
            k = self.k

        if delta is not None:
            self.delta = delta

        if lamb is not None:
            self.lamb = lamb

        self.mf_n = mf_n
        self.t = t

        self.k = k
        
        self.generate_hamiltonian(k)
    
    
    def generate_hamiltonian(self, k = None, n_workers = None):

        start = time()

        self.hamiltonian = None
        
        self.hamiltonian_elements = []
        self.ham_i = []
        self.ham_j = []

        e0 = 2. * self.n * self.delta / 4.
        self.e0 = e0

        spin_confg_items = list(self.representatives.items())

        easy_ks = [0., 2.]
        
        if easy_ks.__contains__(k[0]) and easy_ks.__contains__(k[1]):
            self.d_type = np.float64
        else:
            self.d_type = np.complex128

        if n_workers is None:
            if self.nx == self.ny == 4:
                n_workers = 1
            else:
                n_workers = mp.cpu_count()
            #n_workers = 1

        # Split work into chunks for parallel processing
        chunk_size = len(self.representatives) // (n_workers) + 1
        self.debug = 0
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit chunks
            futures = []
            chunk = []
            for repr, state in tqdm(spin_confg_items, mininterval=5., disable = not self.print):
                chunk.append((repr, state))
                
                if len(chunk) >= chunk_size:
                    futures.append(executor.submit(self._process_hamiltonian_batch, chunk.copy()))
                    chunk.clear()
            
            if len(chunk) > 0:
                futures.append(executor.submit(self._process_hamiltonian_batch, chunk.copy()))
                chunk.clear()

            for future in tqdm(as_completed(futures), total = len(futures), disable = not self.print):
                batch_results = future.result()
                self._merge_ham_results(batch_results)

        repr_size = len(self.representatives)

        self.hamiltonian = csr_matrix((array(self.hamiltonian_elements, dtype=self.d_type), (array(self.ham_i, dtype=int), array(self.ham_j, dtype=int))), shape=(repr_size, repr_size))

        self.hamiltonian_elements = None
        self.ham_i = None
        self.ham_j = None

        if self.print:
            print(f'Getting H took: {round(time()-start, 5)} s')
        
    
    def _process_hamiltonian_batch(self, representatives):
        """Process a batch of states for hamiltonian calculation."""
        results = []
        
        map_int_to_basis = self.map_int_to_basis
        map_int_to_config = self.map_int_to_config
        rotation_map = self.rotation_map

        from collections import defaultdict
        flipped_states_elements = defaultdict(self.d_type)
        
        # Directions for Heisenberg exchange
        directions = [(-1, 0), (0, -1)]  # x and y directions
        flips = [self.flips_change_side, self.flips_change_up]  # x and y directions
        
        easy_ks = [0., 2.]
        k = self.k
        is_easy_k = easy_ks.__contains__(k[0]) and easy_ks.__contains__(k[1])
        
        for repr, state in representatives:
            config = map_int_to_config(repr)
            self_energy = self.e0
            
            flipped_states_elements.clear()
            
            rot = config * rotation_map + (1 - config) * (1 - rotation_map)

            for id, dir in enumerate(directions):
                roll_config = roll(config, dir, axis=(0, 1))
                possible_mixing = mod(config + roll_config, 2)

                config_int_shift = array(possible_mixing * flips[id], dtype=float)
                # multiplying to account for sign
                config_int_shift *= (-1.)**config

                for int_shift in config_int_shift[config_int_shift != 0].astype(int):
                    repr_b, trans = self.get_representative[repr + int_shift]
                    basis_index = map_int_to_basis(repr_b)
                    
                    if basis_index < state:
                        phase = 1.
                        if not is_easy_k:
                            phase = np.mean([np.exp( -1j * np.pi * np.dot(k, t) ) for t in trans])
                        
                        element = 0.5 * (phase * np.sqrt(self.norms[repr]/self.norms[repr_b]) ) * (1 + self.t * self.lamb * self.delta * (1 - self.alpha))
                        
                        flipped_states_elements[basis_index] += element
                    
                # Sz-Sz interaction energies
                self_energy -= self.delta * 0.5 * sum(mod(config + roll_config, 2))

                self_energy += self.delta * np.sum(rot * np.roll(rot, dir, axis=(0, 1)))
                
            self_energy -= 4 * self.lamb * self.delta * self.mf_n * np.sum(rot) * self.alpha

            results.append({
                'state': map_int_to_basis(repr),
                'energy': self_energy,
                'flips': flipped_states_elements.copy(),
            })
                    
        return results
    
    def _merge_ham_results(self, batch_results):
        """Merge results from a batch into main storage."""

        for result in batch_results:
            state = result['state']
            self_energy = result['energy']
            flipped_states_elements : dict = result['flips']
            
            self.ham_i.extend([state]*len(flipped_states_elements))
            self.ham_j.extend(flipped_states_elements.keys())
            self.hamiltonian_elements.extend(flipped_states_elements.values())

            self.ham_i.extend(flipped_states_elements.keys())
            self.ham_j.extend([state]*len(flipped_states_elements))
            self.hamiltonian_elements.extend(np.conj(np.array(list(flipped_states_elements.values()))))

            self.ham_i.append(state)
            self.ham_j.append(state)
            self.hamiltonian_elements.append(self_energy)
    

    def precompute_flips_change(self):
        """
        Precomputed change in the int index of the state with the spin flip term in Heisenberg Ham.
        """
        weight_rolled = roll(self.weight_matrix, (0, -1), axis=(0, 1))
        self.flips_change_up = array(self.weight_matrix - weight_rolled, dtype=int)

        weight_rolled = roll(self.weight_matrix, (-1, 0), axis=(0, 1))
        self.flips_change_side = array(self.weight_matrix - weight_rolled, dtype=int)
        
    
    def get_eigva_eigve(self, n_states:int=None):
        """
        Calculates self.n_lowest eigenvectors and eigenenergies of the sparse matrix under the parameter self.hamiltonian. When called with an int parameter, temporarily changes number of calculated eigenstates.

        Args:
            n_states (int, optional): Temporarily changes the value of self.n_lowest in the class and calculates given number of the lowest eigenstates. Defaults to None.

        Returns:
            Energies (np.array[float]): N lowest eigenenergies of the system.
            Vectors (np.array[float]): Corresponding N lowest eigenenvectors of the system. Returns them as the columns of the 2D array (accessed by [:, ii]).
        """
        
        if n_states is None:
            n_lowest = self.n_lowest
        else:
            n_lowest = n_states

        if n_lowest >= self.size:
            raise ValueError(f"Requested {n_lowest} eigenstates, but matrix size is {self.size}")

        self.hamiltonian = (self.hamiltonian + self.hamiltonian.T.conj())/2
        
        energies, states = eigsh(self.hamiltonian, k=n_lowest, which='SA', ncv = max(2*n_lowest + 1, 60), tol=1e-10)

        mf_const = self.n / 2 * (1 - self.alpha) * self.t**2
        mf_const += 2 * self.n * self.alpha * self.mf_n**2
        mf_const *= self.delta * self.lamb

        energies += mf_const

        self.gs_energy = float(np.min(energies))
        self.gs_in_basis = states[:, np.argmin(energies)]

        if n_lowest > 1:
            self.eign_en = energies
            self.eignstates = states

            return energies, states
        
    def get_ground_state(self):
        return self.gs_energy, self.gs_in_basis
        
        
    def get_representations(self, n_workers=None):
        """
        Generates representations of spin states with parallel execution.
        
        Args:
            n_workers: Number of parallel workers. None = use all CPU cores.
        """

        if n_workers is None:
            n_workers = mp.cpu_count()
        
        # Split work into chunks for parallel processing
        chunk_size = self.size // n_workers + 1

        # Initialize results storage
        self.representatives = []
        self.get_representative = {}
        self.norms = {}
        
        # Process in parallel
        if self.print:
            print(f"Processing {self.size} states with {n_workers} workers...")
        
        def bitmask(combo):
            """Generate bitmasks efficiently."""
            val = 0
            for i in combo:
                val |= 1 << (self.n - 1 - i)
            return val

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit chunks
            futures = []
            chunk = []
            for combo in tqdm(combinations(range(self.n), self.magnon_number), total = self.size, mininterval=5., disable= not self.print):
                chunk.append(bitmask(combo))
                
                if len(chunk) >= chunk_size:
                    futures.append(executor.submit(self._process_state_batch, chunk.copy()))
                    chunk.clear()
            
            if len(chunk) > 0:
                futures.append(executor.submit(self._process_state_batch, chunk.copy()))
                chunk.clear()
            
            # Collect results
            for future in as_completed(futures):
                batch_results = future.result()
                self._merge_results(batch_results)
        
        self.representatives.sort()

        # Create final representative enumeration
        self.representatives = {r: i for i, r in enumerate(self.representatives)}

        if self.print:
            print(f'Total representatives found: {len(self.representatives)}')
            print(f'Matrix reduction: {round(self.size/len(self.representatives), 1)}:1')

    
    def _process_state_batch(self, states):
        """Process a batch of states."""
        reprs = []
        norms = []
        flips = []
        
        for state_int in states:
            # Convert to config using the proper mapping
            state_cfg = self.map_int_to_config(state_int).ravel()
            
            set_states = set()
            all_states = []
            is_representative = True
            
            for w, t in self.weight_matrices:
                rolled_state_int = int(w @ state_cfg)
                
                if rolled_state_int < state_int:
                    is_representative = False
                    break
                
                set_states.add(rolled_state_int)
                all_states.append((rolled_state_int, t))
            
            if is_representative:
                reprs.append(state_int)
                norms.append(len(set_states))
                flips.append(all_states)
            
        return (reprs, norms, flips)
    
    def _merge_results(self, batch_results):
        """Merge results from a batch into main storage."""

        states, norms, flips = batch_results

        for state, norm, all_states in zip(states, norms, flips):
            self.representatives.append(state)
            self.norms[state] = norm
            for _state, trans in all_states:
                if self.get_representative.keys().__contains__(_state):
                    self.get_representative[_state][1].append(trans)
                else:
                    self.get_representative[_state] = [state, [trans]]    


    def get_t(self):

        if self.t_matrices is None:
            self.construct_t_matrix()

        return [np.dot(np.conj(self.gs_in_basis), t @ self.gs_in_basis) / self.n for t in self.t_matrices]
    
    def construct_t_matrix(self):
        chunk_size = len(self.representatives) // (mp.cpu_count()) + 1
        self.t_matrices = [lil_matrix((len(self.representatives), len(self.representatives)), dtype=self.d_type) for _ in range(4)]

        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            # Submit chunks
            futures = []
            chunk = []
            for repr, state in tqdm(self.representatives.items(), mininterval=5., disable = not self.print):
                chunk.append((repr, state))
                
                if len(chunk) >= chunk_size:
                    futures.append(executor.submit(self._process_t_batch, chunk.copy()))
                    chunk.clear()
            
            if len(chunk) > 0:
                futures.append(executor.submit(self._process_t_batch, chunk.copy()))
                chunk.clear()

            for future in tqdm(as_completed(futures), total = len(futures), disable = not self.print):
                res = future.result()
                for ii in range(4):
                    self.t_matrices[ii] += res[ii]

        for ii in range(4):
            self.t_matrices[ii] = csr_matrix(self.t_matrices[ii])

    def _process_t_batch(self, representatives):

        map_int_to_basis = self.map_int_to_basis
        map_int_to_config = self.map_int_to_config
        
        # Directions for Heisenberg exchange
        directions = [(-1, 0), (0, -1)]  # x and y directions
        flips = [self.flips_change_side, self.flips_change_up]  # x and y directions

        small_t_matrix = [lil_matrix((len(self.representatives), len(self.representatives)), dtype=self.d_type) for _ in range(4)]

        for repr, state in representatives:
            config = map_int_to_config(repr)            

            for id, dir in enumerate(directions):
                roll_config = roll(config, dir, axis=(0, 1))
                possible_mixing = mod(config + roll_config, 2)

                config_int_shift = array(possible_mixing * flips[id], dtype=float)
                # multiplying to account for sign
                config_int_shift *= (-1.)**config

                for int_shift in config_int_shift[config_int_shift != 0].astype(int):
                    repr_b, _ = self.get_representative[repr + int_shift]
                    basis_index = map_int_to_basis(repr_b)
                    
                    if basis_index < state:
                        
                        element = 0.5 * np.sqrt(self.norms[repr]/self.norms[repr_b])
                        x, y = np.where(config_int_shift == int_shift)
                        x = x[0]
                        y = y[0]
                        small_t_matrix[2*id + (x + y)%2][basis_index, state] += np.conj(element)
                        small_t_matrix[2*id + (x + y)%2][state, basis_index] += element

        return small_t_matrix

    def get_n(self):
        """Get mean rotated bosons in ground state."""
        gs_state = self.get_ground_state()[1]

        if self.n_in_basis is None:
            self.n_in_basis = np.zeros((2, len(self.representatives)))

            rotation_map = self.rotation_map
            
            for repr, state in self.representatives.items():
                config = self.map_int_to_config(repr)
                rot = config * rotation_map + (1 - config) * (1 - rotation_map)
                self.n_in_basis[0, state] = np.mean(rot * rotation_map)
                self.n_in_basis[1, state] = np.mean(rot * (1 - rotation_map))
        
        return np.dot(self.n_in_basis, np.square(np.abs(gs_state)))

    def get_mag_int(self):
        """Get mean rotated bosons in ground state."""
        gs_state = self.get_ground_state()[1]

        mag_int = 0
        rotation_map = self.rotation_map
        
        for repr, amp in zip(self.representatives.keys(), gs_state):
            config = self.map_int_to_config(repr)
            rot = config * rotation_map + (1 - config) * (1 - rotation_map)

            mag_int += np.square(np.abs(amp)) * np.sum(rot * np.roll(rot, (0, -1), axis = (0, 1)))
            mag_int += np.square(np.abs(amp)) * np.sum(rot * np.roll(rot, (-1, 0), axis = (0, 1)))

        return mag_int / self.n

    def go_sc_mf(self, num_steps = 20, n0 = 0., t0 = 0., running_params = False, show_tqdm = True):
        """
        Performs self-consistent mean-field iterations num_steps times and returns expectation values of n_0, t and ground state energy.

        If running params is True: returns the whole range of parameters at each iteration.
        """
        mf_n = n0
        t = t0
        mfs = []
        ts = []
        ens = []

        for _ in tqdm(range(num_steps), disable = not show_tqdm):
            self.get_hamiltonian(mf_n=mf_n, t = t)
            self.get_eigva_eigve(2)

            mf_n = self.get_n()
            t = self.get_t()
            min_e = self.get_ground_state()[0]

            mfs.append(mf_n)
            ts.append(t)
            ens.append(min_e/self.n)
        
        if running_params:
            return mfs, ts, ens
        else:
            return mfs[-1], ts[-1], ens[-1]

    def map_int_to_config(self, n):
        """Map integer to configuration."""
        return array([(n >> i) & 1 for i in reversed(range(self.n))], dtype=uint8).reshape((self.ny, self.nx))
    
    def map_int_to_basis(self, n):
        """Map integer to basis index."""
        return self.representatives[n]
    
    def map_config_to_int(self, config):
        """Map configuration to integer."""
        return int(dot(self.weight_matrix.ravel(), config.ravel()))
    
    def map_config_to_basis(self, config):
        """Map configuration to basis index."""
        idx = self.map_config_to_int(config)
        return self.representatives[idx]


if __name__ == "__main__":

    nx = 6
    ny = 4
    N = nx * ny
    
    sc = SpinConfiguration(nx, ny, magnon_number=int(N/2)+0, lowest_eignstates=1,
                        delta=1., lamb = 0., k=np.array([0., 0.]), print_data=False, alpha=1/2)
    
    print(f'Before convergence: I = {sc.get_mag_int()}')
    mfs, ts, min_e = sc.go_sc_mf(1)

    #min_e = sc.get_ground_state()[0]
    print(f'GS energy per-site: e_0/J = {round(min_e, 5)}')
    print(f'n_0 = {mfs}')
    print(f't = {ts}')
    print(f'I = {sc.get_mag_int()}')
