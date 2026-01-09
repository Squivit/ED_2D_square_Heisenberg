from numpy import roll, array, mod, dot, uint8, sum, ones
import numpy as np
from itertools import combinations
from time import time

from scipy.special import binom

from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt

from Betts_cluster import Cluster

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

class SpinConfiguration:
    
    def __init__(self, N = 16, number = 'A', key = None, magnon_number = -1, delta = 1., lamb= 1., J2overJ1 = 1., lowest_eignstates = 1, print_data = False, tol : float = 1e-8):
        """
        k is in units of pi
        """
        
        self.print = print_data
        self.n_lowest = lowest_eignstates
                
        if key == None:
            self.n = N
            key = str(N)+number
        else:
            self.n = int(key[:2])
            number = key[-1]

        if magnon_number == -1:
            self.magnon_number = int(self.n/2)
        else:
            self.magnon_number = magnon_number

        self.delta = delta
        self.j2 = J2overJ1
        self.lamb = lamb
        
        side = 24
        if self.n > 26:
            side = 32
            print('WARNING! For more than 24 spin bigmap of size 26 might be too small!')
        
        self.cluster = Cluster(N=self.n, num=number, key=key, bigmap_shift=0, bigmap_side=side)
        
        #get_betts_cluster(key, shift = 0, plot=True)
        #plt.show()
                
        self.tol = tol
        
        self.weight_matrix = self.cluster.bosons_to_cluster_rolled(array([ 2**(ii-1) for ii in range(self.n, 0, -1)]))
        
        self.precompute_flips_change()
        
        self.rotation_map = ones(self.cluster.cluster_map.shape)
        self.rotation_map = self.rotation_map * array( [ [ (1+(-1)**(x+y))/2 for x in range(self.rotation_map.shape[0]) ] for y in range(self.rotation_map.shape[1]) ] )
                
        self.eign_en = []
        self.eignstates = []
        
        if self.magnon_number > self.n:
            raise ValueError('Magnon number should be smaller than the system size')
        
        self.size = int(binom(self.n, self.magnon_number))
        #self.generate_spin_configurations()
        self.get_representations()

           
        start = time()
            
        self.generate_hamiltonian()
        
        if print_data:
            print(f'Getting H took: {round(time()-start, 5)} s')
        start = time()
        
        self.get_eigva_eigve()
        
        if print_data:
            print(f'Finding eigvals and eigvecs took: {round(time()-start, 5)} s')
        
    
    def generate_hamiltonian(self, k = None, lamb = None, delta = None):
        """
        Generates Hamiltonian of the spin system by calulating all coefficients given by Heisenberg hamiltonian.
        Saves the result as sparse matrix (scipy.sparse.csr_matrix) under the value self.hamiltonian
        """
        
        if k is None:
            k = np.array( [0, 0] )

        if lamb is not None:
            self.lamb = lamb

        if delta is not None:
            self.delta = delta
    

        hamiltonian_elements = []
        ham_i = []
        ham_j = []

        # Directions for Heisenberg exchange
        directions = [(-1, 0), (0, -1)]  # x and y directions
        flips = [self.flips_change_side, self.flips_change_up]  # x and y directions
        interactions = [1., self.j2] # interactions in x and y axis

        e0 = 1. * self.n
        e0 += self.j2 * self.n
        e0 *= self.delta / 4.

        # Precompute mappings
        spin_confg_items = list(self.representatives.items())

        map_int_to_basis = self.map_int_to_basis
        map_int_to_config = self.map_int_to_config_extended
        rotation_map = self.rotation_map
        cluster_map = self.cluster.cluster_map
        
        easy_ks = [0., 2.]
        flipped_states_elements = {}
        
        for config_int, state in tqdm(spin_confg_items, disable=not self.print):
            config = map_int_to_config(config_int)
            self_energy = e0
            
            flipped_states_elements.clear()
            
            if self.lamb != 1:
                rot = (config * rotation_map + (1 - config) * (1 - rotation_map))

            for id, dir in enumerate(directions):
                roll_config = roll(config, dir, axis=(0, 1))
                                
                possible_mixing = mod((config + roll_config) * cluster_map, 2)

                config_int_shift = array(possible_mixing * flips[id], dtype=float)
                # multiplying to account for sign
                config_int_shift *= (-1.)**config

                for int_shift in config_int_shift[config_int_shift != 0].astype(int):
                    repr_b, trans = self.get_representative[config_int + int_shift]
                    basis_index = map_int_to_basis(repr_b)
                    
                    if basis_index < state:
                        phase = 0.
                        if not (easy_ks.__contains__(k[0]) and easy_ks.__contains__(k[1])):
                            for t in trans:
                                phase += np.exp( -1j * np.pi * np.dot(k, t) )
                            phase /= len(trans)
                        else:
                            phase = 1.
                        
                        element = 0.5 * interactions[id] * phase * np.sqrt(self.norms[config_int]/self.norms[repr_b])
                        
                        # there is such a state in the list already
                        if flipped_states_elements.keys().__contains__(basis_index):
                            flipped_states_elements[basis_index] += element
                        else:
                            flipped_states_elements[basis_index] = element
                    
            
                # Sz-Sz interaction energies
                self_energy -= self.delta * 0.5 * interactions[id] * sum(mod(cluster_map*(config + roll_config), 2))

                if self.lamb != 1:
                    self_energy -= interactions[id] * self.delta * (self.lamb - 1) * sum(cluster_map*(rot * roll(rot, dir, axis=(0, 1))))

            
            for basis_state, element in flipped_states_elements.items():
                ham_i.append(state)
                ham_j.append(basis_state)
                hamiltonian_elements.append(element)            
            
            ham_i.append(state)
            ham_j.append(state)
            hamiltonian_elements.append(self_energy /2.)
            
        repr_size = len(self.representatives)

        if easy_ks.__contains__(k[0]) and easy_ks.__contains__(k[1]):
            d_type = np.float32
        else:
            d_type = np.complex64

        self.hamiltonian = csr_matrix((array(hamiltonian_elements, dtype=d_type), (array(ham_i, dtype=int), array(ham_j, dtype=int))), shape=(repr_size, repr_size))

    
    def precompute_flips_change(self):
        """
        Precomputed change in the int index of the state with the spin flip term in Heisenberg Ham.
        """
        weight_rolled = roll(self.weight_matrix, (0, -1), axis=(0, 1))
        self.flips_change_up = array(self.cluster.cluster_map*(self.weight_matrix - weight_rolled), dtype=int)

        weight_rolled = roll(self.weight_matrix, (-1, 0), axis=(0, 1))
        self.flips_change_side = array(self.cluster.cluster_map*(self.weight_matrix - weight_rolled), dtype=int)
        
    
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

        energies, states = eigsh(self.hamiltonian, k=n_lowest, which='SA', tol=self.tol)

        # Make correct hamiltonian (now it's only up-triangle with diagonal reduced by a factor of 2)
        H = (self.hamiltonian + np.conj(self.hamiltonian.T))
        
        energies, states = eigsh(H, k=n_lowest, which='SA', tol=self.tol)

        self.gs_energy = float(energies[0])
        self.gs_in_basis = states[:, 0]

        if n_lowest > 1:
            self.eign_en = energies
            self.eignstates = states
            return energies, states


    def get_ground_state(self):
        return self.gs_energy, self.gs_in_basis

    def generate_spin_configurations(self):
        def bitmask(combo):
            val = 0
            for i in combo:
                val |= 1 << (self.n - 1 - i)
            return val

        combos = list(combinations(range(self.n), self.magnon_number))
        # dictionary where:
        # keys: int representation
        # values: number of the state in the basis
        self.spin_confg = {bitmask(c): i for i, c in enumerate(combos)}

        if self.print:
            print(f"Generated {self.size} spin configurations")
    
        
    def get_representations(self, n_workers=None):
        """
        Generates representations of spin states with parallel execution.
        
        Args:
            n_workers: Number of parallel workers. None = use all CPU cores.
        """
        if n_workers is None:
            n_workers = mp.cpu_count()
        
        coords = self.cluster.sorted_coords
        cluster_map = self.cluster.cluster_map
        
        # Pre-compute translations (even only)
        translations = [trans for trans in coords if np.sum(trans) % 2 == 0]
        
        # Pre-compute weight matrices and store as instance variable
        self.weight_matrices = [
            ((cluster_map * np.roll(self.weight_matrix, np.array(t), axis=(0, 1))).ravel(), t)
            for t in translations
        ]
        
        # Pre-compute bitmasks for all combinations
        all_states = list(self._generate_bitmasks())
        
        # Split work into chunks for parallel processing
        chunk_size = min(len(all_states) // n_workers + 1, 25000)
        
        # Initialize results storage
        self.representatives = []
        self.norms = {}
        self.get_representative = {}
        
        # Process in parallel
        if self.print:
            print(f"Processing {len(all_states)} states with {n_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit chunks
            futures = []
            for i in range(0, len(all_states), chunk_size):
                chunk = all_states[i:i + chunk_size]
                futures.append(executor.submit(self._process_state_batch, chunk))
            
            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), disable=not self.print):
                batch_results = future.result()
                self._merge_results(batch_results)
        
        # Create final representative enumeration
        self.representatives = {r: i for i, r in enumerate(self.representatives)}
        
        if self.print:
            print(f'Total representatives found: {len(self.representatives)}')
            print(f'Matrix reduction: {round(self.size/len(self.representatives), 1)}:1')
    
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
        """Map integer to configuration."""
        return self.cluster.bosons_to_cluster(
            array([(n >> i) & 1 for i in reversed(range(self.n))], dtype=uint8)
        )
    
    def map_int_to_config_extended(self, n):
        """Map integer to extended configuration."""
        return self.cluster.bosons_to_cluster_rolled(
            array([(n >> i) & 1 for i in reversed(range(self.n))], dtype=uint8)
        )
    
    def map_int_to_basis(self, n):
        """Map integer to basis index."""
        return self.representatives[n]
    
    def map_config_to_int(self, config):
        """Map configuration to integer."""
        return int(dot(self.weight_matrix.ravel(), config.ravel()).real)
    
    def map_config_to_basis(self, config):
        """Map configuration to basis index."""
        idx = self.map_config_to_int(config)
        return self.representatives[idx]


if __name__ == "__main__":
    
    N = 24
    num = 'A'
    
    sc = SpinConfiguration(N=N, number=num, key=str(N)+num, lowest_eignstates=1, delta=1., lamb=1., print_data=True)
    min_e, gs_state = sc.get_ground_state()
    
    print(f'Ground state energy E_0/J = {round(min_e, 5)}')
    print(f'GS energy per-vertex: e_0/J = {round(-2*min_e/(N), 5)}')
    
    eign = sc.get_eigva_eigve(5)[0]
    print(np.round(eign, 6))
