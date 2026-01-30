import numpy as np
from itertools import combinations

from Betts_cluster import Cluster

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from paths import *

class Parameter_Changer():

    def __init__(self, key = '24A', magnon_number = -1):
        self.n = int(key[:2])
        self.key = key
        
        if magnon_number == -1:
            self.magnon_number = int(self.n/2)
        else:
            self.magnon_number = magnon_number

        # side lenghts optimized for 18A, 24A and 32A
        side = 18
        if self.n > 24:
            side = 24
        
        self.cluster = Cluster(key=key, bigmap_shift=0, bigmap_side=side)
                                
        self.weight_matrix = self.cluster.bosons_to_cluster_rolled(np.array([ 2**(ii-1) for ii in range(self.n, 0, -1)]))
        self.rotation_map = np.ones(self.cluster.cluster_map.shape)
        self.rotation_map = self.rotation_map * np.array( [ [ (1+(-1)**(x+y))/2 for x in range(self.rotation_map.shape[0]) ] for y in range(self.rotation_map.shape[1]) ] )

        self.precompute_flips_change()

        #self.size = int(binom(self.n, self.magnon_number))
        #self.get_representations()

        self.representatives = np.loadtxt(fr'{REPR_PATH}/{key}_Sz={abs(int(self.n/2) - self.magnon_number)}.txt')
        self.representatives = {int(r): i for i, r in enumerate(self.representatives)}


    def change_parameters(self, ham_csr, d = 1., l = 1., d_i = 1., l_i = 1., _print = False):

        self.ham_csr = ham_csr
        
        self.n = int(self.key[:2])

        self.d_f = d
        self.d_i = d_i

        self.l_f = l
        self.l_i = l_i

        n_workers = mp.cpu_count()

        spin_confg_items = list(self.representatives.items())

        # Split work into chunks for parallel processing
        chunk_size = len(self.representatives) // (n_workers) + 1
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit chunks
            futures = []
            chunk = []
            for repr, state in tqdm(spin_confg_items, total=len(self.representatives), disable=not _print):
                chunk.append((repr, state))
                
                if len(chunk) >= chunk_size:
                    futures.append(executor.submit(self._process_hamiltonian_batch, chunk.copy()))
                    chunk.clear()
            
            if len(chunk) > 0:
                futures.append(executor.submit(self._process_hamiltonian_batch, chunk.copy()))
                chunk.clear()

            for future in tqdm(as_completed(futures), total=len(futures), disable = not _print):
                batch_results = future.result()
                self._merge_ham_results(batch_results)
            
        return self.ham_csr



    def _process_hamiltonian_batch(self, representatives):
        """Process a batch of states for hamiltonian calculation."""
        results = []
        
        # Directions for Heisenberg exchange
        directions = [(-1, 0), (0, -1)]  # x and y directions
        flips = [self.flips_change_side, self.flips_change_up]  # x and y directions

        # two bonds per site
        e0 = 1. * 2 * self.n
        e0 *= (self.d_f - self.d_i) / 4.

        map_int_to_basis = self.map_int_to_basis
        map_int_to_config = self.map_int_to_config_extended
        rotation_map = self.rotation_map
        cluster_map = self.cluster.cluster_map

        lamb = self.l_f - self.l_i
        delta = self.d_f - self.d_i
                
        for repr, state in representatives:
            config = map_int_to_config(repr)
            self_energy = e0
                    
            if lamb != 0:
                rot = (config * rotation_map + (1 - config) * (1 - rotation_map))

            for id, dir in enumerate(directions):
                roll_config = np.roll(config, dir, axis=(0, 1))
                                
                possible_mixing = np.mod((config + roll_config) * cluster_map, 2)

                config_int_shift = np.array(possible_mixing * flips[id], dtype=float)
                # multiplying to account for sign
                config_int_shift *= (-1.)**config

                # Sz-Sz interaction energies
                self_energy -= delta * 0.5 * np.sum(np.mod(cluster_map*(config + roll_config), 2))

                if lamb != 0:
                    self_energy -= ( self.d_f * (self.l_f - 1) - self.d_i * (self.l_i - 1) ) * np.sum(cluster_map*(rot * np.roll(rot, dir, axis=(0, 1))))
            
            results.append({
                'state': state,
                'energy': self_energy,
            })
            
        return results
    
    def _merge_ham_results(self, batch_results):
        """Merge results from a batch into main storage."""
        for result in batch_results:
            state = result['state']
            self_energy = result['energy']
            
            self.ham_csr[state, state] += self_energy /2.


    def get_representations(self, n_workers=None):
        """
        Generates representations of spin states with parallel execution.
        
        Args:
            n_workers: Number of parallel workers. None = use all CPU cores.
        """

        # checks if successfully loaded representatives
        #if self.load_representatives():
        #    return

        if n_workers is None:
            n_workers = mp.cpu_count()
        
        # Split work into chunks for parallel processing
        chunk_size = min(self.size // n_workers + 1, 500000)

        # Initialize results storage
        self.representatives = []
        self.norms = {}
        
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
            for combo in tqdm(combinations(range(self.n), self.magnon_number), total = self.size, mininterval=5.):
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
        
        print(f'Total representatives found: {len(self.representatives)}')

    
    def _process_state_batch(self, states):
        """Process a batch of states."""
        results = []
        
        for state_int in states:
            # Convert to config using the proper mapping
            state_cfg = self.map_int_to_config(state_int).ravel()
            
            set_states = set()
            is_representative = True
            
            for w, _ in self.weight_matrices:
                rolled_state_int = int(w @ state_cfg)
                
                if rolled_state_int < state_int:
                    is_representative = False
                    break
                
                set_states.add(rolled_state_int)
            
            if is_representative:
                results.append({
                    'state': state_int,
                    'norm': len(set_states),
                })
            
        return results
    
    def _merge_results(self, batch_results):
        """Merge results from a batch into main storage."""
        for result in batch_results:
            state = result['state']
            norm = result['norm']
            
            self.representatives.append(state)
            self.norms[state] = norm


    def precompute_flips_change(self):
        """
        Precomputed change in the int index of the state with the spin flip term in Heisenberg Ham.
        """
        weight_rolled = np.roll(self.weight_matrix, (0, -1), axis=(0, 1))
        self.flips_change_up = np.array(self.cluster.cluster_map*(self.weight_matrix - weight_rolled), dtype=int)

        weight_rolled = np.roll(self.weight_matrix, (-1, 0), axis=(0, 1))
        self.flips_change_side = np.array(self.cluster.cluster_map*(self.weight_matrix - weight_rolled), dtype=int)
    
    def map_int_to_config(self, n):
        """Map integer to configuration."""
        return self.cluster.bosons_to_cluster(
            np.array([(n >> i) & 1 for i in reversed(range(self.n))], dtype=np.uint8)
        )
    
    def map_int_to_config_extended(self, n):
        n = int(n)
        """Map integer to extended configuration."""
        return self.cluster.bosons_to_cluster_rolled(
            np.array([(n >> i) & 1 for i in reversed(range(self.n))], dtype=np.uint8)
        )
    
    def map_int_to_basis(self, n):
        """Map integer to basis index."""
        return self.representatives[n]
    
    def map_config_to_int(self, config):
        """Map configuration to integer."""
        return int(np.dot(self.weight_matrix.ravel(), config.ravel()).real)
    
    def map_config_to_basis(self, config):
        """Map configuration to basis index."""
        idx = self.map_config_to_int(config)
        return self.representatives[idx]
