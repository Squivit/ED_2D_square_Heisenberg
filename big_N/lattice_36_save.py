from numpy import roll, array, mod, dot, uint8, sum, ones
import numpy as np
from itertools import combinations
from time import time
import os

from scipy.special import binom

from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, save_npz, load_npz

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from paths import *
from tqdm import tqdm

class SpinConfiguration:
    
    def __init__(self, nx = 6, ny = 6, magnon_number = -1, delta = 1., k = np.array([0., 0.]), n_workers = None,
                 lowest_eignstates = 1, print_data = False, force_ham_gen = False, save_ham = True, eigva_ve_only = False, helper_mode = False):
        """
        k is in units of pi
        """

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
        self.k = k
                                        
        self.weight_matrix = array([ 2**(ii-1) for ii in range(self.n, 0, -1)]).reshape((self.ny, self.nx))
        
        self.precompute_flips_change()
        
        self.rotation_map = array( [ [ (1+(-1)**(x+y))/2 for x in range(self.nx) ] for y in range(self.ny) ] ).reshape((self.ny, self.nx))
        
        xs = np.arange(self.nx)
        ys = np.arange(self.ny)
        all_tr = [ [ (-int(y), -int(x)) for x in xs ] for y in ys ]

        # Pre-compute translations (even only)
        translations = []
        
        for trans in all_tr:
            for t in trans:
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
        
        self.size = int(binom(self.n - 1, self.magnon_number))

        if not eigva_ve_only:
            self.get_representations(n_workers=n_workers)

        if not helper_mode:

            self.save_ham = save_ham
            self.get_hamiltonian(force_ham_gen=force_ham_gen, n_workers=n_workers)
            
            #start = time()
            #self.get_eigva_eigve()
            #if print_data:
            #    print(f'Finding eigvals and eigvecs took: {round(time()-start, 5)} s')
        
        if self.print:
            print('Finished process')
    

    def get_hamiltonian(self, k = None, delta = None, force_ham_gen = False, n_workers = None):
        
        if k is None:
            k = self.k

        if delta is not None:
            self.delta = delta
    
        self.k = k
        
        # forcing generation of the new hamiltonian
        if force_ham_gen:
            self.generate_hamiltonian(k, n_workers=n_workers)
            return
        
        if self.print:
            print('Trying to read hamiltonian')
        
        hamilonians = [f for f in os.listdir(HAMILTONIAN_PATH) if os.path.isfile(os.path.join(HAMILTONIAN_PATH, f))]

        # there is a correct hamiltonian saved
        if hamilonians.__contains__(f'{self.nx}x{self.ny}_k={k}_d={self.delta}_Sz={abs(int(self.n/2) - self.magnon_number)}.npz'):
            self.hamiltonian = load_npz(fr'{HAMILTONIAN_PATH}/{self.nx}x{self.ny}_k={k}_d={self.delta}_Sz={abs(int(self.n/2) - self.magnon_number)}.npz')
            if self.print:
                print('Hamiltonian loaded successfully.')

        # no hamiltonian, create and save one
        else:
            self.generate_hamiltonian(k, n_workers=n_workers)
    
    
    def generate_hamiltonian(self, k = None, n_workers = None):
        """
        Generates Hamiltonian of the spin system by calulating all coefficients given by Heisenberg hamiltonian.
        Saves the result as sparse matrix (scipy.sparse.csr_matrix) under the value self.hamiltonian
        """

        if self.print:
            print('No hamiltonian to read or forced to start generation.')

        start = time()

        if k is not None:
            self.k = k
        
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
            n_workers = mp.cpu_count()

        # Split work into chunks for parallel processing
        chunk_size = len(self.representatives) // (n_workers) + 1
        
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

        self.hamiltonian = csr_matrix((array(self.hamiltonian_elements, dtype=self.d_type), (array(self.ham_i, dtype=np.int64), array(self.ham_j, dtype=np.int64))), shape=(repr_size, repr_size))

        if self.print:
            print(f'Getting H took: {round(time()-start, 5)} s')
        
        if self.save_ham:
            self.save_hamiltonian()

    
    def _process_hamiltonian_batch(self, representatives):
        """Process a batch of states for hamiltonian calculation."""

        state_results = []
        energy_results = []
        flip_state_results = []
        flip_energy_results = []
        
        map_int_to_basis = self.map_int_to_basis
        map_int_to_config = self.map_int_to_config

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
            
            for id, dir in enumerate(directions):
                roll_config = roll(config, dir, axis=(0, 1))
                                
                possible_mixing = mod(config + roll_config, 2)

                config_int_shift = array(possible_mixing * flips[id], dtype=float)
                # multiplying to account for sign
                config_int_shift *= (-1.)**config

                for int_shift in config_int_shift[config_int_shift != 0].astype(int):
                    repr_b, trans = self.roll_to_repr(repr + int_shift)
                    basis_index = map_int_to_basis(repr_b)
                    
                    if basis_index < state:
                        phase = 1.
                        if not is_easy_k:
                            phase = np.mean([np.exp( -1j * np.pi * np.dot(k, t) ) for t in trans])
                        
                        element = 0.5 * phase * np.sqrt(self.norms[repr]/self.norms[repr_b])
                        
                        flipped_states_elements[basis_index] += element
                    
                # Sz-Sz interaction energies
                self_energy -= self.delta * 0.5 * sum(mod(config + roll_config, 2))
            
            state_results.append(map_int_to_basis(repr))
            energy_results.append(self_energy)
            flip_state_results.append(list(flipped_states_elements.keys()))
            flip_energy_results.append(list(flipped_states_elements.values()))
                    
        return (state_results, energy_results, flip_state_results, flip_energy_results)
    
    def _merge_ham_results(self, batch_results):
        """Merge results from a batch into main storage."""

        state_results, energy_results, flip_state_results, flip_energy_results = batch_results

        for ir in range(len(state_results)):
            state = state_results[ir]
            self_energy = energy_results[ir]
            flipped_states : list = flip_state_results[ir]
            flip_energies : list = flip_energy_results[ir]
            
            self.ham_i.extend([state]*len(flipped_states))
            self.ham_j.extend(flipped_states)
            self.hamiltonian_elements.extend(flip_energies)

            self.ham_i.extend(flipped_states)
            self.ham_j.extend([state]*len(flipped_states))
            self.hamiltonian_elements.extend(np.conj(np.array(flip_energies)))

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

        self.gs_energy = float(np.min(energies))
        self.gs_in_basis = states[:, np.argmin(energies)]

        if n_lowest > 1:
            self.eign_en = energies
            self.eignstates = states

            self.save_eign_eigva()

            return energies, states
        
        self.save_eign_eigva()

    def get_ground_state(self):
        return self.gs_energy, self.gs_in_basis
        
        
    def get_representations(self, n_workers=None):
        """
        Generates representations of spin states with parallel execution.
        
        Args:
            n_workers: Number of parallel workers. None = use all CPU cores.
        """

        # checks if successfully loaded representatives
        if self.load_representatives():
           return

        if n_workers is None:
            n_workers = mp.cpu_count()
        
        # Split work into chunks for parallel processing
        chunk_size = self.size // n_workers + 1

        # Initialize results storage
        self.representatives = []
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
            for combo in tqdm(combinations(range(1, self.n), self.magnon_number), total = self.size, mininterval=5.):
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
            print(f'Matrix reduction: {round(2*self.size/len(self.representatives), 1)}:1')

        self.save_representatives()

    
    def _process_state_batch(self, states):
        """Process a batch of states."""
        state_results = []
        norm_results = []
        
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
                state_results.append(state_int)
                norm_results.append(len(set_states))
            
        return (state_results, norm_results)
    

    def _merge_results(self, batch_results):
        """Merge results from a batch into main storage."""

        state_results, norm_results = batch_results

        for state, norm in zip(state_results, norm_results):
            self.representatives.append(state)
            self.norms[state] = norm
    
    
    def roll_to_repr(self, state_int : int):

        state_cfg = self.map_int_to_config(state_int).ravel()

        min_int = state_int
        translations = []
        
        for w, trans in self.weight_matrices:
            rolled_state_int = int(np.dot(w, state_cfg))

            # it is the same representative under this translation
            if rolled_state_int == min_int:
                translations.append(trans)
            # we found a better representative
            elif rolled_state_int < min_int:
                min_int = rolled_state_int
                translations = [trans]
        
        return min_int, translations
    
    
    def save_hamiltonian(self):
        try:
            os.makedirs(f'{HAMILTONIAN_PATH}', exist_ok=False)
        except:
            pass

        fname = fr'{HAMILTONIAN_PATH}/{self.nx}x{self.ny}_k={self.k}_d={self.delta}_Sz={abs(int(self.n/2) - self.magnon_number)}.npz'

        save_npz(fname, self.hamiltonian)

    def save_eign_eigva(self):
        try:
            os.makedirs(f'{EIGN_PATH}', exist_ok=False)
        except:
            pass

        try:
            fname = fr'{EIGN_PATH}/{self.nx}x{self.ny}_k={self.k}_d={self.delta}_Sz={abs(int(self.n/2) - self.magnon_number)}_en.txt'
            np.savetxt(fname, self.eign_en)

            fname = fr'{EIGN_PATH}/{self.nx}x{self.ny}_k={self.k}_d={self.delta}_Sz={abs(int(self.n/2) - self.magnon_number)}_vec.txt'
            np.savetxt(fname, self.eignstates)
        except:
            pass
    

    def load_representatives(self):
        try:
            self.representatives = np.loadtxt(fr'{REPR_PATH}/{self.nx}x{self.ny}_Sz={abs(int(self.n/2) - self.magnon_number)}.txt')
            self.representatives = np.array(self.representatives, dtype = np.int64)
            self.representatives = {r: i for i, r in enumerate(self.representatives)}

            self.norms = np.loadtxt(fr'{REPR_PATH}/{self.nx}x{self.ny}_norms_Sz={abs(int(self.n/2) - self.magnon_number)}.txt')
            keys = np.loadtxt(fr'{REPR_PATH}/{self.nx}x{self.ny}_norm_keys_Sz={abs(int(self.n/2) - self.magnon_number)}.txt')
            self.norms = {np.int64(r): float(n) for r, n in zip(keys, self.norms)}
            keys = None

            if self.print:
                print('Representatives loaded successfully')
            return True
        except:
            return False
    
    def save_representatives(self):
        try:
            os.makedirs(f'{REPR_PATH}', exist_ok=False)
        except:
            pass

        try:
            np.savetxt(fr'{REPR_PATH}/{self.nx}x{self.ny}_Sz={abs(int(self.n/2) - self.magnon_number)}.txt', np.array(list(self.representatives.keys()), dtype = int))
            np.savetxt(fr'{REPR_PATH}/{self.nx}x{self.ny}_norms_Sz={abs(int(self.n/2) - self.magnon_number)}.txt', list(self.norms.values()))
            np.savetxt(fr'{REPR_PATH}/{self.nx}x{self.ny}_norm_keys_Sz={abs(int(self.n/2) - self.magnon_number)}.txt', list(self.norms.keys()))
        except:
            pass


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

    nx = 4
    ny = 4
    N = nx * ny
    
    sc = SpinConfiguration(nx, ny, magnon_number=int(N/2)+0, lowest_eignstates=1, n_workers = 20,
                        delta=1., k=np.array([0., 0.]), print_data=True, force_ham_gen=True, eigva_ve_only=False, save_ham=False)
    min_e, gs_state = sc.get_ground_state()
        
    print(f'Ground state energy E_0/J = {round(min_e, 5)}')
    print(f'GS energy per-site: e_0/J = {round(min_e/(N), 5)}')
