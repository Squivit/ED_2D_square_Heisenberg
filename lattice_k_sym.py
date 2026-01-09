import numpy as np
from itertools import combinations
from scipy.special import binom
from time import time
import matplotlib.pyplot as plt

from scipy.sparse.linalg import eigsh, norm
from scipy.sparse import csr_matrix
from numpy import array as cparray

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

class SpinConfiguration:
    
    def __init__(self, N_x = 4, N_y = 4, magnon_number = 8, delta = 1., lamb= 1., J2overJ1 = 1.,
                 lowest_eignstates = 1, print_data = False, tol : float = 1e-8):
        """
        k is in units of pi
        """
        
        self.print = print_data
        self.n_lowest = lowest_eignstates
        
        self.magnon_number = magnon_number
        self.n_y = N_y
        self.n_x = N_x
        self.n = N_x * N_y
                
        self.delta = delta
        self.j2 = J2overJ1
        self.lamb = lamb
        
        self.tol = tol
        
        self.weight_matrix = np.array([ 2**(ii-1) for ii in range(self.n, 0, -1)]).reshape((self.n_y, self.n_x))
        
        self.precompute_flips_change()
        
        self.rotation_map = np.array( [ [ (1 + (-1)**(x + y))/2 for x in range(self.n_x) ] for y in range(self.n_y) ] ).reshape((self.n_y, self.n_x))

        self.eign_en = []
        self.eignstates = []
        
        if self.magnon_number > self.n:
            raise ValueError('Magnon number should be smaller than the system size')
        
        self.size = int(binom(self.n, self.magnon_number))
        self.get_representations()

        start = time()
        
        self.generate_hamiltonian()
        
        if print_data:
            print(f'Getting H took: {time()-start} s')
        start = time()
        
        self.get_eigva_eigve()
        
        if print_data:
            print(f'Finding eigvals and eigvecs took: {time()-start} s')

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
        directions = [(0, -1), (-1, 0)]  # x and y directions
        flips = [self.flips_change_side, self.flips_change_up]  # x and y directions
        interactions = [1., self.j2] # interactions in x and y axis

        e0 = 1. * self.n
        e0 += self.j2 * self.n
        e0 *= self.delta / 4.

        # Precompute mappings
        spin_confg_items = list(self.representatives.items())

        map_int_to_basis = self.map_int_to_basis
        map_int_to_config = self.map_int_to_config
        rotation_map = self.rotation_map
        
        easy_ks = [0., 2.]
        is_easy_k = easy_ks.__contains__(k[0]) and easy_ks.__contains__(k[1])
        flipped_states_elements = {}


        for config_int, state in (spin_confg_items):
            config = map_int_to_config(config_int)
            self_energy = e0
            
            flipped_states_elements.clear()
            
            if self.lamb != 1:
                rot = (config * rotation_map + (1 - config) * (1 - rotation_map))

            for id, dir in enumerate(directions):
                roll_config = np.roll(config, dir, axis=(0, 1))
                possible_mixing = np.mod(config + roll_config, 2)

                config_int_shift = np.array(possible_mixing * flips[id], dtype=float)
                # multiplying to account for sign
                config_int_shift *= (-1.)**config

                for int_shift in config_int_shift[config_int_shift != 0].astype(int):
                    repr_b, trans = self.get_representative[config_int + int_shift]
                    basis_index = map_int_to_basis(repr_b)

                    if basis_index < state:

                        if not is_easy_k:
                            phase = np.mean([np.exp( -1j * np.pi * np.dot(k, t) ) for t in trans ])
                        else:
                            phase = 1.
                        
                        element = 0.5 * interactions[id] * phase * np.sqrt(self.norms[config_int]/self.norms[repr_b])
                        
                        # there is such a state in the list already
                        if flipped_states_elements.keys().__contains__(basis_index):
                            flipped_states_elements[basis_index] += element
                        else:
                            flipped_states_elements[basis_index] = element

                # Sz-Sz interaction energies
                self_energy -= self.delta * 0.5 * interactions[id] * np.sum(np.mod(config + roll_config, 2))

                if self.lamb != 1:
                    self_energy -= interactions[id] * self.delta * (self.lamb - 1) * np.sum(rot * np.roll(rot, dir, axis=(0, 1)))

            for basis_state, element in flipped_states_elements.items():
                ham_i.append(state)
                ham_j.append(basis_state)
                hamiltonian_elements.append(element)
            
            ham_i.append(state)
            ham_j.append(state)
            hamiltonian_elements.append(self_energy / 2.)
        
        repr_size = len(self.representatives)

        if easy_ks.__contains__(k[0]) and easy_ks.__contains__(k[1]):
            d_type = np.float32
        else:
            d_type = np.complex64

        self.hamiltonian = csr_matrix((np.array(hamiltonian_elements, dtype=d_type), (np.array(ham_i, dtype=int), np.array(ham_j, dtype=int))), shape=(repr_size, repr_size))


    def precompute_flips_change(self):
        """
        Precomputed change in the int index of the state with the spin flip term in Heisenberg Ham.
        """
        weight_rolled = np.roll(self.weight_matrix, (-1, 0), axis=(0, 1))
        self.flips_change_up = np.array(self.weight_matrix - weight_rolled, dtype=int)

        weight_rolled = np.roll(self.weight_matrix, (0, -1), axis=(0, 1))
        self.flips_change_side = np.array(self.weight_matrix - weight_rolled, dtype=int)


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

        # Make correct hamiltonian (now it's only up-triangle with diagonal reduced by a factor of 2)
        H = (self.hamiltonian + np.conj(self.hamiltonian.T))
        
        energies, states = eigsh(H, k=n_lowest, which='SA', tol=self.tol)

        self.gs_energy = float(energies[0])
        self.gs_in_basis = states[:, 0]

        if n_lowest > 1:
            self.eign_en = energies
            self.eignstates = states
            return energies, states


    def get_representations(self, n_workers=None):
        """
        Generates representations of spin states with parallel execution.
        
        Args:
            n_workers: Number of parallel workers. None = use all CPU cores.
        """
        if n_workers is None:
            n_workers = mp.cpu_count()
                
        xs = np.arange(self.n_x)
        ys = np.arange(self.n_y)
        all_translations = [[ [-int(y), -int(x)] for x in xs] for y in ys]
        translations = []
        
        for trans in all_translations:
            for t in trans:
                if self.magnon_number != int(self.n / 2):
                    # allow only even translations, as rotation map is invariant to ONLY EVEN T
                    if np.sum(t) % 2 == 0:
                        translations.append(t)
                else:
                    # for S^z_tot = 0 also T_odd * I, where I -- spin inversion, allowed
                    translations.append(t)
                
        # Pre-compute weight matrices and store as instance variable
        # last value is whether the translation needs spin inversion
        self.weight_matrices = [
            ((np.roll(self.weight_matrix, np.array(t), axis=(0, 1))).ravel(), t, np.sum(t)%2)
            for t in translations
        ]
        
        # Pre-compute bitmasks for all combinations
        all_states = list(self._generate_bitmasks())
        
        # Split work into chunks for parallel processing
        chunk_size = max(1, len(all_states) // (n_workers * 4))
        
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
        h_size = 2**self.n - 1
        
        for state_int in states:
            # Convert to config using the proper mapping
            state_cfg = self.map_int_to_config(state_int).ravel()
            
            set_states = set()
            all_states = []
            is_representative = True
            
            for w, trans, inv in self.weight_matrices:
                rolled_state_int = int(h_size * inv + (1 - 2 * inv) * np.dot(w, state_cfg))
                
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


    def get_ground_state(self):
        return self.gs_energy, self.gs_in_basis
    
    def map_int_to_config(self, n):
        return np.array([(n >> i) & 1 for i in reversed(range(self.n))], dtype=np.uint8).reshape((self.n_y, self.n_x))
    
    def map_int_to_basis(self, n):
        return self.representatives[n]
    
    def map_config_to_int(self, config : np.ndarray):
        return int(np.dot(self.weight_matrix.ravel(), config.ravel()))
    
    def map_config_to_basis(self, config):
        idx = self.map_config_to_int(config)
        return self.representatives[idx]

    def draw_configuration_int(self, n):
        a = self.map_int_to_config(n)#.reshape((self.n_y, self.n_x))
        print(a)



if __name__ == "__main__":
    nx = 6
    ny = 4
    n = nx*ny
    
    sc = SpinConfiguration(nx, ny, int(n/2), lowest_eignstates=1, delta=1., lamb=1., print_data=True)
    min_e, gs_state = sc.get_ground_state()
    
    print(f'Ground state energy E_0/J = {round(min_e, 7)}')
    print(f'GS energy per-site: e_0/J = {round(min_e/n, 7)}')
    eign = sc.get_eigva_eigve(5)[0]
    print(np.round(eign, 6))

    
    m_s = 0

    rotation_map = sc.rotation_map
    
    for repr, ii in sc.representatives.items():
        config = sc.map_int_to_config(repr)
        rot = config * rotation_map + (1 - config) * (1 - rotation_map)

        m_s += np.square(np.abs(gs_state[ii])) * (np.mean(rot))

    print(m_s)
    
    """
    for k in k_points:
        print(f'k=({'𝛑/' + str(int(1/k[0])) if k[0] % 1 != 0 else '0' if k[0] == 0 else str(int(k[0])) + '𝛑'}, {'𝛑/' + str(int(1/k[1])) if k[1] % 1 != 0 else '0' if k[1] == 0 else str(int(k[1])) + '𝛑'})')
        sc.generate_hamiltonian(k=k)
        sc.get_eigva_eigve()
        print(np.round(sc.eign_en[:1], 4))
    """    