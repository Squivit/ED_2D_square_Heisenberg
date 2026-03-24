import numpy as np
from itertools import combinations
from scipy.special import binom
from time import time
import matplotlib.pyplot as plt

from scipy.sparse.linalg import eigsh, norm
from scipy.sparse import csr_matrix, lil_matrix
from numpy import array as cparray

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from tqdm import tqdm

class SpinConfiguration:
    
    def __init__(self, N_x = 4, N_y = 4, magnon_number = 8, delta = 1., lamb= 1., J2overJ1 = 1., lowest_eignstates = 1, print_data = False, tol : float = 1e-8, eta = 1.):
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
        self.eta = eta

        self.t_matrix = None
        self.n_in_basis = None
        
        self.tol = tol
        
        self.weight_matrix = np.array([ 2**(ii-1) for ii in range(self.n, 0, -1)]).reshape((self.n_y, self.n_x))
        
        self.precompute_flips_change()
        
        self.rotation_map = np.array( [ [ (1 + (-1)**(x + y))/2 for x in range(self.n_x) ] for y in range(self.n_y) ] ).reshape((self.n_y, self.n_x))

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

        if self.eta != 1:
            # generate angles map
            try:
                eta_phases = np.loadtxt(fr'JW-phases/eta_phases_{self.n_x}x{self.n_y}_Sz={int(self.n_x*self.n_y/2-self.magnon_number)}.txt')
            except:
                eta_phases = np.loadtxt(fr'eta_phases_{self.n_x}x{self.n_y}_Sz={int(self.n_x*self.n_y/2-self.magnon_number)}.txt')
            #eta_phases = np.random.rand(2822634) * 2 * np.pi
            ip = 0

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
                        
                        if self.eta != 1:
                            eta_angle = eta_phases[ip]
                            phase *= np.exp(1j * (1. - self.eta) * np.mod(eta_angle, 2 * np.pi))
                            ip += 1
                        
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

        if easy_ks.__contains__(k[0]) and easy_ks.__contains__(k[1]) and self.eta == 1:
            self.hamiltonian = csr_matrix((cparray(hamiltonian_elements, dtype=np.float32), (cparray(ham_i, dtype=int), cparray(ham_j, dtype=int))), shape=(repr_size, repr_size))
        else:
            self.hamiltonian = csr_matrix((cparray(hamiltonian_elements, dtype=np.complex64), (cparray(ham_i, dtype=int), cparray(ham_j, dtype=int))), shape=(repr_size, repr_size))

        # Make correct hamiltonian (now it's only up-triangle with diagonal reduced by a factor of 2)
        self.H = (self.hamiltonian + np.conj(self.hamiltonian.T))


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

        energies, states = eigsh(self.H, k=n_lowest, which='SA', tol=self.tol)

        self.gs_energy = float(np.min(energies))
        self.gs_in_basis = states[:, 0]

        if n_lowest > 1:
            self.eign_en = energies
            self.eignstates = states
            return energies, states


    def get_representations(self):
        """
        Generates representations of the spin states reduced by the translational, spin inversion and mirror symmetries.
        """
                
        xs = np.arange(self.n_x)
        ys = np.arange(self.n_y)
        temp_translations = [[ [-int(y), -int(x)] for x in xs] for y in ys]
        translations = []
        
        for trans in temp_translations:
            for t in trans:
                # allow only even translations, as rotation map is invariant to ONLY EVEN T
                if np.sum(t) % 2 == 0:
                    translations.append(t)
                
        weight_matrices = []
                
        for t in translations:
            weight_matrices.append(( np.roll(self.weight_matrix, np.array(t), axis=(0, 1)).ravel() , t))
        
        int_to_cfg = self.map_int_to_config
                
        def get_periodicities(state_int):
            state_cfg = int_to_cfg(state_int).ravel()
            set_states = set()
            all_states = []
            
            for w, trans in weight_matrices:
                rolled_state_int = np.dot(w, state_cfg)
                if rolled_state_int < state_int:
                    # there is already a smaller representative -> translational state, whose int is smaller
                    return None, None
                else:
                    set_states.add(rolled_state_int)
                    all_states.append((rolled_state_int, trans))
                        
            return len(set_states), all_states
        
        # list of state_int
        self.representatives = []
        # keys: repr_state_int, values: normalization factor of the representative
        self.norms = {}
        # keys: state_int, values: (its representative int , translation needed to get to it)
        self.get_representative = {}
                
        def bitmask(combo):
            val = 0
            for i in combo:
                val |= 1 << (self.n - 1 - i)
            return val

        for bosons_on_sites in tqdm(combinations(range(self.n), self.magnon_number), total=int(self.size), disable = not self.print):
            state = bitmask(bosons_on_sites)
            norm, all_states = get_periodicities(state)
            if norm is not None:
                self.representatives.append(state)
                self.norms[state] = norm
                for _state, trans in all_states:
                    if self.get_representative.keys().__contains__(_state):
                        self.get_representative[_state][1].append(trans)
                    else:
                        self.get_representative[_state] = [state, [trans]]
        
        self.representatives.sort()
        # this order reduces the number of off-diagonal elements by a bit
        self.representatives.reverse()
        
        # dictionary of:
        # keys: int value of the representative
        # values: enumeration of the new basis of representatives  
        self.representatives = { r: i for i, r in enumerate(self.representatives) }
        
        if self.print:
            print(f'Total number of found representatives: {len(self.representatives.keys())}')
            print(f'Percent of repr in total states: {round(100*len(self.representatives.keys())/self.size, 2)}%')
            print(f'Estimated matrix reduction: {round(self.size/len(self.representatives.keys()), 1)}:1')


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
    
    def map_int_to_config(self, n):
        return np.array([(n >> i) & 1 for i in reversed(range(self.n))], dtype=np.uint8).reshape((self.n_y, self.n_x))
    
    def map_int_to_basis(self, n):
        return self.representatives[n]
    
    def map_config_to_int(self, config : np.ndarray):
        return int(np.dot(self.weight_matrix.ravel(), config.ravel()))
    
    def map_config_to_basis(self, config):
        idx = self.map_config_to_int(config)
        return self.representatives[idx]

    def get_t(self):

        if self.t_matrix is None:
            self.construct_t_matrix()

        return -np.dot(np.conj(self.gs_in_basis), self.t_matrix @ self.gs_in_basis) / (2 * self.n)
    
    def construct_t_matrix(self):
        chunk_size = len(self.representatives) // (mp.cpu_count()) + 1
        self.t_matrix = lil_matrix((len(self.representatives), len(self.representatives)), dtype=np.complex64)

        #self._process_t_batch(list(self.representatives.items()))

        #self.t_matrix = csr_matrix(self.t_matrix)

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
                self.t_matrix += future.result()

        self.t_matrix = csr_matrix(self.t_matrix)

    def _process_t_batch(self, representatives):

        map_int_to_basis = self.map_int_to_basis
        map_int_to_config = self.map_int_to_config
        
        # Directions for Heisenberg exchange
        directions = [(0, -1), (-1, 0)]  # x and y directions
        flips = [self.flips_change_side, self.flips_change_up]  # x and y directions

        small_t_matrix = lil_matrix((len(self.representatives), len(self.representatives)), dtype=np.complex64)

        for repr, state in representatives:
            config = map_int_to_config(repr)            

            for id, dir in enumerate(directions):
                roll_config = np.roll(config, dir, axis=(0, 1))
                possible_mixing = np.mod(config + roll_config, 2)

                config_int_shift = np.array(possible_mixing * flips[id], dtype=float)
                # multiplying to account for sign
                config_int_shift *= (-1.)**config

                for int_shift in config_int_shift[config_int_shift != 0].astype(int):
                    repr_b, _ = self.get_representative[repr + int_shift]
                    basis_index = map_int_to_basis(repr_b)
                    
                    if basis_index < state:
                        
                        element = 0.5 * np.sqrt(self.norms[repr]/self.norms[repr_b])
                        
                        small_t_matrix[basis_index, state] += np.conj(element)
                        small_t_matrix[state, basis_index] += element

        return small_t_matrix


    def get_n(self):
        """Get mean rotated bosons in ground state."""
        gs_state = self.get_ground_state()[1]

        if self.n_in_basis is None:
            self.n_in_basis = np.zeros(len(self.representatives))

            rotation_map = self.rotation_map
            
            for repr, state in self.representatives.items():
                config = self.map_int_to_config(repr)
                rot = config * rotation_map + (1 - config) * (1 - rotation_map)

                self.n_in_basis[state] = np.mean(rot)
        
        return np.dot(self.n_in_basis, np.square(np.abs(gs_state)))


if __name__ == "__main__":
    nx = 4
    ny = 4
    n = nx*ny

    delta = 2.
    lamb = 1.
    eta = 0.
    
    sc = SpinConfiguration(nx, ny, int(n/2), lowest_eignstates=5, delta=delta, lamb=lamb, eta=eta, print_data=True)
    min_e, gs_state = sc.get_ground_state()
    
    print(f'Ground state energy E_0/J = {round(min_e, 7)}')
    print(f'GS energy per-site: e_0/J = {-2*round(min_e/n, 7)}')
    eign = sc.get_eigva_eigve(5)[0]
    print(np.round(eign, 6))

    print((sc.get_t()))
    print(sc.get_n())
    
    sc = SpinConfiguration(nx, ny, int(n/2) - 1, lowest_eignstates=1, delta=delta, lamb=lamb, eta=eta, print_data=True)
    min_e_1, _ = sc.get_ground_state()
    
    print(f'Gap: {min_e_1 - min_e}')

