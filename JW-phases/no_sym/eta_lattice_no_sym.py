import numpy as np
from itertools import combinations
from scipy.special import binom
from time import time
import matplotlib.pyplot as plt

from scipy.sparse.linalg import eigsh, norm
from scipy.sparse import csr_matrix
from numpy import array as cparray

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
        
        self.tol = tol
        
        self.weight_matrix = np.array([ 2**(ii-1) for ii in range(self.n, 0, -1)]).reshape((self.n_y, self.n_x))
        
        self.precompute_flips_change()
        
        self.rotation_map = np.array( [ [ (1 + (-1)**(x + y))/2 for x in range(self.n_x) ] for y in range(self.n_y) ] ).reshape((self.n_y, self.n_x))

        self.eign_en = []
        self.eignstates = []
        
        if self.magnon_number > self.n:
            raise ValueError('Magnon number should be smaller than the system size')
        
        self.size = int(binom(self.n, self.magnon_number))
        self.generate_spin_configurations()

        start = time()
        
        self.generate_hamiltonian()
        
        if print_data:
            print(f'Getting H took: {time()-start} s')
        start = time()
        
        self.get_eigva_eigve()
        
        if print_data:
            print(f'Finding eigvals and eigvecs took: {time()-start} s')

    def generate_hamiltonian(self, lamb = None, delta = None):
        """
        Generates Hamiltonian of the spin system by calulating all coefficients given by Heisenberg hamiltonian.
        Saves the result as sparse matrix (scipy.sparse.csr_matrix) under the value self.hamiltonian
        """
        
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
        spin_confg_items = list(self.spin_confg.items())

        map_int_to_basis = self.map_int_to_basis
        map_int_to_config = self.map_int_to_config
        rotation_map = self.rotation_map
        
        flipped_states_elements = {}

        if self.eta != 1:
            # generate angles map
            eta_phases = np.loadtxt(fr'JW-phases/no_sym/eta_phases_{self.n_x}x{self.n_y}_Sz={int(self.n_x*self.n_y/2-self.magnon_number)}.txt')
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
                    basis_index = map_int_to_basis(config_int + int_shift)

                    if basis_index < state:
                                                
                        element = 0.5 * interactions[id]
                        
                        if self.eta != 1:
                            eta_angle = eta_phases[ip]
                            element *= np.exp(1j * (1. - self.eta) * eta_angle)
                            ip += 1

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
        
        repr_size = len(self.spin_confg)

        d_type = np.float32

        if self.eta != 1:
            d_type = np.complex64

        self.hamiltonian = csr_matrix((cparray(hamiltonian_elements, dtype=d_type), (cparray(ham_i, dtype=int), cparray(ham_j, dtype=int))), shape=(repr_size, repr_size))


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
        return self.spin_confg[n]
    
    def map_config_to_int(self, config : np.ndarray):
        return int(np.dot(self.weight_matrix.ravel(), config.ravel()))
    
    def map_config_to_basis(self, config):
        idx = self.map_config_to_int(config)
        return self.spin_confg[idx]



if __name__ == "__main__":
    nx = 4
    ny = 4
    n = nx*ny
    
    sc = SpinConfiguration(nx, ny, int(n/2), lowest_eignstates=5, delta=1., J2overJ1=1., lamb=1., eta=0., print_data=True)
    min_e, gs_state = sc.get_ground_state()
    
    print(f'Ground state energy E_0/J = {round(min_e, 7)}')
    print(f'GS energy per-site: e_0/J = {-2*round(min_e/n, 7)}')
    eign = sc.get_eigva_eigve(5)[0]
    print(np.round(eign, 6))
    
    """
    for k in k_points:
        print(f'k=({'𝛑/' + str(int(1/k[0])) if k[0] % 1 != 0 else '0' if k[0] == 0 else str(int(k[0])) + '𝛑'}, {'𝛑/' + str(int(1/k[1])) if k[1] % 1 != 0 else '0' if k[1] == 0 else str(int(k[1])) + '𝛑'})')
        sc.generate_hamiltonian(k=k)
        sc.get_eigva_eigve()
        print(np.round(sc.eign_en[:1], 4))
    """    