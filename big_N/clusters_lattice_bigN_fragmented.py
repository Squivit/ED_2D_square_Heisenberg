from numpy import roll, array, mod, dot, uint8, sum, ones
import numpy as np
from time import time
import os

from scipy.special import binom

from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, save_npz, load_npz

from Betts_cluster import Cluster

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from paths import *
from tqdm import tqdm


class Memory_Fragment:

    def __init__(self, n : int, representatives : np.ndarray, norms : np.ndarray, cluster : Cluster, rotation_map : np.ndarray, weight_matrices : list, flips,
                 k, delta : float, lamb : float, n_workers : int = None, _print : bool = False):
        
        self.n = n

        self.representatives = representatives
        self.norms = norms

        self.cluster = cluster
        self.rotation_map = rotation_map
        self.weight_matrices = weight_matrices
        self.weight_matrix = weight_matrices[0][0]
        self.flips = flips

        self.k = k
        self.delta = delta
        self.lamb = lamb

        self.n_workers = n_workers
        self.print = _print

        e0 = 2. * self.n * self.delta / 4.
        self.e0 = e0


    def generate_hamiltonian(self, k = None, n_workers = None, ind_start = 0, ind_end = None):

        #start = time()

        easy_ks = [0., 2.]
        
        if easy_ks.__contains__(k[0]) and easy_ks.__contains__(k[1]):
            self.d_type = np.float32
            self.is_easy_k = True
        else:
            self.d_type = np.complex64
            self.is_easy_k = False

        repr_size = len(self.representatives)
        self.hamiltonian = csr_matrix((array([], dtype=self.d_type), (array([], dtype=int), array([], dtype=int))), shape=(repr_size, repr_size))

        if n_workers is None:
            if self.n_workers is None:
                n_workers = mp.cpu_count()
            else:
                n_workers = self.n_workers

        if ind_end is not None:
            slice_len = ind_end - ind_start
        else:
            slice_len = repr_size - ind_start

        # Split work into chunks for parallel processing
        chunk_size = slice_len // (n_workers) + 1
        
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context('spawn')) as executor:
            # Submit chunks
            futures = []
            chunk = []

            start_ind = ind_start
            end_ind = start_ind + chunk_size

            for _ in range(n_workers - 1):
                futures.append(executor.submit(self._process_hamiltonian_batch, (start_ind, end_ind)))
                start_ind += chunk_size
                end_ind += chunk_size

            futures.append(executor.submit(self._process_hamiltonian_batch, (start_ind, ind_end)))
            chunk.clear()

            for future in as_completed(futures):
                batch_results = future.result()
                self._merge_ham_results(batch_results)
        
        #if self.print:
        #    print(f'Getting H slice took: {round(time()-start, 5)} s')

        return self.hamiltonian
        

    def _process_hamiltonian_batch(self, params):
        """Process a batch of states for hamiltonian calculation."""

        start_ind, end_ind = params

        results_i = []
        results_j = []
        results_element = []
        
        map_int_to_basis = self.map_int_to_basis
        map_int_to_config = self.map_int_to_config_extended
        rotation_map = self.rotation_map
        cluster_map = self.cluster.cluster_map

        from collections import defaultdict
        flipped_states_elements = defaultdict(self.d_type)
        rotation_map = self.rotation_map
        
        # Directions for Heisenberg exchange
        directions = [(-1, 0), (0, -1)]  # x and y directions
        flips = self.flips  # x and y directions
        
        k = self.k
        is_easy_k = self.is_easy_k
        
        for _state, repr in enumerate(self.representatives[start_ind : end_ind]):
            config = map_int_to_config(repr)
            self_energy = self.e0
            state = _state + start_ind

            repr_norm = self.norms[state]
            
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
                    repr_b, trans = self.roll_to_repr(repr + int_shift)
                    basis_index = map_int_to_basis(repr_b)
                    
                    if basis_index < state:
                        phase = 1.
                        if not is_easy_k:
                            phase = np.mean([np.exp( -1j * np.pi * np.dot(k, t) ) for t in trans])
                        
                        element = 0.5 * phase * np.sqrt(repr_norm/self.norms[basis_index])
                        
                        flipped_states_elements[basis_index] += element
                    
                # Sz-Sz interaction energies
                self_energy -= self.delta * 0.5 * sum(mod(cluster_map*(config + roll_config), 2))

                if self.lamb != 1:
                    self_energy -= self.delta * (self.lamb - 1) * sum(cluster_map*(rot * roll(rot, dir, axis=(0, 1))))
            
            results_i.append(state)
            results_j.append(state)
            results_element.append(self_energy)

            results_i.extend([state]*len(flipped_states_elements))
            results_j.extend(flipped_states_elements.keys())
            results_element.extend(flipped_states_elements.values())

            results_i.extend(flipped_states_elements.keys())
            results_j.extend([state]*len(flipped_states_elements))
            results_element.extend(np.conj(np.array(list(flipped_states_elements.values()))))

        return (np.array(results_i, dtype=np.int32),
                np.array(results_j, dtype=np.int32),
                np.array(results_element, dtype=self.d_type))
    

    def _merge_ham_results(self, batch_results):
        """Merge results from a batch into main storage."""
        results_i, results_j, results_element = batch_results

        repr_size = len(self.representatives)
        self.hamiltonian += csr_matrix((array(results_element, dtype=self.d_type), (array(results_i, dtype=int), array(results_j, dtype=int))), shape=(repr_size, repr_size))


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
        return np.searchsorted(self.representatives, n)
    
    def map_config_to_int(self, config):
        """Map configuration to integer."""
        return int(dot(self.weight_matrix.ravel(), config.ravel()))
    
    def map_config_to_basis(self, config):
        """Map configuration to basis index."""
        idx = self.map_config_to_int(config)
        return self.map_int_to_basis(idx)



class SpinConfiguration:
    
    def __init__(self, N = 16, number = 'A', key = None, magnon_number = -1, delta = 1., lamb= 1., k = np.array([0., 0.]), n_workers = None,
                 lowest_eignstates = 1, print_data = False, force_ham_gen = False, save_ham = True, eigva_ve_only = False, helper_mode = False):
        """
        k is in units of pi
        """
        
        self.print = print_data
        self.n_lowest = lowest_eignstates
        self.n_workers = n_workers

        if key == None:
            self.n = N
            key = str(N)+number
        else:
            self.n = int(key[:2])
            number = key[-1]

        self.key = key

        if magnon_number == -1:
            self.magnon_number = int(self.n/2)
        else:
            self.magnon_number = magnon_number

        self.delta = delta
        self.k = k
        self.lamb = lamb
        
        # side lenghts optimized for 18A, 24A and 32A
        side = 18
        if self.n > 24:
            side = 26

        self.cluster = Cluster(N=self.n, num=number, key=key, bigmap_shift=0, bigmap_side=side)
                                
        self.weight_matrix = self.cluster.bosons_to_cluster_rolled(array([ 2**(ii-1) for ii in range(self.n, 0, -1)]))
        
        self.precompute_flips_change()
        
        self.rotation_map = ones(self.cluster.cluster_map.shape)
        self.rotation_map = self.rotation_map * array( [ [ (1+(-1)**(x+y))/2 for x in range(self.rotation_map.shape[0]) ] for y in range(self.rotation_map.shape[1]) ] )
        
        coords = self.cluster.sorted_coords
        cluster_map = self.cluster.cluster_map
        
        # Pre-compute translations (even only)
        translations = [trans for trans in coords if np.sum(trans) % 2 == 0]
        
        # Pre-compute weight matrices and store as instance variable
        self.weight_matrices = [
            ((cluster_map * np.roll(self.weight_matrix, np.array(t), axis=(0, 1))).ravel(), t)
            for t in translations
        ]

        self.eign_en = []
        self.eignstates = []
        
        if self.magnon_number > self.n:
            raise ValueError('Magnon number should be smaller than the system size')
        
        self.size = int(binom(self.n, self.magnon_number))

        if not eigva_ve_only:
            self.get_representations()

        if not helper_mode:

            self.save_ham = save_ham
            self.get_hamiltonian(force_ham_gen=force_ham_gen)
            
            start = time()

            self.get_eigva_eigve()
            
            if print_data:
                print(f'Finding eigvals and eigvecs took: {round(time()-start, 5)} s')
    

    def get_hamiltonian(self, k = None, lamb = None, delta = None, force_ham_gen = False):
        
        if k is None:
            k = self.k

        if lamb is not None:
            self.lamb = lamb

        if delta is not None:
            self.delta = delta
    
        self.k = k
        
        # forcing generation of the new hamiltonian
        if force_ham_gen:
            self.generate_hamiltonian(k)
            return
        
        if self.print:
            print('Trying to read hamiltonian')
        
        hamilonians = [f for f in os.listdir(HAMILTONIAN_PATH) if os.path.isfile(os.path.join(HAMILTONIAN_PATH, f))]

        # there is a correct hamiltonian saved
        if hamilonians.__contains__(f'{self.key}_k={k}_d={self.delta}_l={self.lamb}_Sz={abs(int(self.n/2) - self.magnon_number)}.npz'):
            self.hamiltonian = load_npz(fr'{HAMILTONIAN_PATH}/{self.key}_k={k}_d={self.delta}_l={self.lamb}_Sz={abs(int(self.n/2) - self.magnon_number)}.npz')
            if self.print:
                print('Hamiltonian loaded successfully.')

        # no hamiltonian, create and save one
        else:
            self.generate_hamiltonian(k)
    
    
    def get_memory_fragment(self, start_index, end_index, k, n_workers):
        a = self.mf.generate_hamiltonian(k, n_workers, start_index, end_index)
        self.hamiltonian += a


    def generate_hamiltonian(self, k = None, n_workers = None):
        """
        Generates Hamiltonian of the spin system by calulating all coefficients given by Heisenberg hamiltonian.
        Saves the result as sparse matrix (scipy.sparse.csr_matrix) under the value self.hamiltonian
        """

        if self.print:
            print('No hamiltonian to read or forced to start generation.')

        start = time()

        if k is None:
            k = self.k

        easy_ks = [0., 2.]
        
        if easy_ks.__contains__(k[0]) and easy_ks.__contains__(k[1]):
            self.d_type = np.float32
        else:
            self.d_type = np.complex64

        repr_size = len(self.representatives)
        self.hamiltonian = csr_matrix((array([], dtype=self.d_type), (array([], dtype=int), array([], dtype=int))), shape=(repr_size, repr_size))

        n_fragments = 50

        # Split work into chunks for parallel processing
        fragment_size = len(self.representatives) // (n_fragments) + 1

        self.mf = Memory_Fragment(self.n, self.representatives, self.norms, self.cluster, self.rotation_map, self.weight_matrices, [self.flips_change_side, self.flips_change_up],
                                k, self.delta, self.lamb, n_workers, self.print)

        start_ind = -fragment_size
        end_ind = 0
        
        for _ in tqdm(range(n_fragments), disable = not self.print, smoothing=0.):
            start_ind += fragment_size
            end_ind += fragment_size
            
            if end_ind > repr_size:
                end_ind = None

            self.get_memory_fragment(start_ind, end_ind, k, n_workers)
        
        if self.print:
            print(f'Getting full H took: {round(time()-start, 5)} s')
        
        if self.save_ham:
            self.save_hamiltonian()

    
    
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

        #self.hamiltonian = (self.hamiltonian + self.hamiltonian.T.conj())/2
        
        #energies, states = eigsh(self.hamiltonian, k=n_lowest, which='SA', ncv = max(2*n_lowest + 1, 60), tol=1e-6)
        energies, states = eigsh(self.hamiltonian, k=n_lowest, which='SA', ncv = n_lowest + 5, tol=1e-3)
        self.save_eign_eigva(energies, states)

        if n_lowest > 1:
            return energies, states
            
    def get_representations(self, n_workers=None):
        """
        Generates representations of spin states with parallel execution.
        
        Args:
            n_workers: Number of parallel workers. None = use all CPU cores.
        """

        # checks if successfully loaded representatives
        if self.load_representatives():
           return

    
    def save_hamiltonian(self):
        try:
            os.makedirs(f'{HAMILTONIAN_PATH}', exist_ok=False)
        except:
            pass

        fname = fr'{HAMILTONIAN_PATH}/{self.key}_k={self.k}_d={self.delta}_l={self.lamb}_Sz={abs(int(self.n/2) - self.magnon_number)}.npz'

        save_npz(fname, self.hamiltonian)


    def save_eign_eigva(self, en, vec):
        try:
            os.makedirs(f'{EIGN_PATH}', exist_ok=False)
        except:
            pass


        try:
            fname = fr'{EIGN_PATH}/{self.key}_k={self.k}_d={self.delta}_l={self.lamb}_Sz={abs(int(self.n/2) - self.magnon_number)}_en.txt'
            np.savetxt(fname, en)

            fname = fr'{EIGN_PATH}/{self.key}_k={self.k}_d={self.delta}_l={self.lamb}_Sz={abs(int(self.n/2) - self.magnon_number)}_vec.txt'
            np.savetxt(fname, vec)
        except:
            pass
    
    def load_representatives(self):
        try:
            npy_repr_fname = fr'{REPR_PATH}/{self.key}_Sz={abs(int(self.n/2) - self.magnon_number)}.npy'

            try:
                self.representatives = np.load(npy_repr_fname, mmap_mode='r')
            except:
                representatives = np.loadtxt(fr'{REPR_PATH}/{self.key}_Sz={abs(int(self.n/2) - self.magnon_number)}.txt')
                representatives = np.array(representatives, dtype = int)
                np.save(npy_repr_fname, representatives)
                representatives = None

                self.representatives = np.load(npy_repr_fname, mmap_mode='r')

            npy_norms_name = fr'{REPR_PATH}/{self.key}_norms_Sz={abs(int(self.n/2) - self.magnon_number)}.npy'

            try:
                self.norms = np.load(npy_norms_name, mmap_mode='r')
            except:
                norms : np.ndarray = np.loadtxt(fr'{REPR_PATH}/{self.key}_norms_Sz={abs(int(self.n/2) - self.magnon_number)}.txt')
                norms = np.array(norms, dtype = np.int16)
                np.save(npy_norms_name, norms)
                norms = None

                self.norms = np.load(npy_norms_name, mmap_mode='r')

            if self.print:
                print('Representatives loaded successfully')

            return True
        except:
            return False
    
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
        return np.searchsorted(self.representatives, n)
    
    def map_config_to_int(self, config):
        """Map configuration to integer."""
        return int(dot(self.weight_matrix.ravel(), config.ravel()))
    
    def map_config_to_basis(self, config):
        """Map configuration to basis index."""
        idx = self.map_config_to_int(config)
        return self.map_int_to_basis(idx)


if __name__ == "__main__":

    N = 28
    num = 'A'

    delta = 1.44
    lamb = 0.
    
    sc = SpinConfiguration(N=N, number=num, key=str(N)+num, magnon_number=int(N/2)+0, lowest_eignstates=2, n_workers=None,
                        delta=delta, lamb=lamb, k=np.array([0., 0.]), print_data=True, force_ham_gen=False, save_ham=True, eigva_ve_only=False)
    min_e = sc.get_eigva_eigve(2)[0][0]

    sc = SpinConfiguration(N=N, number=num, key=str(N)+num, magnon_number=int(N/2)+1, lowest_eignstates=2, n_workers=None,
                        delta=delta, lamb=lamb, k=np.array([1., 0.]), print_data=True, force_ham_gen=False, save_ham=True, eigva_ve_only=False)
    min_e_1 = sc.get_eigva_eigve(2)[0][0]

    print(f'Gap: {round(min_e_1 - min_e, 5)}')

    
    #eign = sc.get_eigva_eigve(5)[0]
    #print(np.round(eign, 6))
