import numpy as np
import matplotlib.pyplot as plt
import time

from Betts_cluster import *
import os

from multiprocessing import cpu_count
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
from paths import *
from clusters_lattice_bigN_fragmented import SpinConfiguration

class SpinCorrelatorMachine:
    
    def __init__(self, cluster_key = '16B', mag = -1, _delta : float = 1., _lamb : float = 1., _J2overJ1 = 1., ks = [[0, 0], [0, .5]], print_data = False):
        
        self.delta = _delta
        self.lamb = _lamb
        self.J2 = _J2overJ1
        
        self.key = cluster_key
        self.n = int(cluster_key[:2])
        
        self.mag = mag
        
        if mag == -1:
            self.mag = int(self.n/2)
        
        self.k_points = ks
        
        # for k-vector = (0, 0)
        self.sc = None
        self.sc_minus_helper = None
        
        self.print_data = print_data
    
    def load_en_vecs(self, k = np.array([0., 0.]), Sz = 1, gs = True):
        fname_npy = fr'{EIGN_PATH}/npy/{self.key}_k={k}_d={self.delta}_l={self.lamb}_Sz={Sz}'
        fname = fr'{EIGN_PATH}/{self.key}_k={k}_d={self.delta}_l={self.lamb}_Sz={Sz}'

        try:
            if gs:
                self.gs_en = np.load(f'{fname_npy}_en.npy', mmap_mode='r')[0]
                self.gs_state = np.load(f'{fname_npy}_vec.npy', mmap_mode='r')[:, 0]
            else:
                #print(f'{fname_npy}_en.npy')
                self.energy = np.load(f'{fname_npy}_en.npy', mmap_mode='r')
                self.states = np.load(f'{fname_npy}_vec.npy', mmap_mode='r')
        except:
            energy = np.loadtxt(f'{fname}_en.txt')
            states = np.loadtxt(f'{fname}_vec.txt', dtype=np.complex64)

            np.save(f'{fname_npy}_en.npy', energy)
            np.save(f'{fname_npy}_vec.npy', states)
            energy = None
            states = None

            if gs:
                self.gs_en = np.load(f'{fname_npy}_en.npy', mmap_mode='r')[0]
                self.gs_state = np.load(f'{fname_npy}_vec.npy', mmap_mode='r')[:, 0]
            else:
                self.energy = np.load(f'{fname_npy}_en.npy', mmap_mode='r')
                self.states = np.load(f'{fname_npy}_vec.npy', mmap_mode='r')
        
        
    def get_SDSF_zz(self, get_n_lowest = 150, print_en_diff = False):
        
        # magnitudes
        points_mag = np.zeros( (get_n_lowest, len(self.k_points)) )
        # energy differences from GS
        points_en = np.zeros( (get_n_lowest, len(self.k_points)) )

        self.load_en_vecs(Sz=0, gs = True)

        self.sc = SpinConfiguration(key=key, helper_mode=True)
        
        for ik, k in enumerate(self.k_points):
            if self.print_data:
                print(f'Calculating SDSF for k = {k}')
            
            self.load_en_vecs(k = k, Sz=0, gs = False)

            dEnergy = self.energy - self.gs_en
            
            if print_en_diff and self.print_data:
                print(f'Highest energy difference: {round(dEnergy[-1], 5)}J')
                print(f'Lowest energy difference: {round(dEnergy[0], 5)}J')
            
            # get S^z_k mapped GS in k = 00
            x_exp = np.exp(1j * np.pi * k[0] * np.arange(self.sc.cluster.bigmap_side))
            y_exp = np.exp(1j * np.pi * k[1] * np.arange(self.sc.cluster.bigmap_side))
            self.phase_map = np.outer(x_exp, y_exp)
            
            self.S_z_k_mapped = np.zeros(len(self.sc.representatives), dtype=np.complex64)
            
            n_workers = mp.cpu_count()

            len_states = len(self.sc.representatives)

            # Split work into chunks for parallel processing
            fragment_size = len_states // (n_workers) + 1

            start_ind = -fragment_size
            end_ind = 0
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = []

                for _ in range(n_workers):
                    start_ind += fragment_size
                    end_ind += fragment_size
                    
                    if end_ind > len_states:
                        end_ind = None

                    futures.append(executor.submit(self._process_sz_batch, (start_ind, end_ind)))

                for future in tqdm(as_completed(futures), total=len(futures), disable = not self.print_data):
                    batch_results = future.result()
                    self._merge_sz_results(batch_results)
            
            Szk_gs = self.S_z_k_mapped * self.gs_state

            for ii in range(self.states.shape[1]):
                overlap : np.complex64 = np.dot( np.conj(self.states[:, ii]), Szk_gs )
                ov_mag_sq = np.abs(overlap)**2
                
                points_mag[ii, ik] = ov_mag_sq
                points_en[ii, ik] = dEnergy[ii]
        
        return points_mag, points_en
    
    def _process_sz_batch(self, params):
        """Process a batch of states for hamiltonian calculation."""
        results_basis = []
        results_coeff = []

        norm = 1. / np.sqrt(float(self.n))

        start_ind, end_ind = params
        
        for ii, repr in enumerate(self.sc.representatives[start_ind:end_ind]):
            Szk_amp = 0
            
            _, translations = self.sc.roll_to_repr(repr)
            
            repr_config = self.sc.map_int_to_config_extended(repr) - .5
            
            for trans in translations:
                translated_repr = np.roll( repr_config, trans, axis=(0, 1) ) * self.sc.cluster.cluster_map
                Szk_amp += np.sum(self.phase_map * translated_repr)
                
            Szk_amp /= len(translations)

            results_basis.append(ii)
            results_coeff.append(Szk_amp * norm)

        return (results_basis, results_coeff)
    
    def _merge_sz_results(self, batch_results):
        """Merge results from a batch into main storage."""
        res_basis, res_coeff = batch_results

        for ii, coeff in zip(res_basis, res_coeff):
            self.S_z_k_mapped[ii] = coeff


    def get_SDSF_minus(self, get_n_lowest = 5, print_en_diff = False):
        
        # magnitudes
        points_mag = np.zeros( (get_n_lowest, len(self.k_points)) )
        # energy differences from GS
        points_en = np.zeros( (get_n_lowest, len(self.k_points)) )
        
        if self.sc is None:
            self.sc = SpinConfiguration(key=self.key, helper_mode=True)
        
        if self.print_data:
            print('Loading ground state energy and vector')

        self.load_en_vecs(Sz=0, gs = True)
        
        for ik, k in enumerate(self.k_points):
            if self.print_data:
                print(f'Calculating k = {k}')
            
            if self.sc_minus_helper is None:
                if self.print_data:
                    print('Getting Sz=1 helper...')
                self.sc_minus_helper = SpinConfiguration(key=self.key, magnon_number=int(int(self.key[:2])/2)+1, helper_mode=True)

            if self.print_data:
                print('Loading eigenstates and eigenenergies')

            self.load_en_vecs(k = k, Sz=1, gs=False)

            dEnergy = self.energy - self.gs_en
            
            if print_en_diff and self.print_data:
                print(f'Highest energy difference: {round(dEnergy[-1], 5)}J')
                print(f'Lowest energy difference: {round(dEnergy[0], 5)}J')

            # get S^z_k maps
            x_exp = np.exp(1j * np.pi * k[0] * np.arange(self.sc.cluster.bigmap_side))
            y_exp = np.exp(1j * np.pi * k[1] * np.arange(self.sc.cluster.bigmap_side))
            self.phase_map = np.outer(x_exp, y_exp)
            
            self.S_minus_k_mapped_gs = np.zeros(len(self.sc_minus_helper.representatives), dtype=np.complex64)
            
            n_workers = mp.cpu_count()
            #n_workers = 1

            len_states = len(self.sc.representatives)

            # Split work into chunks for parallel processing
            fragment_size = len_states // (n_workers) + 1

            start_ind = -fragment_size
            end_ind = 0
            
            print('Starting calculations')
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = []

                for _ in tqdm(range(n_workers), disable = not self.print_data):
                    start_ind += fragment_size
                    end_ind += fragment_size
                    
                    if end_ind > len_states:
                        end_ind = None

                    futures.append(executor.submit(self._process_splus_batch, (start_ind, end_ind)))

                for future in tqdm(as_completed(futures), total=len(futures), disable = not self.print_data, smoothing=0.):
                    batch_results = future.result()
                    self._merge_splus_results(batch_results)
            

            #result = self._process_splus_batch((0, None))
            #self._merge_splus_results(result)
                        
            for ii in range(self.states.shape[1]):
                overlap : np.complex64 = np.dot( np.conj(self.states[:, ii]), self.S_minus_k_mapped_gs )
                ov_mag_sq = np.abs(overlap)**2
                
                points_mag[ii, ik] = ov_mag_sq
                points_en[ii, ik] = dEnergy[ii]
            
            print(points_en)
            print(points_mag)
        
        return points_mag, points_en

    def _process_splus_batch(self, params):
        """Process a batch of states."""

        results_coeff = np.zeros(len(self.sc_minus_helper.representatives), dtype=np.complex64)

        norm = 1. / np.sqrt(float(self.n))

        start_ind, end_ind = params
        
        for repr, coeff in zip(self.sc.representatives[start_ind:end_ind], self.gs_state[start_ind:end_ind]):
        #for repr, coeff in tqdm(zip(self.sc.representatives[start_ind:end_ind], self.gs_state[start_ind:end_ind]), total = len(self.sc.representatives[start_ind:end_ind])):
            config = self.sc.map_int_to_config(repr)
            state = self.sc.map_int_to_basis(repr)
            
            int_shifts = self.sc.weight_matrix * self.sc.cluster.cluster_map * (1 - config)
            
            for x, y in zip(*int_shifts.nonzero()):
                state_flipped = repr + int_shifts[x, y]
                a = config
                a[x, y] += 1
                repr_f, translations = self.sc_minus_helper.roll_to_repr(state_flipped)
                
                ind = self.sc_minus_helper.map_int_to_basis(repr_f)

                phase = self.phase_map[x, y]

                # average over all translations
                phase_r_2 = 0
                for trans in translations:
                    ty, tx = np.array(trans)
                    phase_r_2 += self.phase_map[ty, tx]
                
                phase_r_2 /= len(translations)
                        
                norm_k = np.sqrt( self.sc.norms[state] / self.sc_minus_helper.norms[ind] )

                results_coeff[ind] = coeff * phase * phase_r_2 * norm * norm_k
            
        return results_coeff
    
    def _merge_splus_results(self, batch_results):
        """Merge results from a batch into main storage."""
        
        self.S_minus_k_mapped_gs += batch_results


def get_SDSF(scm = None, key = '16B', lamb = 1, delta = 1, dir = '+-', plot = False, print_data = False, save = True):
    
    clust = Cluster(key=key)
    _, k_points = clust.get_k_space()
    
    if scm is None:
        scm = SpinCorrelatorMachine(cluster_key=key, ks=k_points, _lamb = lamb, _delta = delta, print_data=print_data)
    
    scm.print_data = print_data
    scm.lamb = lamb
    scm.delta = delta
    
    if dir == 'zz':
        p_s, p_x = scm.get_SDSF_zz(print_en_diff=False)
    else:
        p_s, p_x = scm.get_SDSF_minus(print_en_diff=True)
    
    p_s = np.round(p_s, 9)
    #print('Rounding result to the 1e-9 to reduce numerical inaccuracies')
    
    if save:
        try:
            os.makedirs(fr'data/d={delta}_l={lamb}', exist_ok=False)
        except:
            pass
        np.savetxt(fr'data/d={delta}_l={lamb}/{key}_S_{dir}_mag.txt', p_s)
        np.savetxt(fr'data/d={delta}_l={lamb}/{key}_S_{dir}_en.txt', p_x)
    
    #print(f'Max magnitude of the overlap: {np.max(p_s)}')
    
    if plot:
        sizes = p_s*20
        
        from fractions import Fraction
        
        # ---- Step 1: Build cumulative "distance" along the path ----
        dist = [0.0]
        for i in range(1, len(k_points)):
            dk = np.linalg.norm(np.array(k_points[i]) - np.array(k_points[i-1]))
            dist.append(dist[-1] + dk)
        dist = np.array(dist)

        # ---- Helper function to format fractions of π ----
        def format_pi_fraction(value):
            frac = Fraction(value).limit_denominator(12)  # limit denominator (adjust if needed)
            if frac == 0:
                return "0"
            elif frac == 1:
                return r"$\pi$"
            else:
                return fr"$\frac{{{frac.numerator}}}{{{frac.denominator}}}\pi$"

        # ---- Step 2: Define special labels for high symmetry points ----
        labels = []
        tick_positions = []

        for d, (kx, ky) in zip(dist, k_points):
            if (kx, ky) == (0.0, 0.0):
                labels.append(r'$\Gamma$')
            elif (kx, ky) == (1.0, 0.0):
                labels.append(r"$X$")
            elif (kx, ky) == (1.0, 1.0):
                labels.append(r"$M$")
            else:
                kx_label = format_pi_fraction(kx)
                ky_label = format_pi_fraction(ky)
                labels.append(rf"({kx_label}, {ky_label})")
            tick_positions.append(d)
        
        for ik, k in enumerate(k_points):
                    
            for ii in range(p_s.shape[0]):
                if sizes[ii, ik] > 0:
                    plt.scatter( dist[ik], p_x[ii, ik], sizes[ii, ik], c='gray' )
        
        if dir == '+-' and lamb == 0.:
            x_range = np.arange( dist[0], dist[-1], 0.001 )
            plt.plot( x_range, np.sin(np.pi/2 * x_range) + np.min(p_x), 'g--' )
        
        plt.ylim([0, 6])
        plt.xticks(tick_positions, labels, rotation = 0)
        plt.title(r'$S^{{{up}}}(q, \omega)$'.format(up=dir)
                + rf', {key}, $\Delta$ = {delta}'
                + (rf', $\lambda$ = {lamb}' if (delta != 0) else ''))
        plt.ylabel(r'$\omega$/J')
    
    return scm

def save_process_data_to_txt(filename : str,label : str, exec_time : float):
    try:
        os.makedirs(fr'datalogs', exist_ok=False)
    except:
        pass

    with open(rf'datalogs/{filename}', 'a+') as file:
        mp_text = f'cpu-cores: {cpu_count()}, '
        time_text = f'exec. time: {int(exec_time // 3600)}h {int((exec_time%3600)//60)}min {np.round(exec_time % 60, 3)}s'
        file.write(label + mp_text + time_text + '\n')


def get_lambda_var(key = '32A', filename = '', delta = 1.):
    
    variables = [1., 0.]
    scm = None
    
    for var in variables:
        start_time = time.time()
        scm = get_SDSF(scm, key, lamb=var, delta=delta, dir='+-', plot=False, print_data = True)
        work_time = time.time() - start_time
        save_process_data_to_txt(filename=filename, label = f'S+-, l: {var}, d: {delta}, key: {key}, ', exec_time=work_time)
    
    
    #for var in variables:
    #    start_time = time.time()
    #    scm = get_SDSF(scm, key, lamb=var, delta=delta, dir='zz', plot=False, print_data = True)
    #    work_time = time.time() - start_time
    #    save_process_data_to_txt(filename=filename, label = f'Szz, l: {var}, d: {delta}, key: {key}, ', exec_time=work_time)



def get_delta_var(key = '18A', filename = '', lamb = 1.):
    
    variables = [2.]
    scm = None

    for var in variables:
        start_time = time.time()
        scm = get_SDSF(scm, key, lamb=lamb, delta=var, dir='+-', plot=False, print_data = True)
        work_time = time.time() - start_time
        save_process_data_to_txt(filename=filename, label = f'S+-, l: {lamb}, d: {var}, key: {key}, ', exec_time=work_time)
    
    
    #for var in variables:
    #    start_time = time.time()
    #    scm = get_SDSF(scm, key, lamb=lamb, delta=var, dir='zz', plot=False, print_data = True)
    #    work_time = time.time() - start_time
    #    save_process_data_to_txt(filename=filename, label = f'Szz, l: {lamb}, d: {var}, key: {key}, ', exec_time=work_time)


if __name__ == "__main__":
    key = '32A'
    date_time = time.asctime(time.localtime()).replace(' ', '_').replace(':', '_')
    filename = rf"{key}__{date_time}.txt"

    get_lambda_var(key, filename, delta=1.)

    #get_delta_var(key, filename, lamb=1.)
