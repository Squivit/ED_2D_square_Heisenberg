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
from lattice_36_save import SpinConfiguration

class SpinCorrelatorMachine:
    
    def __init__(self, nx = 6, ny = 6, mag = -1, _delta : float = 1., ks = [[0, 0], [0, .5]], print_data = False):
        
        self.delta = _delta
        
        self.nx = nx
        self.ny = ny
        self.n = nx*ny
        
        self.mag = mag
        
        if mag == -1:
            self.mag = int(self.n/2)
        
        self.k_points = ks
        
        # for k-vector = (0, 0)
        self.sc = None
        self.sc_minus_helper = None
        
        self.print_data = print_data
    
        
    def get_SDSF_zz(self, get_n_lowest = 150, print_en_diff = False):
        
        # magnitudes
        points_mag = np.zeros( (get_n_lowest, len(self.k_points)) )
        # energy differences from GS
        points_en = np.zeros( (get_n_lowest, len(self.k_points)) )

        fname = fr'{EIGN_PATH}/{self.nx}x{self.ny}_k={np.array([0., 0.])}_d={self.delta}_Sz={0}'

        gs_en = np.loadtxt(f'{fname}_en.txt')[0]
        gs_state = np.loadtxt(f'{fname}_vec.txt')[:, 0]

        self.gs_en = gs_en
        self.gs_state = gs_state

        self.sc = SpinConfiguration(nx, ny, helper_mode=True)
        
        for ik, k in enumerate(self.k_points):
            if self.print_data:
                print(f'Calculating SDSF for k = {k}')
            
            if not (k[0] == 0 and k[1] == 0):
                fname = fr'{EIGN_PATH}/{self.nx}x{self.ny}_k={k}_d={self.delta}_Sz={0}'

            energy = np.loadtxt(f'{fname}_en.txt')
            states = np.loadtxt(f'{fname}_vec.txt')
            
            non_gs_states : np.ndarray = states

            dEnergy = energy - self.gs_en
            
            if print_en_diff and self.print_data:
                print(f'Highest energy difference: {round(dEnergy[-1], 5)}J')
                print(f'Lowest energy difference: {round(dEnergy[0], 5)}J')
            
            # get S^z_k mapped GS in k = 00
            x_exp = np.exp(1j * np.pi * k[0] * np.arange(self.nx))
            y_exp = np.exp(1j * np.pi * k[1] * np.arange(self.ny))
            self.phase_map = np.outer(x_exp, y_exp)
            
            self.S_z_k_mapped = np.zeros(len(self.sc.representatives), dtype=np.complex64)
            
            n_workers = mp.cpu_count()

            # Split work into chunks for parallel processing
            chunk_size = np.ceil(len(self.sc.representatives) / n_workers)
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Submit chunks
                futures = []
                chunk = []
                for repr, ii in self.sc.representatives.items():
                    chunk.append((repr, ii))
                    
                    if len(chunk) >= chunk_size:
                        futures.append(executor.submit(self._process_sz_batch, chunk.copy()))
                        chunk.clear()
                
                if len(chunk) > 0:
                    futures.append(executor.submit(self._process_sz_batch, chunk.copy()))
                    chunk.clear()

                for future in tqdm(as_completed(futures), total=len(futures), disable = not self.print_data):
                    batch_results = future.result()
                    self._merge_sz_results(batch_results)
            
            Szk_gs = self.S_z_k_mapped * self.gs_state

            for ii in range(non_gs_states.shape[1]):
                overlap : np.complex64 = np.dot( np.conj(non_gs_states[:, ii]), Szk_gs )
                ov_mag_sq = np.abs(overlap)**2
                
                points_mag[ii, ik] = ov_mag_sq
                points_en[ii, ik] = dEnergy[ii]
        
        return points_mag, points_en
    
    def _process_sz_batch(self, representatives):
        """Process a batch of states for hamiltonian calculation."""
        results = []
        norm = 1. / np.sqrt(float(self.n))
        
        for repr, ii in representatives:
            Szk_amp = 0
            
            _, translations = self.sc.roll_to_repr(repr)
            
            repr_config = self.sc.map_int_to_config(repr) - .5
            
            for trans in translations:
                translated_repr = np.roll( repr_config, trans, axis=(0, 1) )
                Szk_amp += np.sum(self.phase_map * translated_repr)
                
            Szk_amp /= len(translations)

            results.append({
                'basis': ii,
                'coeff': Szk_amp * norm,
            })
            
        return results
    
    def _merge_sz_results(self, batch_results):
        """Merge results from a batch into main storage."""
        for result in batch_results:
            ii = result['basis']
            coeff = result['coeff']
            
            self.S_z_k_mapped[ii] = coeff


    def get_SDSF_minus(self, get_n_lowest = 50, print_en_diff = False):
        
        # magnitudes
        points_mag = np.zeros( (get_n_lowest, len(self.k_points)) )
        # energy differences from GS
        points_en = np.zeros( (get_n_lowest, len(self.k_points)) )
        
        if self.sc is None:
            self.sc = SpinConfiguration(nx = self.nx, ny = self.ny, helper_mode=True)
        
        if self.print_data:
            print('Loading ground state energy and vector')

        fname = fr'{EIGN_PATH}/{self.nx}x{self.ny}_k={np.array([0., 0.])}_d={self.delta}_Sz={0}'

        gs_en = np.loadtxt(f'{fname}_en.txt')[0]
        gs_state = np.loadtxt(f'{fname}_vec.txt', dtype=np.complex64)[:, 0]

        self.gs_en = gs_en
        self.gs_state = gs_state
        
        for ik, k in enumerate(self.k_points):
            if self.print_data:
                print(f'Calculating k = {k}')
            
            if self.sc_minus_helper is None:
                if self.print_data:
                    print('Getting Sz=1 helper...')
                self.sc_minus_helper = SpinConfiguration(nx = self.nx, ny = self.ny, magnon_number=int(self.n/2)+1, helper_mode=True)

            if self.print_data:
                print('Loading eigenstates and eigenenergies')

            fname = fr'{EIGN_PATH}/{self.nx}x{self.ny}_k={k}_d={self.delta}_Sz={1}'

            energy = np.loadtxt(f'{fname}_en.txt')
            states = np.loadtxt(f'{fname}_vec.txt', dtype=np.complex64)

            non_gs_states : np.ndarray = states

            dEnergy = energy - self.gs_en
            
            if print_en_diff and self.print_data:
                print(f'Highest energy difference: {round(dEnergy[-1], 5)}J')
                print(f'Lowest energy difference: {round(dEnergy[0], 5)}J')

            # get S^z_k maps
            x_exp = np.exp(1j * np.pi * k[0] * np.arange(self.nx))
            y_exp = np.exp(1j * np.pi * k[1] * np.arange(self.ny))
            self.phase_map = np.outer(x_exp, y_exp)
            
            self.S_minus_k_mapped_gs = np.zeros(len(self.sc_minus_helper.representatives), dtype=np.complex64)
            
            n_workers = mp.cpu_count()
            # Split work into chunks for parallel processing
            chunk_size = len(self.sc.representatives) // n_workers + 1
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Submit chunks
                futures = []
                chunk = []
                for repr, ii in tqdm(self.sc.representatives.items(), total=len(self.sc.representatives), disable = not self.print_data):
                    chunk.append((repr, self.gs_state[ii]))
                    
                    if len(chunk) >= chunk_size:
                        futures.append(executor.submit(self._process_sminus_batch, chunk.copy()))
                        chunk.clear()
                
                if len(chunk) > 0:
                    futures.append(executor.submit(self._process_sminus_batch, chunk.copy()))
                    chunk.clear()

                for future in tqdm(as_completed(futures), total=len(futures), disable = not self.print_data):
                    batch_results = future.result()
                    self._merge_sminus_results(batch_results)
                        
            for ii in range(non_gs_states.shape[1]):
                overlap : np.complex64 = np.dot( np.conj(non_gs_states[:, ii]), self.S_minus_k_mapped_gs )
                ov_mag_sq = np.abs(overlap)**2
                
                points_mag[ii, ik] = ov_mag_sq
                points_en[ii, ik] = dEnergy[ii]
        
        return points_mag, points_en

    def _process_sminus_batch(self, representatives):
        """Process a batch of states."""
        results = []
        norm = 1. / np.sqrt(float(self.n))

        for repr, coeff in representatives:
            config = self.sc.map_int_to_config(repr)
            
            int_shifts = self.sc.weight_matrix * (1 - config)
            
            for x, y in zip(*np.where(int_shifts != 0)):
                state_flipped = repr + int_shifts[x, y]
                repr_f, translations = self.sc_minus_helper.roll_to_repr(state_flipped)
                
                ind = self.sc_minus_helper.representatives[repr_f]

                phase = self.phase_map[x, y]

                # average over all translations
                phase_r_2 = 0
                for trans in translations:
                    ty, tx = np.array(trans)
                    phase_r_2 += self.phase_map[ty, tx]
                
                phase_r_2 /= len(translations)
                        
                norm_k = np.sqrt( self.sc.norms[repr] / self.sc_minus_helper.norms[repr_f] )

                results.append({
                    'basis': ind,
                    'coeff': coeff * phase * phase_r_2 * norm * norm_k,
                })
            
        return results
    
    def _merge_sminus_results(self, batch_results):
        """Merge results from a batch into main storage."""
        for result in batch_results:
            ind = result['basis']
            coeff = result['coeff']
            
            self.S_minus_k_mapped_gs[ind] += coeff


def get_SDSF(scm = None, nx = 6, ny = 6, delta = 1, dir = '+-', plot = False, print_data = False, save = True):
    
    k_points = []

    kxs = [2 * x / nx for x in range(int(nx/2) + 1)]
    for kx in kxs:
        k_points.append( np.array( [0, kx] ) )

    kys = [2 * y / ny for y in range(int(ny/2) + 1)]
    for ky in kys:
        k_points.append( np.array( [ky, 1] ) )

    if nx == ny:
        for ky in (kys[:-1]).__reversed__():
            k_points.append( np.array( [ky, ky] ) )

    if scm is None:
        scm = SpinCorrelatorMachine(nx = nx, ny = ny, ks=k_points, _delta = delta, print_data=print_data)
    
    scm.print_data = print_data
    scm.delta = delta
    
    if dir == 'zz':
        p_s, p_x = scm.get_SDSF_zz(print_en_diff=False)
    else:
        p_s, p_x = scm.get_SDSF_minus(print_en_diff=True)
    
    p_s = np.round(p_s, 9)
    #print('Rounding result to the 1e-9 to reduce numerical inaccuracies')
    
    if save:
        try:
            os.makedirs(fr'data/d={delta}_l={1.}', exist_ok=False)
        except:
            pass
        np.savetxt(fr'data/d={delta}_l={1.}/{nx}x{ny}_S_{dir}_mag.txt', p_s)
        np.savetxt(fr'data/d={delta}_l={1.}/{nx}x{ny}_S_{dir}_en.txt', p_x)
            
    return scm

def save_process_data_to_txt(filename : str, label : str, exec_time : float):
    try:
        os.makedirs(fr'datalogs', exist_ok=False)
    except:
        pass

    with open(rf'datalogs/{filename}', 'a+') as file:
        mp_text = f'cpu-cores: {cpu_count()}, '
        time_text = f'exec. time: {int(exec_time // 3600)}h {int((exec_time%3600)//60)}min {np.round(exec_time % 60, 3)}s'
        file.write(label + mp_text + time_text + '\n')


def get_single_sdsf(nx = 6, ny = 6, filename = '', delta = 1.):
    start_time = time.time()
    scm = get_SDSF(scm, nx, ny, delta=delta, dir='+-', plot=False, print_data = True)
    work_time = time.time() - start_time
    save_process_data_to_txt(filename=filename, label = f'S+-, l: {1.}, d: {delta} ', exec_time=work_time)


if __name__ == "__main__":
    nx = 6
    ny = 6

    date_time = time.asctime(time.localtime()).replace(' ', '_').replace(':', '_')
    filename = rf"{nx}x{ny}__{date_time}.txt"

    get_single_sdsf(nx, ny, filename, delta=1.)