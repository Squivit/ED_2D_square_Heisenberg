from eta_lattice import SpinConfiguration
import numpy as np
import matplotlib.pyplot as plt
import os

from time import time

class SpinCorrelatorMachine:
    
    def __init__(self, _system_size = (2, 2), mag = -1, _delta : float = 1., _lamb : float = 1., _J2overJ1 = 1., ks = [[0, 0], [0, .5]], eta = 1.):
        
        self.delta = _delta
        self.lamb = _lamb
        self.J2 = _J2overJ1
        self.eta = eta
        
        self.nx = _system_size[0]
        self.ny = _system_size[1]
        self.n = self.nx*self.ny
        
        # system is 1D in y-axis
        if self.ny == 1:
            self.J2overJ1 = 0.
        
        if mag == -1:
            self.mag = int(self.n/2)
        else:
            self.mag = mag
                
        self.k_points = ks
        
        # for k-vector = (0, 0)
        self.sc = SpinConfiguration(self.nx, self.ny, magnon_number=self.mag, print_data=False, delta = self.delta, J2overJ1=self.J2, lamb=self.lamb, lowest_eignstates=1, eta = eta)
        self.sc_minus_helper = None
    
    
    def get_SDSF_zz(self, get_n_lowest = 35, print_en_diff = False):

        # magnitudes
        points_mag = np.zeros( (get_n_lowest, len(self.k_points)) )
        # energy differences from GS
        points_en = np.zeros( (get_n_lowest, len(self.k_points)) )
        
        self.sc.generate_hamiltonian(lamb=self.lamb, delta=self.delta)
        self.sc.get_eigva_eigve()

        gs_en, gs_state = self.sc.get_ground_state()
        self.gs_en = (gs_en)
        self.gs_state = (gs_state)
                
        for ik, k in enumerate(self.k_points):
            print(f'Calculating SDSF for k = {k}')
            
            if not (k[0] == 0 and k[1] == 0):
                self.sc.generate_hamiltonian(k=k, lamb=self.lamb, delta=self.delta)
            
            energy, states = self.sc.get_eigva_eigve(n_states=get_n_lowest)

            energy = (energy)
            non_gs_states : np.ndarray = (states)

            dEnergy = energy - self.gs_en
            
            if print_en_diff:
                print(f'Highest energy difference: {round(dEnergy[-1], 5)}J')
                print(f'Lowest energy difference: {round(dEnergy[0], 5)}J')

            
            # get phase map
            x_exp = np.exp(1j * np.pi * k[1] * np.arange(self.nx))
            y_exp = np.exp(1j * np.pi * k[0] * np.arange(self.ny))
            phase_map = np.outer(x_exp, y_exp)

            norm = 1. / np.sqrt(float(self.n))
            
            S_z_k_mapped = np.zeros(len(self.sc.representatives), dtype=np.complex64)
            
            for repr, ii in self.sc.representatives.items():
                Szk_amp = 0
                
                _, translations = self.sc.get_representative[repr]
                
                repr_config = self.sc.map_int_to_config(repr) - .5
                
                for trans in translations:
                    translated_repr = np.roll( repr_config, trans, axis=(0, 1) )
                    Szk_amp += np.sum(phase_map * translated_repr)
                    
                Szk_amp /= len(translations)
                                
                S_z_k_mapped[ii] = Szk_amp * norm
            
            Szk_gs = S_z_k_mapped * self.gs_state

            for ii in range(non_gs_states.shape[1]):
                overlap : np.complex64 = np.dot( np.conj(non_gs_states[:, ii]), Szk_gs )
                ov_mag_sq = np.abs(overlap)**2
                
                points_mag[ii, ik] = ov_mag_sq
                points_en[ii, ik] = dEnergy[ii]
        
        return points_mag, points_en


    def get_SDSF_minus(self, get_n_lowest = 35, print_en_diff = False):

        if self.sc_minus_helper is None:
            print('Calculating Sz=-1 helper...')
            self.sc_minus_helper = SpinConfiguration(self.nx, self.ny, self.mag - 1, self.delta, self.lamb, self.J2, lowest_eignstates=get_n_lowest, eta = self.eta)
        else:
            self.sc_minus_helper.delta = self.delta
            self.sc_minus_helper.lamb = self.lamb
        
        # magnitudes
        points_mag = np.zeros( (get_n_lowest, len(self.k_points)) )
        # energy differences from GS
        points_en = np.zeros( (get_n_lowest, len(self.k_points)) )
        
        self.sc.generate_hamiltonian(lamb=self.lamb, delta=self.delta)
        self.sc.get_eigva_eigve()
        
        gs_en, gs_state = self.sc.get_ground_state()
        self.gs_en = gs_en
        self.gs_state = gs_state
        
        for ik, k in enumerate(self.k_points):
            print(f'Calculating SDSF for k = {k}')
            
            if not (k[0] == 0 and k[1] == 0):
                self.sc_minus_helper.generate_hamiltonian(k=k, lamb=self.lamb, delta=self.delta)
            
            energy, non_gs_states = self.sc_minus_helper.get_eigva_eigve(n_states=get_n_lowest)

            dEnergy = energy - self.gs_en
            
            if print_en_diff:
                print(f'Highest energy difference: {round(dEnergy[-1], 5)}J')
                print(f'Lowest energy difference: {round(dEnergy[0], 5)}J')

            # get phase maps
            x_exp = np.exp(1j * np.pi * k[1] * np.arange(self.nx))
            y_exp = np.exp(1j * np.pi * k[0] * np.arange(self.ny))
            phase_map = np.outer(y_exp, x_exp)

            norm = 1. / np.sqrt(float(self.n))
                        
            S_minus_k_mapped_gs = np.zeros(len(self.sc_minus_helper.representatives), dtype=np.complex64)

            for repr, ii in self.sc.representatives.items():
                coeff = self.gs_state[ii]
                config = self.sc.map_int_to_config(repr)
                
                # finds where we can down-flip spin
                # It is this sum exp(ikj) S^-_j on GS
                for y, x in zip(*np.where(config == 1)):
                    flipped = config.copy()
                    flipped[y, x] = 0
                    
                    state_flipped = self.sc_minus_helper.map_config_to_int(flipped)
                    repr_f, translations = self.sc_minus_helper.get_representative[state_flipped]
                    ind = self.sc_minus_helper.representatives[repr_f]

                    phase = phase_map[y, x]

                    # average over all translations
                    phase_r_2 = np.mean(phase_map[tuple(np.transpose(translations))])
                          
                    norm_k = np.sqrt( self.sc.norms[repr] / self.sc_minus_helper.norms[repr_f] )
                    S_minus_k_mapped_gs[ind] += coeff * phase * phase_r_2 * norm * norm_k

            for ii in range(non_gs_states.shape[1]):
                overlap : np.complex64 = np.dot( np.conj(non_gs_states[:, ii]), S_minus_k_mapped_gs )
                ov_mag_sq = np.abs(overlap)**2
                
                points_mag[ii, ik] = ov_mag_sq
                points_en[ii, ik] = dEnergy[ii]
        
        return points_mag, points_en
    

def find_degeneracy(p_mag : np.ndarray, p_en : np.ndarray, limit = 4):

    en_rows = [ np.round(p_en[:, ii], limit) for ii in range(p_en.shape[1]) ]

    for ir, row in enumerate(en_rows):
        unique_vals, counts = np.unique_counts(row)

        degeneracy = np.where(counts > 1)[0]

        for deg_ind in degeneracy:
            real_ind = np.where(row == unique_vals[deg_ind])[0]

            for ind in real_ind[1:]:
                p_mag[real_ind[0], ir] += p_mag[ind, ir]
                p_mag[ind, ir] = 0

    return p_mag


def plot_SDSF(scm = None, nx = 4, ny = 4, lamb = 1, delta = 1, dir = '+-', to_diameter = False, graph_normalized = False, marker_s = 900, k_lowest = 10, save = False, eta = 1.):
    
    k_points = []
    
    k_x = [ 2*x/nx for x in range(int(nx/2)+1) ]
    for kx in k_x:
        k_points.append( np.array( [0, kx] ) )
        
    k_y = [ 2*y/ny for y in range(1, int(ny/2)+1) ]
    
    for ky in k_y:
        k_points.append( np.array( [ky, 1] ) )
        
    if nx == ny:
        for ky in (k_y[:-1]).__reversed__():
            k_points.append( np.array( [ky, ky] ) )
    print(k_points)
        
    if scm is None:
        scm = SpinCorrelatorMachine((nx, ny), ks=k_points, _lamb = lamb, _delta = delta, eta = eta)
    
    scm.lamb = lamb
    scm.delta = delta
    
    if dir == 'zz':
        p_s, p_x = scm.get_SDSF_zz(print_en_diff=False, get_n_lowest=k_lowest)
    else:
        p_s, p_x = scm.get_SDSF_minus(print_en_diff=False, get_n_lowest=k_lowest)
    
    p_s = find_degeneracy(p_mag=p_s, p_en=p_x, limit = 4)

    p_s = np.round(p_s, 9)
    print('Rounding result to the 1e-9 to reduce numerical inaccuracies')
    
    if save:
        try:
            os.makedirs(fr'data/d={delta}_l={lamb}_e={eta}', exist_ok=False)
        except:
            pass
        np.savetxt(fr'data/d={delta}_l={lamb}_e={eta}/{nx}x{ny}_S_{dir}_mag.txt', p_s)
        np.savetxt(fr'data/d={delta}_l={lamb}_e={eta}/{nx}x{ny}_S_{dir}_en.txt', p_x)
    
    print(f'Max magnitude of the overlap: {np.max(p_s)}')
    
    if graph_normalized:
        sizes = (p_s/np.max(p_s)*marker_s)
    else:
        sizes = p_s*marker_s
    
    if to_diameter:
        sizes = sizes**2
    
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

    for d, (ky, kx) in zip(dist, k_points):
        if (kx, ky) == (0.0, 0.0):
            labels.append(r'$\Gamma$')
        elif (kx, ky) == (1.0, 0.0) or (kx, ky) == (0.0, 1.0):
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
                plt.scatter( dist[ik], p_x[ii, ik], sizes[ii, ik], c='gray', alpha = .5 )
    
    if eta == 0:
        x_axis = np.arange(0, 2+np.sqrt(2)+0.001, 0.001)
        kx = []
        ky = []
        for x in x_axis:
            if x <= 1:
                kx.append(np.pi * x)
                ky.append(0.)
            elif x <= 2.:
                ky.append(np.pi * (x-1))
                kx.append(np.pi)
            else:
                kx.append(np.pi * (np.sqrt(2)-(x-2))/np.sqrt(2))
                ky.append(np.pi * (np.sqrt(2)-(x-2))/np.sqrt(2))

        kx = np.array(kx)
        ky = np.array(ky)

        analytics = 2*np.sqrt(delta**2 + (np.sin(kx)**2 + np.sin(ky)**2)/4)
        plt.plot(x_axis, analytics, 'g-', label='JW fermions')
    
    plt.ylim([0, 6])
    plt.xticks(tick_positions, labels, rotation = 0)
    plt.title(r'$S^{{{up}}}(q, \omega)$'.format(up=dir)
            + rf', {nx}x{ny}, $\Delta$ = {delta}'
            + (rf', $\lambda$ = {lamb}' if (delta != 0) else '')
            + (rf', $\eta$ = {eta}' ))
    plt.ylabel(r'$\omega$/J')
    #plt.xlabel(r'q: $\Gamma$ → $X$ → $M$')
    #plt.show()
    
    return scm



if __name__ == "__main__":
    #lamb = 0.
    #delta = 1.
    to_diameter = False
    graph_normalized = True
    marker_size = 40
    
    if not to_diameter:
        marker_size = marker_size**2
    
    scm = None
    
    usefull_sizes = [ (4, 4), (6, 4) ]
    
    #for ik, rect in enumerate(usefull_sizes):
    #    nx, ny = rect
    #    for il, lamb in enumerate([0., 1.]):
    #        plt.subplot(2, 2, ik + 1 + 2 * il)
    #        plot_SDSF(scm, nx, ny, lamb=lamb, delta=delta, dir='+-', marker_s=marker_size, graph_normalized=graph_normalized, to_diameter=to_diameter, save=False)
    
    deltas = [0., 1., 2.]
    lambdas = [0., 1.]

    deltas = [1.]
    lambdas = [1., 0.]
    
    
    variables = [1., .8, .6, .4, .2, 0.]
    
    for iv, var in enumerate(variables):
        plt.subplot(2, 3, iv + 1)
        plot_SDSF(scm, 4, 4, lamb=1., delta=1., eta=var, dir='+-', save=True)
    
    plt.show()
    

    """
    start_time = time()

    for id, delta in enumerate(deltas):
        for il, lamb in enumerate(lambdas):
            if delta != 0. or lamb != 0.:
                plt.subplot(len(deltas), len(lambdas), il + 1 + len(lambdas) * id)
                plot_SDSF(scm, 4, 4, lamb=lamb, delta=delta, dir='+-', eta = 0., save=True)
    #plot_SDSF(scm, 4, 4, lamb=1., delta=1., dir='zz', marker_s=marker_size, graph_normalized=graph_normalized, to_diameter=to_diameter, save=False)
    print(f'Execution time: {round(time() - start_time, 3)}s')

    plt.show()
    """
