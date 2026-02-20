from Betts_cluster import Cluster
from lattice_36_save import SpinConfiguration
import numpy as np

if __name__ == "__main__":
    nx = 6
    ny = 6

    delta = 1.

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

    N = nx * ny
    #sc = SpinConfiguration(nx, ny, magnon_number=int(N/2), lowest_eignstates=10, delta = delta, k=np.array([0., 0.]),
    #                       n_workers=20, print_data=True, force_ham_gen=True, save_ham=True, eigva_ve_only=False)
    #min_e, gs_state = sc.get_ground_state()
    #    
    #print(f'Ground state energy E_0/J = {round(min_e, 5)}')
    #print(f'GS energy per-site: e_0/J = {round(min_e/(N), 5)}')

    start_index = 2

    sc = SpinConfiguration(nx, ny, magnon_number=int(N/2)+1, lowest_eignstates=1, delta = delta, k=k_points[start_index],
                           n_workers=17, print_data=True, force_ham_gen=True, save_ham=True, eigva_ve_only=False)

    for k in k_points[1:]:

        print()
        print(f'Calculating k = {k} block')

        sc.generate_hamiltonian(k = k)
            #sc.get_eigva_eigve(). 



        sc.hamiltonian = None
        sc.hamiltonian_elements = None
        sc.ham_i = None
        sc.ham_j = None
        
        #min_e, gs_state = sc.get_ground_state()
        
        #print(f'Ground state energy E_0/J = {round(min_e, 5)}')
        #print(f'GS energy per-site: e_0/J = {round(min_e/(N), 5)}')
