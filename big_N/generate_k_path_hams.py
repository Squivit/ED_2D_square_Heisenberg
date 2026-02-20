from Betts_cluster import Cluster
from clusters_lattice_bigN import SpinConfiguration
import numpy as np

if __name__ == "__main__":
    N = 32
    num = 'A'
    key = str(N)+num

    delta = 1.
    lamb = 0.

    c = Cluster(key=key)
    path = c.get_k_space(False)[1]

    sc = SpinConfiguration(N=N, number=num, key=key, magnon_number=int(N/2), lowest_eignstates=10,
                           delta = delta, lamb = lamb, k=np.array([0., 0.]), print_data=True, force_ham_gen=True, save_ham=False, eigva_ve_only=False)
    min_e, gs_state = sc.get_ground_state()
        
    print(f'Ground state energy E_0/J = {round(min_e, 5)}')
    print(f'GS energy per-site: e_0/J = {round(min_e/(N), 5)}')

    for k in path:
        print()
        print(f'Calculating k = {k} block')

        sc = SpinConfiguration(N=N, number=num, key=key, magnon_number=int(N/2)+1, lowest_eignstates=10,
                            delta = delta, lamb = lamb, k=k, print_data=True, force_ham_gen=True, save_ham=False, eigva_ve_only=False)
        min_e, gs_state = sc.get_ground_state()
            
        print(f'Ground state energy E_0/J = {round(min_e, 5)}')
        print(f'GS energy per-site: e_0/J = {round(min_e/(N), 5)}')
