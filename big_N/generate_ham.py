from clusters_lattice_bigN_mem_eff_mk2 import SpinConfiguration
from numpy import array
import sys

if __name__ == "__main__":
    key, S_z_tot, delta, lamb, n_proc, kx, ky = sys.argv[1:]
    N = int(key[:2])

    sc = SpinConfiguration(key=key, magnon_number=int(N/2) + int(S_z_tot), n_workers = n_proc, lowest_eignstates=5, delta=float(delta), lamb=float(lamb),
                           k=array([float(kx), float(ky)]), print_data=False, force_ham_gen=True)
