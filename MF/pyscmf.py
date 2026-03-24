import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def gamma_k(_kx, _ky):
    return (np.cos(_kx) + np.cos(_ky))/2

def gamma2_k(_kx, _ky, eta = 1.):
    return 1/4 * (np.cos(_kx)**2 + np.cos(_ky)**2 + 2 * np.cos(_kx) * np.cos(_ky) * np.cos(np.pi/2 * (1 - eta)))

def a_func(t, n, delta = 1., lamb = 0., alpha = 1.):
    return 2 * (delta + t - 2 * n * (t/2 + alpha * lamb * delta))

def b_func(t, n, delta = 1., lamb = 0., beta = 1.):
    return 2 * (1 - 2 * n + n**2 + 3/4 * (t)**2 + beta * t * lamb * delta )

def get_bog_pars(t, n, _kx, _ky, delta = 1., lamb = 0., eta = 1., alpha = 1., beta = 1.):
    a = a_func(t, n, delta, lamb, alpha)
    b = b_func(t, n, delta, lamb, beta)

    wk = np.abs(np.sqrt(a**2 - b**2 * gamma2_k(_kx, _ky, eta) + 0j))
    
    return a, b, wk

def t_func(b, wk, gamma2k, N):
    return np.abs(b)/N * np.real(np.sum(np.divide(gamma2k, wk)))

def n_func(a, wk, N):
    return 1/(2*N) * np.sum(np.abs(np.divide(a, wk) - 1))


def get_flow(n0, t0, delta = 1., lamb = 0., eta = 1., alpha = 1., beta = 1., k_bins = 100):
    # strating parameters
    n_old = n0
    t_old = t0

    N = k_bins**2
    k = np.linspace(-np.pi, np.pi, k_bins, endpoint = False)

    px, py = np.meshgrid(k, k, indexing = 'ij')

    # get general parameters
    a, b, wk = get_bog_pars(t_old, n_old, px, py, delta, lamb, eta, alpha, beta)
    gamma2k = gamma2_k(px, py, eta)

    # get new t
    t_new = t_func(b, wk, gamma2k, N)

    # get new n
    n_new = n_func(a, wk, N)

    # get flow
    t_flow = t_new - t_old
    n_flow = n_new - n_old

    return n_flow, t_flow


def get_flow_map(delta = 1., lamb = 0., eta = 1., alpha = 1., beta = 1., n_min = 0., n_max = 1., t_min = 0., t_max = 1., n_bins = 101, t_bins = 101, k_bins = 100, _show_tqdm = True):
    np.seterr(all='ignore')
    ns = np.linspace(n_min, n_max, n_bins)
    ts = np.linspace(t_min, t_max, t_bins)

    flow_map = np.zeros((len(ts), len(ns)))

    for i_n, n in tqdm(enumerate(ns), total = len(ns), disable = not _show_tqdm):
        for it, t in enumerate(ts):
            n_flow, t_flow = get_flow(n, t, delta, lamb, eta, alpha = alpha, beta = beta, k_bins = k_bins)
            flow_map[it, i_n] = np.atan2(t_flow, n_flow)

    return flow_map, ns, ts


def process_map_col(params):
    np.seterr(all='ignore')

    i_n, ns, ts, delta, lamb, eta, alpha, beta, k_bins = params
    mini_map = np.zeros((len(ts), len(ns)))
    n = ns[i_n]

    for it, t in enumerate(ts):
        n_flow, t_flow = get_flow(n, t, delta, lamb, eta, alpha = alpha, beta = beta, k_bins = k_bins)
        mini_map[it, i_n] = np.atan2(t_flow, n_flow)
    
    return mini_map

def get_flow_map_mp(delta = 1., lamb = 0., eta = 1., alpha = 1., beta = 1., n_min = 0., n_max = 1., t_min = 0., t_max = 1., n_bins = 101, t_bins = 101, k_bins = 100, _show_tqdm = True):
    np.seterr(all='ignore')
    ns = np.linspace(n_min, n_max, n_bins)
    ts = np.linspace(t_min, t_max, t_bins)

    flow_map = np.zeros((len(ts), len(ns)))

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as exec:
        futures = []

        for i_n in range(len(ns)):
            futures.append(exec.submit(process_map_col, (i_n, ns, ts, delta, lamb, eta, alpha, beta, k_bins)))

        for future in tqdm(as_completed(futures), total = len(futures), disable = not _show_tqdm):
            flow_map += future.result()


    return flow_map, ns, ts

def plot_polar_dir(ax):
    N = 400
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)

    angle_cb = np.arctan2(-Y, X)
    radius = np.sqrt(X**2 + Y**2)

    mask = radius <= 1

    norm = plt.Normalize(-np.pi, np.pi)
    colors = plt.get_cmap('twilight')(norm(angle_cb))
    colors[~mask] = (1, 1, 1, 0)  # transparent outside

    ax.imshow(colors, extent=(-1, 1, -1, 1))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Direction")

def plot_flow_map(map, ns, ts, title = None, additional_points = [], show_dir_plot = True, show = False):
    
    if show_dir_plot:
        fig, ax = plt.subplots(figsize=(8, 5))

    if title is not None:
        plt.title(title)
    
    plt.imshow(map, cmap='twilight', aspect = 'auto', extent = (ns[0], ns[-1], ts[0], ts[-1]))
    plt.xlabel('n')
    plt.ylabel('t')

    for point in additional_points:
        plt.scatter([point[0]], [point[1]], s=50, facecolors = 'none', edgecolors = 'y')

    if np.max(ns) > 1. or np.min(ns) < 0. or np.max(ts) > 1. or np.min(ts) < 0.:
        plt.hlines([0., 1.], 0., 1., ls = '--', colors = 'white', alpha = .2)
        plt.vlines([0., 1.], 0., 1., ls = '--', colors = 'white', alpha = .2)

    if show_dir_plot:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="30%", pad=0.4)
        plot_polar_dir(cax)

    if show:
        plt.show()
