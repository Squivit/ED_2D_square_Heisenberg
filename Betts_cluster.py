import numpy as np
import matplotlib.pyplot as plt

betts_clusters = {
    '16A': ( [4, 2], [0, 4] ),
    '16B': ( [4, 0], [0, 4] ),
    
    '18A': ( [3, 3], [-3, 3] ),
    '18B': ( [3, 3], [-2, 4] ),

    '20A': ( [4, 2], [-2, 4] ),
    '20E': ( [5, 1], [0, 4] ),
    
    '22A': ( [2, 4], [-5, 1] ),
    
    '24A': ( [4, 4], [-4, 2] ),
    '24C': ( [6, 0], [0, 4] ),
    # '24D': ( [5, -1], [1, -5] ), # buggy af dunno why
    '24E': ( [3, 3], [-5, 3] ),
    '24F': ( [4, 4], [-3, 3] ),

    '26A': ( [4, 2], [-3, 5] ),
    '26B': ( [1, 5], [-5, 1] ),

    '28A': ( [2, 4], [-6, 2] ),
    '28B': ( [5, 3], [-1, 5] ),
    
    '30A': ( [5, 3], [-5, 3] ),
    
    '32A': ( [4, 4], [-4, 4] ),
    '32B': ( [2, 6], [-4, 4] ),
    '32C': ( [3, 5], [-4, 4] ),
}

# bipartite topological imperfections
cluster_topological_imperfections = {
    '16A': 0,
    '16B': 1,
    
    '18A': 0,
    '18B': 0,

    '20A': 0,
    '20E': 1,
    
    '22A': 0,
    
    '24A': 0,
    '24C': 2,
    # '24D': 0, # buggy af dunno why
    '24E': 0,
    '24F': 0,

    '26A': 0,
    '26B': 1,

    '28A': 1,
    '28B': 0,
    
    '30A': 0,
    
    '32A': 0,
    '32B': 0,
    '32C': 0,
}

usefull_clusters = [ '18A', '24A', '32A' ]

class Cluster:
    def __init__(self, N = 16, num = 'A', key = None, bigmap_side = 30, bigmap_shift = 10):
        
        if key == None:
            self.key = str(N) + num
        else:
            self.key = key
        
        coords = get_betts_cluster(self.key)
        a_vec, b_vec = betts_clusters[self.key]
        
        self.coords = np.array(coords)
        
        self.sorted_coords = coords.copy()
        self.sorted_coords.sort(key = lambda coord: np.sqrt(coord[0]**2 + coord[1]**2))
        self.sorted_coords = np.array(self.sorted_coords)
                
        self.shift = bigmap_shift
        self.rows = self.coords[:, 0] + self.shift
        self.columns = self.coords[:, 1] + self.shift
        
        self.a = np.array(a_vec)
        self.b = np.array(b_vec)
        
        self.directions = [
            self.a,
            self.b,
            self.a + self.b,
            self.a - self.b
        ]
        
        self.n = len(coords)
        self.pos_to_idx = {tuple(pos): i for i, pos in enumerate(coords)}
        
        self.generate_cluster_map(side_length=bigmap_side)
        
        self.bigmap_side = bigmap_side
        
        self.k_space = None
    
    def generate_cluster_map(self, side_length = 15):
        self.cluster_map = np.zeros( (side_length, side_length) , dtype=bool)
        
        self.cluster_map[ self.rows, self.columns ] = 1
    
    def get_index_by_pos(self, pos):
        return self.pos_to_idx[pos]
    
    def plot_cluster(self):
        plt.imshow(self.cluster_map, cmap = 'plasma')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.show()
        
    def plot_rolled_cluster(self):
        rolled_cluster = self.cluster_map.copy().astype(float)
        
        for dir in self.directions:
            rolled_cluster += np.multiply(.5, np.roll(self.cluster_map, dir, axis=(0, 1)))
            rolled_cluster += np.multiply(.5, np.roll(self.cluster_map, -dir, axis=(0, 1)))

        plt.imshow(rolled_cluster, cmap = 'plasma')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.show()
    
    def bosons_to_cluster(self, bosons):
        mapped_to_cluster = np.zeros(self.cluster_map.shape, dtype=int)
        
        mapped_to_cluster[self.rows, self.columns] = bosons
                
        rolled_cluster = mapped_to_cluster.copy()
        
        return rolled_cluster

    def bosons_to_cluster_rolled(self, bosons):
        mapped_to_cluster = np.zeros(self.cluster_map.shape, dtype=int)
        
        mapped_to_cluster[self.rows, self.columns] = bosons
                
        rolled_cluster = mapped_to_cluster.copy()
        
        for dir in self.directions:
            rolled_cluster += np.roll(mapped_to_cluster, dir, axis=(0, 1))
            rolled_cluster += np.roll(mapped_to_cluster, -dir, axis=(0, 1))

        return rolled_cluster

    def get_k_space(self, plot = False, inv_x = False, inv_y = False):
        
        if self.k_space is  None:
            self.k_space = []
            
            supercell_matrix = np.array(
                [self.a, self.b]
            )
                        
            reci_matrix = 2 * np.linalg.inv(supercell_matrix).T
            
            # Compute reciprocal supercell vectors
            g1 = reci_matrix @ np.array([1, 0])
            g2 = reci_matrix @ np.array([0, 1])

            # Generate 20 unique k-points: m, n in 0..4 and filter out duplicates
            k_points = []
            seen = set()
            for m in range(5):
                for n in range(5):
                    k = m * g1 + n * g2
                    # Round to avoid floating point duplicates
                    k_rounded = tuple(np.round(k, decimals=10))
                    if k_rounded not in seen and len(k_points) < 20:
                        seen.add(k_rounded)
                        k_points.append(k)

            k_points = np.array(k_points)
            self.k_space = k_points

            self.GXMG_path = []
            GX = []
            XM = []
            MG = []
            
            for k in k_points:
                if k[1] == 0 and k[0] >= 0 and k[0] <= 1.:
                    GX.append(k)
                    self.GXMG_path.append(k)
                elif k[0] == 1. and k[1] >= 0 and k[1] <= 1.:
                    XM.append(k)
                    self.GXMG_path.append(k)
                    
                if k[0] == k[1] and k[0] > 0 and k[0] < 1.:
                    MG.append(k)
                    self.GXMG_path.append(k)
                    
            # lambda k: k[0] + 2 * k[1] + 10 * np.sign(k[0]) * np.sign(k[1]) - k[0] * k[1]
            self.GXMG_path.sort(key= lambda k: k[0] + k[1] + (5 - k[0] * k[1] * 3) * np.sign(k[0] * k[1]) * int(k[0] - k[1] == 0) )
            self.GXMG_path = np.array(self.GXMG_path)
            
            if plot:
                #print(self.k_space)
                # Plot the k-points
                plt.figure(figsize=(6,6))
                
                max_x =  1. * (1 - 2*int(inv_x))
                max_y =  1. * (1 - 2*int(inv_y))
                
                plt.plot([0., max_x], [0., 0.], 'g--')
                plt.plot([max_x, max_x], [0., max_y], 'g--')
                plt.plot([0., max_x], [0., max_y], 'r--')
                
                plt.scatter(k_points[:,0], k_points[:,1], c='blue', s=40, marker='o')
                plt.scatter(self.GXMG_path[:,0], self.GXMG_path[:,1], c='purple', s=40, marker='o')
                
                plt.title(f"k-points in supercell for {self.key}")
                plt.xlabel(r"$k_x/\pi$")
                plt.ylabel(r"$k_y/\pi$")
                #plt.axhline(0, color='gray', linewidth=0.5)
                #plt.axvline(0, color='gray', linewidth=0.5)
                #plt.grid(True)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.tight_layout()
                

                plt.show()
        
        return self.k_space, self.GXMG_path
    


def plot_grid(x_min = -5, x_max = 5, y_min = -5, y_max = 5, symbol_size = 10, color = 'blue'):
    
    x = np.arange(x_min, x_max + 0.1, 1)
    y = np.arange(y_min, y_max + 0.1, 1)
    X, Y = np.meshgrid(x, y)
    plt.scatter(X, Y, marker="o", color = color, s=symbol_size)
    plt.xlabel('x')
    plt.ylabel('y')
    
    return X, Y

def get_betts_cluster(betts_key = '26A', shift = 0, plot = False):
    vectors = np.array([*betts_clusters[betts_key]])
    offset = np.sum(vectors, axis = 0)

    lower_a = []
    upper_a = []
    lower_b = []
    upper_b = []

    min_x = 0
    min_y = 0
    max_x = 0
    max_y = offset[1]

    for dx, dy in betts_clusters[betts_key]:
        xs = np.array([0, dx]) - shift
        ys = np.array([0, dy]) - shift
        
        if plot:
            plt.plot(xs, ys, '--g')
            
        xs = np.array([offset[0], offset[0]-dx]) - shift
        ys = np.array([offset[1], offset[1]-dy]) - shift
        
        if plot:
            plt.plot(xs, ys, '--g')
                
        if np.min(xs) < min_x:
            min_x = np.min(xs)
        
        if np.min(ys) < min_y:
            min_y = np.min(ys)
        
        if np.max(xs) > max_x:
            max_x = np.max(xs)
        
        if np.max(ys) > max_y:
            max_y = np.max(ys)
        
        if dx != 0:
            lower_a.append(dy/dx)
            upper_a.append(dy/dx)
            lower_b.append(0)
            upper_b.append(offset[1] - dy/dx * offset[0])
        else:
            lower_a.append('up')
            lower_b.append(0)
            upper_a.append('up')
            upper_b.append(offset[0])

    x = np.arange(min_x - 1, max_x + 0.1, 1)
    y = np.arange(min_y - 1, max_y + 0.1, 1)
    X, Y = np.meshgrid(x, y)
    
    # plot points inside the cluster
    pts = []

    EPSILON = 0.001
    
    for ii in range(X.shape[0]):
        for jj in range(X.shape[1]):
            x = int(X[ii, jj])
            y = int(Y[ii, jj])
            
            inside = True
            
            flip = 0
            
            for a,b in zip(lower_a, lower_b):
                if a != 'up':
                    if a * x + b - y >= ((-1)**flip) * EPSILON:
                        inside = False
                else:
                    if b - x >= EPSILON:
                        inside = False
                flip += 1
            
            flip = 0
            
            for a,b in zip(upper_a, upper_b):
                if a != 'up':
                    # also + EPSILON to not include double edge
                    if a * x + b - y <= ((-1)**flip) * EPSILON:
                        inside = False
                else:
                    if b - x <= +EPSILON:
                        inside = False
                flip += 1
            if inside:
                pts.append( [x, y] )

    # temporary solution
    if betts_key == '24A':
        pts = [np.array(p) - [-1, 1] for p in pts]

    if plot:
        plot_grid(min_x - 1, max_x + 1, min_y - 1, max_y + 1)
        for pt in pts:
            plt.scatter(pt[0], pt[1], marker="o", color = 'red', s=10)
        
        plt.title(f"Visualisation of Betts' cluster {betts_key}, N = {len(pts)}")
    
    return pts

if __name__ == "__main__":
    c = Cluster(key='24A')
    c.get_k_space(plot=False)
    print(c.coords)
