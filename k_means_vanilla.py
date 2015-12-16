import numpy as np
from PIL import Image
import random

class kmeans():
    def __init__(self, filepath, k = 10, rounds=10, scale=False):
        self.img = Image.open(filepath)
        #self.img.show()
        self.k = k
        # initialize pixel map = m x n array (row by column rather than x & y)
        pixels = np.array(self.img)

        self.m, self.n, cols = pixels.shape
        if cols > 3:
            pixels = pixels[:,:,:3]

        self.dims = (self.m,self.n)
        if scale:
            self.ratio = 255.0/max(self.dims)
            self.dims = tuple(self.ratio*d for d in dims)
        else:
            self.ratio = 1.0

        idx_lst = [ (j*self.ratio, k*self.ratio) for j in xrange(self.m) for k in xrange(self.n) ]
        idx_arr = np.array(idx_lst).reshape((self.m, self.n, 2))

        # 2D array = array of [m, n, r, g, b] arrays
        self.arr = np.concatenate((idx_arr, pixels), axis=2).\
                   ravel().reshape((1,self.m*self.n,5))

        centroids = random.sample(self.arr, self.k)
        import code; code.interact(local=locals())
        centroids_history = [centroids]
        # initialize list containing dictionary of k starting values (centroids)
        # must convert np array to hashable type (tuple) for key
        # will append to list after each stage of clustering, so access most recent by [-1] index
        self.d_k_clusters = { (k_mn + tuple(self.arr[k_mn])): [] for k_mn in k_vals }
        self.d_k_clusters_lst = [self.d_k_clusters]
        # 3D numpy array populated with arrays representing corresponding [row, column]

        for _ in xrange(rounds):
            self.assign_to_clusters()
            centroids = self.update_centroids(centroids=centroids)
            centroids_history.append(centroids)

    def assign_to_clusters():
        pass

    def generate_image():
        pass

    def update_centroids():
        pass



if __name__ == "__main__":
    x = kmeans(filepath='/Users/miriamshiffman/Downloads/536211-78101.jpg')
