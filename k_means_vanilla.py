import numpy as np
from PIL import Image
import random
import os

class kmeans():
    def __init__(self, filepath, k = 10, rounds=10, scale=False):
        self.basename = os.path.basename(filepath)
        img = Image.open(filepath)
        #img.show()
        self.k = k
        # initialize pixel map = m x n array (row by column rather than x & y)
        self.pixels = np.array(img)

        m, n, cols = self.pixels.shape
        if cols > 3:
            self.pixels = self.pixels[:,:,:3]

        dims = (m,n)
        if scale:
            self.ratio = 255.0/max(dims)
            dims = tuple(self.ratio*d for d in dims)
        else:
            self.ratio = 1.0

        idx_lst = [ (j*self.ratio, k*self.ratio) for j in xrange(m) for k in xrange(n) ]
        idx_arr = np.array(idx_lst).reshape((m, n, 2))

        # 2D array = array of [m, n, r, g, b] arrays
        self.arr = np.concatenate((idx_arr, self.pixels), axis=2).\
                   ravel().reshape((m*n,5))

        centroids = random.sample(self.arr, self.k)
        self.centroids_history = []

        for _ in xrange(rounds):
            centroids = self.update_centroids(centroids=centroids)
            self.centroids_history.append(centroids)
            print "cluster round {} done! --> {} 'roids".format(_, len(centroids))


    #@property
    #def centroids(self):
        ##return self._centroids
        #print 'hi'
        #return self.centroids_history[-1]
#
    #@centroids.setter
    #def centroids(self, value):
        #self._centroids = self.centroids_history[-1]


    def update_centroids(self, centroids, with_dropout=True):
        d_centroids = {tuple(k): np.zeros([6], dtype=np.float32)
                       for k in centroids}
        for arr in self.arr:
            # dictionary values = [num points, summed dim1, ..., summed dim 5]
            d_centroids[self.nearest_centroid(arr, centroids)] \
                += np.concatenate([[1.0], arr])
        #for k,v in d_centroids.iteritems():
            #print 'centroid {}: {} assigned'.format(k,v[0])
        return [ val_arr[1:]/val_arr[0] for val_arr in d_centroids.itervalues()
                 if val_arr[0] > 0 ]


    def nearest_centroid(self, pixel, centroids):
        """Find np array representing best centroid by minimizing distance, and return as tuple (b/c hashable)"""
        def euclidean_dist(v1,v2):
            """Returns Frobenius norm (float) of two vectors (=Euclidean distance)"""
            return np.linalg.norm(v1-v2)
        dists = [ (k, euclidean_dist(pixel,k)) for k in centroids ]
        best_k, _ = min( dists, key = lambda t: t[1] )
        return tuple(best_k)


    def cluster(self, centroids):
        """Cluster image array by assigning pixels to the nearest element among the input list of centroids, and return as dictionary"""
        d_clusters = {tuple(k): [] for k in centroids}
        for arr in self.arr:
            # dictionary values = [num points, summed dim1, ..., summed dim 5]
            d_clusters[self.nearest_centroid(arr, centroids)].append(arr)
        return d_clusters


    def generate_image(self, round_idx=-1, outdir=None):
        """Generate new image by clustering pixels according to given centroids (defaulting to last round of clustering) and assiging centroid RGB to each cluster"""
        assert round_idx <= len(self.centroids_history)
        new_arr = np.empty_like(self.pixels, dtype=np.uint8)
        d_clusters = self.cluster(centroids=self.centroids_history[round_idx])
        for centroid, pixels in d_clusters.iteritems():
            centroid_rgb = [int(rgb) for rgb in centroid[-3:]]
            for pixel in pixels:
                new_arr[int(pixel[0]/self.ratio),int(pixel[1]/self.ratio)] \
                    = centroid_rgb
        new_img = Image.fromarray(new_arr)
        new_img.show()
        if outdir:
            outfile = os.path.join(outdir, self.basename + '_kmeanz_{}'.format(round_idx))
            new_img.save(outfile, format='JPEG')


if __name__ == "__main__":
    INFILE = '/Users/miriamshiffman/Downloads/536211-78101.jpg'
    OUTDIR = '/Users/miriamshiffman/Downloads'
    x = kmeans(filepath=INFILE, k=1000, rounds=5)
    for i in xrange(5):
        x.generate_image(round_idx=i, outdir=OUTDIR)
