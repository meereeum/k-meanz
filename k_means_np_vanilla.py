import random
import os
import datetime
from PIL import Image
import numpy as np

import argparser


class kmeans():
    def __init__(self, filepath, k = 10, rounds = 10, scale = False,
                 generate_all = False, outdir = None):
        self.now = ''.join(c for c in str(datetime.datetime.today())
                           if c in '0123456789 ')[2:13].replace(' ','_') # YYMMDD_HHMM
        self.k = k
        self.scale = scale
        self.filename = os.path.expanduser(filepath)
        self.outdir = (os.path.expanduser(outdir) if outdir else outdir)
        # basename sans extension
        self.basename = os.path.splitext(os.path.basename(filepath))[0]

        img = Image.open(filepath)
        #img.show()
        # initialize pixel map = m x n array (row by column rather than x & y)
        self.pixels = np.array(img)

        m, n, cols = self.pixels.shape
        if cols > 3: # clean gratuitous additions to RGB data
            self.pixels = self.pixels[:,:,:3]
        self.ratio = (255.0/max(m,n) if scale else 1.0) # rescale by max dimension

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
            if generate_all:
                self.generate_image()

        if not generate_all: # final image only
            self.generate_image()


    def update_centroids(self, centroids):
        """Given previous centroids, cluster pixels (tallying sums and cluster sizes), and return new list of centroids"""
        d_centroids = {tuple(k): np.zeros([6], dtype=np.float32)
                       for k in centroids}
        for arr in self.arr:
            # dictionary values = [num points, summed dim1, ..., summed dim 5]
            d_centroids[self.nearest_centroid(arr, centroids)] \
                += np.concatenate([[1.0], arr])
        #for k,v in d_centroids.iteritems():
            #print 'centroid {}: {} assigned'.format(k,v[0])
        return [ val_arr[1:]/val_arr[0] for val_arr in d_centroids.itervalues()
                 if val_arr[0] > 0 ] # drop centroids with empty clusters


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


    def generate_image(self, round_idx = -1, save = True):
        """Generate new image by clustering pixels according to given centroids (defaulting to last round of clustering) and assiging centroid RGB to each cluster"""
        assert round_idx <= len(self.centroids_history)
        new_arr = np.empty_like(self.pixels, dtype=np.uint8)
        d_clusters = self.cluster(centroids = self.centroids_history[round_idx])
        for centroid, pixels in d_clusters.iteritems():
            centroid_rgb = [int(rgb) for rgb in centroid[-3:]]
            for pixel in pixels:
                new_arr[int(pixel[0]/self.ratio),int(pixel[1]/self.ratio)] \
                    = centroid_rgb
        new_img = Image.fromarray(new_arr)
        new_img.show()
        if save:
            if round_idx < 0:
                # rename round_idx for outfile name
                round_idx = len(self.centroids_history) + round_idx
            outfile = os.path.join(self.outdir, '{}_kmeanz_{}.jpg'.\
                                   format(self.basename,round_idx))
            new_img.save(outfile, format='JPEG')



def doWork():
    args, kwargs = argparser.parse_args()
    kmeans(*args, **kwargs)



if __name__ == "__main__":
    doWork()
