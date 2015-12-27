import tensorflow as tf
import numpy as np
from PIL import Image
import random


INFILE = '~/Downloads/536211-78101.jpg'

class kmeans():
    def __init__(self, filepath=INFILE, rounds=1, k=10):
        self.k = k
        img = Image.open(filepath)
        self.pixels = np.array(img)

        m, n, cols = pixels.shape
        if cols > 3:
            pixels = pixels[:,:,:3]
        dims = (m,n)
        self.n_pixels = m*n

        if scale:
            ratio = 255.0/max(dims)
            dims = tuple(ratio*d for d in dims)
        else:
            ratio = 1.0

        idx_lst = [ (j*ratio, k*ratio) for j in xrange(m) for k in xrange(n) ]
        idx_arr = np.array(idx_lst).reshape((m, n, 2))

        # 2D array = array of [m, n, r, g, b] arrays
        arr = np.concatenate((idx_arr, pixels), axis=2).\
                    ravel().reshape((m*n,5))

        centroids = random.sample(arr, k)
        self._build_graph()

        with tf.Session() as sesh:
            sesh.run(tf.initialize_all_variables())
            for i in xrange(rounds):
                assert type(centroids) == list
                centroids = tf.run(centroids,
                                   feed_dict={old_centroids: np.array(centroids),
                                                 pixels: arr})
                # print something


    def _build_graph():
        pixels = tf.placeholder(tf.float32, name="pixels",
                                shape=(self.n_pixels,5))
        old_centroids = tf.placeholder(tf.float32, name="old_centroids",
                                       shape=(self.k,5))
        # tiled should be shape(self.n_pixels,self.k,5)
        tiled_roids = tf.tile(tf.expand_dims(old_centroids,0),
                              multiples=[self.n_pixels,1,1])
        tiled_pix = tf.tile(tf.expand_dims(pixels,1),
                            multiples=[1,self.k,1])
        def radical_euclidean_dist(x,y):
            return tf.square(tf.sub(x,y))
        # no need to take square root b/c positive reals and sqrt are isomorphic
        # should be shape(self.n_pixels, self.k)
        distances = tf.reduce_sum(radical_euclidean_dist(tiled_pix, tiled_roids),
                                  reduction_indices=2)
        # should be shape(self.n_pixels)
        nearest = tf.argmin(distances,1)

        # should be list of len self.k with tensors of shape(size_cluster, 5)
        clusters = tf.dynamic_partition(pix,nearest,self.k)

        centroids = [tf.reduce_mean(cluster,0) for cluster in clusters]

if __name__=="__main__":
    x = kmeans() #TODO
