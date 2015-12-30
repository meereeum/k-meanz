import tensorflow as tf
import numpy as np
from PIL import Image
import random
import itertools
import os
import datetime


class kmeans():
    def __init__(self, filepath, rounds=2, k=10, scale=False, generate_all=True, outdir=None):
        self.now = ''.join(c for c in str(datetime.datetime.today()) if c in '0123456789 ')[2:13].replace(' ','_') # YYMMDD_HHMM
        self.k = k
        self.scale = scale
        self.filename = filepath
        # default to parent directory of input
        self.outdir = (os.path.dirname if not outdir else outdir)
        # basename sans extension
        self.basename = os.path.splitext(os.path.basename(filepath))[0]

        self.pixels = np.array(Image.open(filepath))
        #m, n, cols = self.pixels.shape
        #if cols > 3:
            #self.pixels = self.pixels[:,:,:3]
        #self.n_pixels = m*n
        #self.ratio = (255.0/max(m,n) if scale else 1.0) # rescale by max dimension
        #idx_lst = [ (j*self.ratio, k*self.ratio) for j in xrange(m) for k in xrange(n) ]
        #idx_arr = np.array(idx_lst).reshape((m, n, 2))
        ## 2D array = array of [m, n, r, g, b] arrays
        #self.arr = np.concatenate((idx_arr, self.pixels), axis=2).\
                    #ravel().reshape((m*n,5))

        #centroids = np.array(random.sample(self.arr, self.k), dtype=np.float32)
        #centroids_in, centroids_out = self._build_graph()

        with tf.Session() as sesh:
            self._image_to_data()
            self._build_graph()
            sesh.run(tf.initialize_all_variables())
            feed_dict = None # first round initialized with random centroids by tf node
            for i in xrange(rounds):
                centroids = sesh.run(self.centroids, feed_dict)
                feed_dict = {self.centroids_in: centroids}
                print "round {} -->  centroids: {}".format(i,centroids)

                if generate_all:
                    self.generate_image(round_id=i)

            if not generate_all: # final image only
                self.generate_image(round_id=i)


    def _image_to_data(self):
        with open(self.filename, 'rb') as f:
            img_str = f.read()
        pixels = tf.image.decode_jpeg(img_str)
        m, n, chann = tf.shape(pixels).eval()
        ratio = (255.0/max(m,n) if self.scale else 1.0) # rescale by max dimension
        self.ratio = tf.constant(ratio, dtype=tf.float32)
        #idxs = tf.constant([(j*self.ratio,k*self.ratio) for j in xrange(m) for k in xrange(n)])
        idxs = tf.mul(self.ratio, tf.constant([(j,k) for j in xrange(m)
                                               for k in xrange(n)], dtype=tf.float32))
        self.arr = tf.concat(1, [idxs, tf.to_float(tf.reshape(pixels,
                                                              shape=(m*n,chann)))])
        self.n_pixels, self.dim = tf.shape(self.arr).eval() # i.e. m*n, chann + 2


    def _build_graph(self):
        """Construct tensorflow nodes for round of clustering"""
        #pixels = tf.constant(self.arr, name="pixels", dtype=tf.float32)
        #self.centroids_in = tf.placeholder(name="centroids_in", dtype=tf.float32,
                                           #shape=(self.k,5))
        #self.centroids_in = tf.Variable(np.array(random.sample(self.arr, self.k),
                                                 #dtype=np.float32), name="centroids_in")
        # N.B. without tf.Variable, makes awesome glitchy clustered images
        self.centroids_in = tf.Variable(tf.slice(tf.random_shuffle(self.arr),
                                     [0,0],[self.k,-1]), name="centroids_in")
        # tiled should be shape(self.n_pixels,self.k,5)
        tiled_roids = tf.tile(tf.expand_dims(self.centroids_in,0),
                              multiples=[self.n_pixels,1,1], name="tiled_roids")
        tiled_pix = tf.tile(tf.expand_dims(self.arr,1),
                            multiples=[1,self.k,1], name="tiled_pix")

        def radical_euclidean_dist(x,y):
            """Takes in 2 tensors and returns euclidean distance radical, i.e. dist**2"""
            return tf.square(tf.sub(x,y))

        # no need to take square root b/c positive reals and sqrt are isomorphic
        # should be shape(self.n_pixels, self.k)
        distances = tf.reduce_sum(radical_euclidean_dist(tiled_pix, tiled_roids),
                                  reduction_indices=2, name="distances")
        # should be shape(self.n_pixels)
        nearest = tf.to_int32(tf.argmin(distances,1), name="nearest")
        # should be list of len self.k with tensors of shape(size_cluster, 5)
        self.clusters = tf.dynamic_partition(self.arr,nearest,self.k)
        # should be shape(self.k,5)
        self.centroids = tf.reshape(
            tf.concat(0,[tf.reduce_mean(cluster,0) for cluster in self.clusters]),
            shape=(self.k,self.dim), name="centroids_out")


    def generate_image(self, round_id, save=True):
        new_arr = np.empty_like(self.pixels, dtype=np.uint8)
        #centroids_rgb = np.int32(self.centroids.eval()[:,2:])
        centroids_rgb = tf.to_int32(tf.slice(self.centroids,[0,2],[-1,-1])).eval()
        for centroid_rgb, cluster in itertools.izip(centroids_rgb,self.clusters):
            #cluster_mn = np.int32(cluster.eval()[:,:2]/self.ratio)
            cluster_mn = tf.to_int32(tf.div(tf.slice(cluster,[0,0],[-1,2]),
                                            self.ratio))
            for pixel in cluster_mn.eval():
                new_arr[tuple(pixel)] = centroid_rgb
        new_img = tf.image.encode_jpeg(tf.constant(new_arr, dtype=tf.uint8)).eval()
        #new_img = Image.fromarray(new_arr)
        #new_img.show()

        if save:
            outfile = os.path.join(self.outdir, '{}_{}_k{}_{}.jpg'.\
                                format(self.basename,self.now,self.k,round_id))
        with open(outfile, 'w') as f:
            f.write(new_img)
        #new_img.save(outfile, format='JPEG')



if __name__=="__main__":
    INFILE = '/Users/miriamshiffman/Downloads/536211-78101.jpg'
    OUTDIR = '/Users/miriamshiffman/Downloads/kmeanz'
    kmeans(INFILE, outdir=OUTDIR, k=12, rounds=2)
