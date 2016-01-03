import tensorflow as tf
import numpy as np
from PIL import Image
import random
import itertools
import os
import datetime


class kmeans():
    def __init__(self, filepath, rounds=2, k=10, scale=False, generate_all=True, outdir=None):
        self.now = ''.join(c for c in str(datetime.datetime.today())
                           if c in '0123456789 ')[2:13].replace(' ','_') # YYMMDD_HHMM
        self.k = k
        self.scale = scale
        self.filename = filepath
        # default to parent directory of input
        self.outdir = (os.path.dirname if not outdir else outdir)
        # basename sans extension
        self.basename = os.path.splitext(os.path.basename(filepath))[0]

        self.pixels = np.array(Image.open(filepath))

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
        self.m, self.n, chann = tf.shape(pixels).eval()
        ratio = (255.0/max(m,n) if self.scale else 1.0) # rescale by max dimension
        self.ratio = tf.constant(ratio, dtype=tf.float32)
        #idxs = tf.constant([(j*self.ratio,k*self.ratio) for j in xrange(m) for k in xrange(n)])
        idxs = tf.mul(self.ratio, tf.constant([(j,k) for j in xrange(self.m)
                                               for k in xrange(self.n)], dtype=tf.float32))
        self.arr = tf.concat(1, [idxs, tf.to_float(tf.reshape(pixels,
                                                              shape=(self.m*self.n,chann)))])
        self.n_pixels, self.dim = tf.shape(self.arr).eval() # i.e. m*n, chann + 2


    def _build_graph(self):
        """Construct tensorflow nodes for round of clustering"""
        # N.B. without tf.Variable, makes awesome glitchy clustered images
        self.centroids_in = tf.Variable(tf.slice(tf.random_shuffle(self.arr),
                                     [0,0],[self.k,-1]), name="centroids_in")
        # tiled should be shape(self.n_pixels,self.k,5)
        #tiled_roids = tf.tile(tf.expand_dims(self.centroids_in,0),
                              #multiples=[self.n_pixels,1,1], name="tiled_roids")
        #tiled_pix = tf.tile(tf.expand_dims(self.arr,1),
                            #multiples=[1,self.k,1], name="tiled_pix")

        def radical_euclidean_dist(x,y):
            """Takes in 2 tensors and returns euclidean distance radical, i.e. dist**2"""
            return tf.square(tf.sub(x,y))

        # no need to take square root b/c positive reals and sqrt are isomorphic
        # should be shape(self.n_pixels, self.k)
        #distances = tf.reduce_sum(radical_euclidean_dist(tiled_pix, tiled_roids),
                                  #reduction_indices=2, name="distances")
        # should be shape(self.n_pixels)
        #nearest = tf.to_int32(tf.argmin(distances,1), name="nearest")

        nearest = tf.to_int32(tf.argmin(tf.reduce_sum(
            radical_euclidean_dist( tf.tile(tf.expand_dims(self.arr,1),
                                           multiples=[1,self.k,1]),
                                   tf.tile(tf.expand_dims(self.centroids_in,0),
                                           multiples=[self.n_pixels,1,1]) ),
            reduction_indices=2),1))

        # should be list of len self.k with tensors of shape(size_cluster, 5)
        self.clusters = tf.dynamic_partition(self.arr,nearest,self.k)
        # should be shape(self.k,5)
        self.centroids = tf.reshape(
            tf.concat(0,[tf.reduce_mean(cluster,0) for cluster in self.clusters]),
            shape=(self.k,self.dim), name="centroids_out")


    def generate_image(self, round_id, save=True):
        #centroids_rgb = np.int32(self.centroids.eval()[:,2:])
        centroids_rgb = tf.slice(self.centroids,[0,2],[-1,-1]).eval()
        #centroids_rgb = tf.to_int32(tf.slice(self.centroids,[0,2],[-1,-1])).eval()

        def array_put():
            new_arr = np.empty_like(self.pixels, dtype=np.uint8)
            for centroid_rgb, cluster in itertools.izip(centroids_rgb,self.clusters):
                #cluster_mn = np.int32(cluster.eval()[:,:2]/self.ratio)
                cluster_mn = tf.to_int32(tf.div(tf.slice(cluster,[0,0],[-1,2]),
                                                self.ratio))
                for pixel in cluster_mn.eval():
                    new_arr[tuple(pixel)] = centroid_rgb
            new_img = tf.image.encode_jpeg(tf.constant(new_arr, dtype=tf.uint8))
            if save:
                outfile = os.path.join(self.outdir, '{}_{}_k{}_{}.jpg'.\
                                    format(self.basename,self.now,self.k,round_id))
                with open(outfile, 'w') as f:
                    f.write(new_img)

        def array_sort():
            to_concat = []
            for centroid_rgb, cluster in itertools.izip(centroids_rgb,self.clusters):
                new_idxed_arr = tf.concat(1,[tf.slice(cluster,[0,0],[-1,2]),
                                            tf.tile(tf.expand_dims(tf.constant(centroid_rgb),0),
                                                multiples=[len(cluster.eval()),1])])
                to_concat.append(new_idxed_arr)
            concated = tf.to_int32(tf.concat(0,to_concat))
            #sorted_by_idx = np.sort(concated.eval())[:,2:]
            sorted_arr = np.array(sorted([list(arr) for arr in concated.eval()]), dtype=np.uint8)[:,2:]
            new_img = Image.fromarray(sorted_arr.reshape([self.m,self.n,self.dim-2]))
            new_img.show()
            if save:
                outfile = os.path.join(self.outdir, '{}_{}_k{}_{}.jpg'.\
                                    format(self.basename,self.now,self.k,round_id))
                new_img.save(outfile, format='JPEG')

        #array_sort()
        array_put()
        print



if __name__=="__main__":
    #INFILE = '/Users/miriamshiffman/Downloads/536211-78101.jpg'
    INFILE = '/Volumes/Media/Pictures/to_org/Wyeth 2.jpg'
    OUTDIR = '/Users/miriamshiffman/Downloads/kmeanz'
    kmeans(INFILE, outdir=OUTDIR, k=10, rounds=2)
