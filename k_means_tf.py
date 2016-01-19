import itertools
import os
import datetime
from PIL import Image
import numpy as np
import tensorflow as tf

import argparser


class kmeans():
    """k-means clustering of image data using Google's TensorFlow"""
    def __init__(self, filepath, rounds = 2, k = 10, scale = False,
                 generate_all = True, outdir = None):
        self.now = ''.join(c for c in str(datetime.datetime.today())
                           if c in '0123456789 ')[2:13].replace(' ','_') # YYMMDD_HHMM
        self.k = k
        self.scale = scale
        self.filename = os.path.expanduser(filepath)
        self.outdir = (os.path.expanduser(outdir) if outdir else outdir)
        # basename sans extension
        self.basename = os.path.splitext(os.path.basename(filepath))[0]

        with tf.Session() as sesh:
            self._image_to_data()
            print '\nimage shape = ({},{},{})'.format(self.m, self.n, self.chann)
            print 'pixels: {}\n'.format(self.n_pixels)
            self._build_graph()
            sesh.run(tf.initialize_all_variables())
            for i in xrange(rounds):
                self.update_roids.eval()
                print "round {} -->  centroids: {}".format(i,self.centroids.eval())
                if generate_all or i==(rounds-1): # all or final image only
                    self.generate_image(round_id=i)


    def _image_to_data(self):
        """Convert image to 1D array of image data: (m, n, R, G, B) per pixel"""
        with open(self.filename, 'rb') as f:
            img_str = f.read()
        pixels = tf.image.decode_jpeg(img_str)
        self.m, self.n, self.chann = tf.shape(pixels).eval()
        ratio = (255.0/max(self.m,self.n) if self.scale else 1.0) # rescale by max dimension
        self.ratio = tf.constant(ratio, dtype=tf.float32)
        idxs = tf.mul(self.ratio, tf.constant([(j,k) for j in xrange(self.m)
                                               for k in xrange(self.n)], dtype=tf.float32))
        self.arr = tf.concat(1, [idxs, tf.to_float(tf.reshape(pixels, shape=
                                                              (self.m*self.n,self.chann)))])
        self.n_pixels, self.dim = tf.shape(self.arr).eval() # i.e. m*n, chann + 2


    def _build_graph(self):
        """Construct tensorflow nodes for round of clustering"""
        # N.B. without tf.Variable, makes awesome glitchy clustered images
        self.centroids_in = tf.Variable(tf.slice(tf.random_shuffle(self.arr),
                                     [0,0],[self.k,-1]), name="centroids_in")
        # tiled should be shape(self.n_pixels,self.k,5)
        tiled_pix = tf.tile(tf.expand_dims(self.arr,1),
                            multiples=[1,self.k,1], name="tiled_pix")

        # no need to take square root b/c positive reals and sqrt are isomorphic
        def radical_euclidean_dist(x,y):
            """Takes in 2 tensors and returns euclidean distance radical, i.e. dist**2"""
            return tf.square(tf.sub(x,y))

        # should be shape(self.n_pixels, self.k)
        distances = tf.reduce_sum(radical_euclidean_dist(tiled_pix, self.centroids_in),
                                  reduction_indices=2, name="distances")
        # should be shape(self.n_pixels)
        nearest = tf.to_int32(tf.argmin(distances,1), name="nearest")

        # should be list of len self.k with tensors of shape(size_cluster, 5)
        self.clusters = tf.dynamic_partition(self.arr,nearest,self.k)
        # should be shape(self.k,5)
        self.centroids = tf.pack([tf.reduce_mean(cluster,0) for cluster in self.clusters],
            name="centroids_out")
        self.update_roids = tf.assign(self.centroids_in, self.centroids)


    def generate_image(self, round_id, save = True):
        #centroids_rgb = self.centroids.eval()[:,2:]
        centroids_rgb = tf.slice(self.centroids,[0,2],[-1,-1]).eval()
        if save:
            addon = ('_scaled' if self.scale else '')
            outfile = os.path.join(self.outdir, '{}_{}_k{}_{}{}.jpg'.\
                                format(self.basename,self.now,self.k,round_id,addon))
        def array_put():
            """Generate new image array by putting (R,G,B) values in place for each pixel"""
            new_arr = np.empty([self.m,self.n,self.chann], dtype=np.uint8)
            for centroid_rgb, cluster in itertools.izip(centroids_rgb,self.clusters):
                #cluster_mn = np.int32(cluster.eval()[:,:2]/self.ratio)
                cluster_mn = tf.to_int32(tf.div(tf.slice(cluster,[0,0],[-1,2]),
                                                self.ratio))
                for pixel in cluster_mn.eval():
                    new_arr[tuple(pixel)] = centroid_rgb
            new_img = tf.image.encode_jpeg(tf.constant(new_arr, dtype=tf.uint8)).eval()
            if save:
                with open(outfile, 'w') as f:
                    f.write(new_img)
                #import code; code.interact(local=locals())
                os.popen("open '{}'".format(outfile))

        def array_sort():
            """Generate new image array by sorting (m,n,R,G,B) values according to position (m,n),
            then slicing down to (R,G,B) per pixel"""
            to_concat = []
            for centroid_rgb, cluster in itertools.izip(centroids_rgb,self.clusters):
                # no need to revisit ratio
                new_idxed_arr = tf.concat(1,[tf.slice(cluster,[0,0],[-1,2]),
                                             tf.tile(tf.expand_dims(tf.constant(centroid_rgb),0),
                                                     multiples=[len(cluster.eval()),1])])
                to_concat.append(new_idxed_arr)
            concated = tf.concat(0,to_concat)
            #sorted_by_idx = np.sort(concated.eval())[:,2:]
            sorted_arr = np.array(sorted([list(arr) for arr in concated.eval()]),
                                  dtype=np.uint8)[:,2:]
            new_img = Image.fromarray(sorted_arr.reshape([self.m,self.n,self.chann]))
            if save:
                new_img.save(outfile, format='JPEG')
                os.popen("open '{}'".format(outfile))
            else:
                new_img.show()

        array_sort()
        #array_put()
        print



def doWork():
    args, kwargs = argparser.parse_args()
    kmeans(*args, **kwargs)



if __name__=="__main__":
    doWork()
