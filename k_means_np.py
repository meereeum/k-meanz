#!/usr/bin/env python
###############################################################################
#
# k_means.py - Implement k-means clustering on image data!
#
###############################################################################
# #
# This program is free software: you can redistribute it and/or modify #
# it under the terms of the GNU General Public License as published by #
# the Free Software Foundation, either version 3 of the License, or #
# (at your option) any later version. #
# #
# This program is distributed in the hope that it will be useful, #
# but WITHOUT ANY WARRANTY; without even the implied warranty of #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the #
# GNU General Public License for more details. #
# #
# You should have received a copy of the GNU General Public License #
# along with this program. If not, see <http://www.gnu.org/licenses/>. #
# #
###############################################################################

__author__ = "Miriam Shiffman"
__copyright__ = "Copyright 2015"
__credits__ = ["Miriam Shiffman"]
__license__ = "GPL3"
__version__ = "0.0.1"
__maintainer__ = "Miriam Shiffman"
__email__ = ""
__status__ = "Development"

###############################################################################

import numpy as np
from PIL import Image
import random
#import code

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class kmeans():
    """Cluster pixels by k means algorithm"""
    def __init__(self, filepath, k = 10):
        self.img = Image.open(filepath)
        self.img.show()
        self.k = k
        # initialize pixel map = m x n array (row by column rather than x & y)
        self.arr = np.array(self.img)

        m, n, _ = self.arr.shape
        idx_lst = [ (j,k) for j in xrange(m) for k in xrange(n) ]
        k_vals = random.sample(idx_lst, self.k)

        # initialize dictionary of k starting values (centroids)
        # must convert np array to hashable type (tuple) for key
        self.d_k_clusters = { (k_mn + tuple(self.arr[k_mn])): [] for k_mn in k_vals }

        # 3D numpy array populated with arrays representing corresponding [row, column]
        self.idx_arr = np.array(idx_lst).reshape((m, n, 2))


    def minimize_distance(self, pixel, metric):
        """Given tuple representing pixel in image, return centroid that minimizes distance by given metric"""
        dists = [ (k, metric(np.array(pixel), np.array(k))) for k in self.d_k_clusters.iterkeys() ]
        # find tuple representing best k group by minimizing distance
        best_k, _ = sorted( dists, key = lambda t: t[1] )[0]
        return best_k


    def minimize_distance_arr(self, pixel, metric):
        """Given np array representing pixel in image, return centroid that minimizes distance by given metric"""
        dists = [ (k, metric(pixel, np.array(k))) for k in self.d_k_clusters.iterkeys() ]
        # find np array representing best k group by minimizing distance
        best_k, _ = sorted( dists, key = lambda t: t[1] )[0]
        return best_k


    def assign_pixels_for_loop(self, metric):
        """Clusters pixels by iterating over numpy array with for loop"""

        def mn2mnrgb(self, t_mn):
            """Given (m,n) of pixel, returns np array of [ m, n, R, G, B ]"""
            return np.append(t_mn, self.arr[t_mn[0], t_mn[1]])

        for tup in ( (m,n) for m in xrange(0, self.arr.shape[0]) \
                               for n in xrange(0, self.arr.shape[1]) ):
            # convert (m, n) of pixel location to ((m, n), (r, g, b))
            tval = self.mn2mnrgb(tup)
            # Append to dictionary value list corresponding to key of k-mean
            # that minimizes distance by given metric
            self.d_k_clusters[ self.minimize_distance( tval, metric ) ].append(tval)


    def assign_pixels_nditer(self, metric):
        """Assign all pixels in image to closest matching group in self.d_k_groups, according to given distance metric, by iterating over numpy array of pixels with np.nditer method"""
        #print 'assigning pixels'

        #TODO: implement itertools.groupby before appending to dictionary ??
        #clusters = []
        #data = sorted()

        # try iterating over numpy array by adding multi-index to nditer
        # (iterates over 1-D array)
        it = np.nditer(self.arr, flags=['multi_index'])
        tval = []
        # use C-style do-while in order to access index at each value
        while not it.finished:
            # it.multi_index yields (i, j, index of RGB val) - where index is 0,1,2
            # it[0] at that index yields array(value, dtype=uint8)
            # tval = [i,j] + [R,G,B]
            i, j, rgb_i = it.multi_index
            # initialize tval with i,j position in array
            if rgb_i == 0:
                tval = [i, j]
            # accumulate successive R,G,B values onto tval
            tval.append(int(it[0]))
            # end of R,G,B values corresponding to that position i,j in array
            if rgb_i == 2:
                # update cluster dictionary with tval and clear for next value
                self.d_k_clusters[ self.minimize_distance( tval, metric ) ].append(tval)
                tval = []
            it.iternext()


    def assign_pixels_map(self, metric):
        # try mapping array index onto r,g,b pixel value to generate array of (m,n,r,g,b) values
        self.arr_extended = np.concatenate((self.idx_arr, self.arr), axis=2)

        def update_clusters(pixelval):
            self.d_k_clusters[ self.minimize_distance( pixelval, metric ) ].append(pixelval)
            return pixelval

        np.apply_along_axis(update_clusters, 2, self.arr_extended)


    def generate_image(self, warholize=False):
        """Once all pixels have been assigned to k clusters, use d_k_clusters to generate image data, with new pixel values determined by mean RGB of the cluster, or random color palette if warholize=True"""
        def mean_rgb(k):
            """Given key value in self.d_k_clusters, return k mean by averaging (r,g,b) value over all values in group"""
            val_arr = np.array(self.d_k_clusters[k])
            # returns np array of ints corresponding to R,G,B
            return np.mean(val_arr, axis=0).astype(int)[-3:]

        if warholize:
            random_colors = random_color_palette(self.k)

        self.new_arr = np.empty(self.arr.shape, dtype=np.uint8)
        #print 'putting pixels'
        for i, (k, v_lst) in enumerate(self.d_k_clusters.iteritems()):
            #print '.'
            pixelval = ( random_colors[i] if warholize else mean_rgb(k) )
            for m, n, _r, _g, _b in v_lst:
                self.new_arr[m, n] = pixelval

        self.new_img = Image.fromarray(self.new_arr)
        self.new_img.show()


    def generate_image_2(self, warholize=False):
        """Once all pixels have been assigned to k clusters, use d_k_clusters to generate image data, with new pixel values determined by mean RGB of the cluster, or random color palette if warholize=True"""
        def mean_mnrgb(k):
            """Given key value in self.d_k_clusters, return new centroid
            by averaging (m,n,r,g,b) over all values in group"""
            val_arr = np.array(self.d_k_clusters[k])
            new_centroid = np.mean(val_arr, axis=0)
            return tuple(new_centroid)

        # update dictionary keys with new centroid values
        for k in self.d_k_clusters.iterkeys():
            self.d_k_clusters[mean_mnrgb(k)] = self.d_k_clusters.pop(k)

        self.new_arr = np.empty(self.arr.shape, dtype=np.uint8)

        if warholize:
            random_colors = random_color_palette(self.k)

        for i, (k, v_lst) in enumerate(self.d_k_clusters.iteritems()):
            pixelval = ( random_colors[i] if warholize else (int(rgb) for rgb in k[-3:]) )
            for m, n, _r, _g, _b in v_lst:
                self.new_arr[m, n] = pixelval

        self.new_img = Image.fromarray(self.new_arr)
        self.new_img.show()


###############################################################################
###############################################################################
###############################################################################
###############################################################################


def euclidean_dist_np(p1,p2):
     """Compute Euclidean distance between 2 pts (np arrays) of any (equal) dimensions
     using numpy's Linear Alg norm
    IN: two np arrays
    OUT: float"""
     return np.linalg.norm(p1-p2)


# inspired by http://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/
def random_color_palette(n, RGB=True):
    """Generates a random, aesthetically pleasing set of n colors (list of RGB tuples if RGB; else HSV)"""
    SATURATION = 0.6
    VALUE = 0.95
    GOLDEN_RATIO_INVERSE = 0.618033988749895

    # see: https://en.wikipedia.org/wiki/HSL_and_HSV#Converting_to_RGB
    def hsv2rgb(hsv):
        h, s, v = hsv
        # compute chroma
        c = v*s
        h_prime = h*6.0
        x = c*( 1 - abs(h_prime %2 - 1) )
        if h_prime >= 5: rgb = (c,0,x)
        elif h_prime >= 4: rgb = (x,0,c)
        elif h_prime >= 3: rgb = (0,x,c)
        elif h_prime >= 2: rgb = (0,c,x)
        elif h_prime >= 1: rgb = (x,c,0)
        else: rgb = (c,x,0)
        m = v-c
        return tuple( int(255*(val+m)) for val in rgb )

    # random float in [0.0, 1.0)
    hue = random.random()
    l_hues = [hue]

    for i in xrange(n-1):
        # generate evenly distributed hues by random walk using the golden ratio!
        # (mod 1, to stay within hue space)
        hue += GOLDEN_RATIO_INVERSE
        hue %= 1
        l_hues.append(hue)

    if not RGB:
        return [ (h, SATURATION, VALUE) for h in l_hues ]

    return [ hsv2rgb((h, SATURATION, VALUE)) for h in l_hues ]


def implement(infile, k, warholize=False):
    x = kmeans(infile, k=k)
    x.assign_pixels_map(metric=euclidean_dist_np)
    x.generate_image_2(warholize=warholize)


FILE_IN = '/Users/miriamshiffman/Desktop/Pics/Art/sc236393.jpg'
K=40


if __name__ == "__main__":
    implement(FILE_IN, K)
