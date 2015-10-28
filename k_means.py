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

from PIL import Image
import random
from itertools import chain, izip

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
        self.pixelMap = self.img.load()
        #TODO: figure out how to sample from generator/iterable
        self.pixels_xy = [ (x,y) for x in xrange(self.img.size[0]) for y in xrange(self.img.size[1]) ]
        self.__initialize_k_dict__( random.sample( self.pixels_xy, self.k ) )

    def __initialize_k_dict__(self, k_vals):
        """Generate dictionary of k clusters based on list of (x,y) tuples for k means"""
        # Initialize k clusters with pixels in group, starting with points representing k clusters themselves
        self.d_k_clusters = { self.xy2xyrgb(t_xy): [ self.xy2xyrgb(t_xy) ] for t_xy in k_vals }

    def minimize_distance(self, pixel, metric):
        """Given tuple representing pixel in image, return k group that minimizes distance by given metric"""
        dists = [ (k, metric(pixel, k)) for k in self.d_k_clusters.iterkeys() ]
        # find tuple representing best k group by minimizing distance
        best_k, _ = sorted( dists, key = lambda t: t[1] )[0]
        return best_k

    def xy2xyrgb(self, t_xy):
        """Given (x,y) of pixel, returns ((x,y), (R, G, B))"""
        return ( t_xy, self.pixelMap[ t_xy[0], t_xy[1] ] )

    def assign_pixels(self, metric):
        """Assign all pixels in image to closest matching group in self.d_k_groups, according to given distance metric"""
        print 'assigning pixels'
        for t in self.pixels_xy:
            # convert (x, y) of pixel location to ((x, y), (r, g, b))
            tval = self.xy2xyrgb(t)
            # append to dictionary value list corresponding to key of k-mean that minimizes distance by given metric
            self.d_k_clusters[ self.minimize_distance( tval, metric ) ].append(tval)

    def generate_image(self):
        self.new_img = Image.new('RGB', self.img.size, "black")
        # create pixel map
        pixels = self.new_img.load()

        def mean_rgb(k):
            """Given key value in self.d_k_clusters, return k mean by averaging (r,g,b) value over all values in group"""
            vals = self.d_k_clusters[k]
            # in order to sum tuple values, have to zip all tuples via splatted generator of (r,g,b) vals
            summed_rgb = tuple( sum(rgb) for rgb in izip( *( v[1] for v in vals )) )
            return tuple( int(rgb / len(vals)) for rgb in summed_rgb )

        for k, v_list in self.d_k_clusters.iteritems():
            print 'putting pixels'
            # find mean R,G,B value of cluster
            k_mean = mean_rgb(k)
            for t_xy, _ in v_list:
                print t_xy
                pixels[t_xy[0], t_xy[1]] = k_mean

        self.new_img.show()


###############################################################################
###############################################################################
###############################################################################
###############################################################################

def euclidean_dist(p1, p2):
    """Compute Euclidean distance between 2 pts of any (equal) dimensions
    IN: two iterables (tuples, lists)
    OUT: float"""
    return sum( abs(x1-x2) for x1, x2 in zip(chain.from_iterable(p1), chain.from_iterable(p2)) )**0.5

def implement(infile):
    x = kmeans(infile)
    x.assign_pixels(metric=euclidean_dist)
    x.generate_image()
