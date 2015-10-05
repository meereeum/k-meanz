#!/usr/bin/env python

###############################################################################
#
# __glitch_random__.py - Given image as input, glitch at random by selecting
# a random chunk of the file and inserting in a random number of times in
# another place!
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
__email__ = "meereeum@gmail.com"
__status__ = "Development"

###############################################################################

import random
import subprocess
import re
import glob
import code
import webbrowser
import requests
#from bs4 import BeautifulSoup

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class glitch():
    """Glitchable object class"""

    MAX_CHUNK = 300

    def __init__(self, path):
        """Given filepath to image, initialize glitchable object"""
        self.data = self.read_from_file(path)
        # set name of file with call to `basename`
        self.name = subprocess.check_output("basename {}".format(path), shell=True).strip()
        self.change_log = []
        self.glitchname = re.sub('(\.[^\.]*)$', r'.glitched\1', self.name) # add '.glitched' before file suffix


    def __repr__(self):
        """Print image name and description of changes (glitches, any output to file)"""
        return '\n*~~~~~*~********~*********~***********~~~~~******\n' + \
            'Hey {}... get glitchy with it'.format(self.name) + \
            '\n*~~~~~*~********~*********~***********~~~~~******\n' + \
            '\n'.join(self.change_log) + '\n'


    def write_to_file(self, outdir, launch_in_browser=True):
        """Write glitch art to file"""
        outfile = outfile_path(outdir, self.glitchname)

        with open(outfile, "w") as file_out:
            file_out.write(self.data)
            self.change_log.append('>>>>>>>>>>>>> {}'.format(file_out.name))
        print self

        # After printing to file, remove this line from changelog to reset for future files
        self.change_log = self.change_log[:-1]

        if launch_in_browser:
            webbrowser.get("open -a /Applications/Google\ Chrome.app %s")\
                .open('file://localhost{}'.format(outfile), new=2) # open in new tab, if possible


    def read_from_file(self, path_to_file):
        """Read in image data from local file"""
        with open(path_to_file, "r") as file_in:
            return file_in.read()


    def _random_chunk(self, max_chunk = MAX_CHUNK):
        """Return random chunk of image data and tuple of start/endpoints,
        with optional parameter for max size - output: [ str, (int, int) ]"""
        splice_origin = random.randint(0, len(self.data))
        splice_end = random.randint(splice_origin, splice_origin + max_chunk)
        return [self.data[splice_origin : splice_end], (splice_origin, splice_end)]


    def genome_rearrange(self, max_n=5, max_chunk = MAX_CHUNK):
        """Glitch according to rules of basic genome rearrangement:
        splice chunk, n times, into another location"""
        n = random.randint(1, max_n)
        site = random.randint(0, len(self.data))
        [chunk, (a, b)] = self._random_chunk(max_chunk)

        self.data = self.data[:site] + chunk*n + self.data[site:]

        # update change log
        self.change_log.append('Chunk of {} char ({} to {}) moved to {}, x{}'\
                            .format(b - a, a, b, site, n))


    def digit_increment(self, max_chunk = MAX_CHUNK, max_n=1):
        """Glitch by incrementing all digits in random data chunk by n (mod 10)"""
        n = random.randint(1, max_n)
        [chunk, (a, b)] = self._random_chunk(max_chunk)

        self.data = self.data[:a] + \
                    ''.join( str( (int(x)+n) % 10 ) if x.isdigit() else x for x in chunk ) + \
                    self.data[b:]

        self.change_log.append('Digits in chunk of {} char ({} to {}) incremented by {}'\
                            .format(b - a, a, b, n))


###############################################################################
###############################################################################
###############################################################################
###############################################################################


def flickr_hit(keywd):
    """Use keyword to download random image hit from Flickr API and return local path to image"""
    API_KEY='6fb8c9ee707ff3eb8c610d4bfba9ddaf'
    # get random images matching given keyword using Flickr's API
    curled = subprocess.check_output("curl 'https://api.flickr.com/services/rest/?method=flickr.photos.search&api_key={}&text={}&format=json&nojsoncallback=1'".format(API_KEY, keywd), shell = True)
    # select random hit from among top hits
    random_hit = random.choice( [ img.split(",") for img in curled.split('{') ][3:] )
    # use hit data to initialize dictionary of image data and construct image URL
    d_hit = dict( tuple( txt.strip('"') for txt in elem.split(':')[:2] ) for elem in random_hit if len(elem)>0 )
    hit_url = 'https://farm{}.staticflickr.com/{}/{}_{}.jpg'\
              .format( *( d_hit[key] for key in ['farm','server','id','secret'] ) )
    # download image
    outfile = outfile_path( PATH_OUT, '{}.jpg'.format(d_hit['id']) )
    with open(outfile, "w") as file_out:
        file_out.write( requests.get(hit_url).content )
    return outfile


def outfile_path(path_to_dir, filename):
    """Given directory and desired filename, returns path to outfile that will not
    overwrite existing file/s by appending '_<i>' to filename"""
    # remove trailing slash, if exists
    path_to_dir = re.sub('/$', '', path_to_dir)

    # only alter filename if necessary to avoid clobbering preexisting file
    if glob.glob('{}/{}'.format(path_to_dir, filename)):
        i = 1
        # add '_<i>' before file suffix
        filename = re.sub('(\.[^\.]*)$', r'_{}\1'.format(i), filename)
        # continually check for pre-existing file
        while glob.glob('{}/{}'.format(path_to_dir, filename)):
            # update i in filename
            filename = re.sub('_[0-9]*(\.[^\.]*)$', r'_{}\1'.format(i), filename)
            # ^^ 'r' for raw string to enable back-referencing
            i += 1

    return '{}/{}'.format(path_to_dir, filename)


###############################################################################
###############################################################################
###############################################################################
###############################################################################

KEY = 'australia'
#IMG_IN = '/Users/miriamshiffman/Desktop/Pics/Mir/IMG_0869.jpg'
PATH_OUT = '/Users/miriamshiffman/Downloads/'
IMG_IN = flickr_hit(KEY)


if __name__ == '__main__':
    x = glitch(IMG_IN)
    for i in xrange(3):
        x.digit_increment(max_chunk = 1000, max_n=4)
        x.genome_rearrange()
    x.write_to_file(PATH_OUT)
