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
from bs4 import BeautifulSoup

###############################################################################
###############################################################################
###############################################################################
###############################################################################

class glitch():
    """Glitchable object class"""

    MAX_CHUNK = 300

    def __init__(self, path, from_file=True):
        """Given filepath to image, initialize glitchable object"""
        if from_file:
            self.data = self.read_from_file(path)
            self.name = subprocess.check_output("basename {}".format(path), shell=True).strip()
            # set name of file with call to `basename`

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
        """Return random chunk of image data and tuple of start/endpoints, with optional parameter for max size -
        i.e. [ str, (int, int) ]"""
        splice_origin = random.randint(0, len(self.data))
        splice_end = random.randint(splice_origin, splice_origin + max_chunk)
        return [self.data[splice_origin : splice_end], (splice_origin, splice_end)]


    def genome_rearrange(self, max_n=5, max_chunk = MAX_CHUNK):
        """Glitch according to rules of basic genome rearrangement: splice chunk, n times, into another location"""
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
        
        self.data = self.data[:a] + ''.join(str( (int(x)+n) % 10 ) if x.isdigit() else x for x in chunk) + self.data[b:]

        self.change_log.append('Digits in chunk of {} char ({} to {}) incremented by {}'\
                            .format(b - a, a, b, n))

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def outfile_path(path_to_dir, filename):
    """Given directory and desired filename, returns path to outfile that will not
    overwrite existing file/s by appending '_<i>' to filename"""
    path_to_dir = re.sub('/$', '', path_to_dir) # remove trailing slash, if exists
    if glob.glob('{}/{}'.format(path_to_dir, filename)):
        # only alter filename if necessary to avoid clobbering preexisting file
        i = 1
        filename = re.sub('(\.[^\.]*)$', r'_{}\1'.format(i), filename)  # add '_<i>' before file suffix
        while glob.glob('{}/{}'.format(path_to_dir, filename)): # continually check for pre-existing file
            filename = re.sub('_[0-9]*(\.[^\.]*)$', r'_{}\1'.format(i), filename) # update i in filename
            # N.B. r for raw string to enable back referencing
            i += 1
    return '{}/{}'.format(path_to_dir, filename)

###############################################################################
###############################################################################
###############################################################################
###############################################################################

img_in = '/Users/miriamshiffman/Desktop/Pics/Mir/IMG_0869.jpg'
path_out = '/Users/miriamshiffman/Downloads/'


if __name__ == '__main__':
    x = glitch(img_in)
    for i in xrange(5):
        x.digit_increment(max_chunk = 1000, max_n=4)
    x.write_to_file(path_out)
