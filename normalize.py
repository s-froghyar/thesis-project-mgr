import os
import subprocess
from shutil import rmtree
from distutils.dir_util import copy_tree
from os.path import join


class Normaliser():
    def __init__(self, _path):
        self._path = _path

    def normalise(self):
        """
        - Normalises wav files found in _path/
        - Moves normalized folder to wav/normalized
        Input: path to wav files
        """
        subprocess.call(
            'ffmpeg-normalize {}*.wav -c:a pcm_s16le -ext wav'.format(
                self._path,
            ),
            shell=True
        )
        normalised_path = join(self._path, 'normalised/')
        copy_tree('normalized/', normalised_path)
        rmtree('normalized/')

        # .DS_Store messes with the stats
        _ds_store = join(normalised_path, '.DS_Store')
        if os.path.exists(_ds_store):
            os.remove(_ds_store)

        # Handling of occasional hidden files
        number_of_files = len(
            [f for f in os.listdir(normalised_path) if not f.startswith('.')])
        print('Normalised {} files. Found at {}'.format(
            number_of_files, normalised_path))