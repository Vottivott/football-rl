from os import listdir
import os
from os.path import isfile, join


def clear_folder(path): #including last '/'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for fname in files:
        fpath = join(path, fname)
        os.remove(fpath)