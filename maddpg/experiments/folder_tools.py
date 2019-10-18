from os import listdir
import os
from os.path import isfile, join


def clear_folder(path): #including last '/'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for fname in files:
        fpath = join(path, fname)
        os.remove(fpath)

def rename_single_file_in_folder(folder, newname):
    files = os.listdir(folder)
    os.rename(folder + "/" + files[0], folder + "/" + newname)

def read_name_of_single_file_in_folder(folder):
    files = os.listdir(folder)
    return files[0]

if __name__=="__main__":
    print(read_name_of_single_file_in_folder("../../current_episode_num"))
    rename_single_file_in_folder("../../current_episode_num", "hejhej")
    print(read_name_of_single_file_in_folder("../../current_episode_num"))
