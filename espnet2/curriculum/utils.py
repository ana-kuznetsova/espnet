import numpy as np

def str2numpy(string):
    string = string.replace('[', '').replace(']', '').split()
    arr = np.array([float(i) for i in string])
    return arr