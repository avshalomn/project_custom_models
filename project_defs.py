import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import idx2numpy
import math
import json

"""
project imports
"""
import project_prop
import project_utils
import project_layers
import matplotlib.pyplot as plt
import json

"""
Consts
"""
PI = np.pi
FAILURE = 1
LAYER_PHASE_MAX = 2*PI
LAYER_PHASE_MIN = 0.0
LAYER_AMP_MAX = 1
LAYER_AMP_MIN = 0


"""
Header
"""
header = "\n"\
"|**************************************************************|\n" \
"|____________Diffractive Deep Neural Network Trainer___________|\n" \
"|* Model Name:                {}                               \n" \
"|* Number of Layers:          {}                               \n" \
"|* Distance between layers:   {}                               \n" \
"|* Wavelength in pixles:      {}                               \n" \
"|* Refractive Index           {}                               \n" \
"|                                                              \n" \
"|                                                              \n" \
"|                                                              \n" \
"|                                                              \n" \
"|**************************************************************|\n" \

