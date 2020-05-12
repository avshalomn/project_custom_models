# project_custom_models

This project target is to supply a (base) module that can train a Neural Net - mimicing a physical 
diffractive 3D network

## How to use
Currently, the module is capable of dealing the MNIST-digits dataset.
The flow is as follows:
* create a model using the *CustomModel* class in *project_model* with the proper parameters.
* call *model.trainModel()*
* call *model.testModel with the right parameters

## Notes
This is still a work in progress - any input is more than welcome!

## Example 
```python
!pip install -q pyyaml h5py  # Required to save models in HDF5 format

from google.colab import drive
try:
  %tensorflow_version 2.x
except Exception:
  pass



drive.mount ("/content/drive",force_remount=True)
!pip install idx2numpy


import sys,os
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import idx2numpy

LOCAL_PATH = "/content/drive/My Drive/Colab Notebooks/project_custom_models/src"
MNIST_DB = "/content/drive/My Drive/FourthYearProject/Latest_20_2_20/DB/"
MNIST_TRAIN = MNIST_DB +"train-images.idx3-ubyte"
MNIST_2D_LABLES = MNIST_DB + "TRAIN_TARGET_LABELS_28x28_FULL.idx3-ubyte"
MNIST_TRAIN_LABLES = MNIST_DB + "train-labels.idx1-ubyte"
MNIST_HALF_LABLES = MNIST_DB  + "LABLES_HALF.idx3-ubyte"
MNIST_28x28_LABLES = MNIST_DB + "28X28_LABLES_FIXED.idx3-ubyte"

if LOCAL_PATH not in list(sys.path):
    sys.path.append(LOCAL_PATH)

from project_defs import *
from project_layers import *
from project_prop import *
from project_utils import *
from project_model import *

#########################
##### GLOBAL CONSTS #####
#########################

PI = np.pi




#############################
###### Consts ###############
#############################
MODEL_NAME              = "MNIST_amp&phase&z=75&#layers=3&wavelen=1.3"
INPUTS_PATH             = MNIST_2D_LABLES
TARGETS_PATH            = MNIST_28x28_LABLES
NUM_OF_SAMPLES          = 60000
NUM_OF_LAYERS           = 3
LAYERS_DISTANCE         = 75.0
WAVELENGTH_IN_PIXELS    = 1.3
NM                      = 1.0
LEARNING_RATE           = 0.001
BATCH_SIZE              = 8
EPOCHS                  = 1
DIR_TO_SAVE             = "/content/drive/My Drive/Colab Notebooks/project_custom_models/models"
NAME_TO_SAVE            = MODEL_NAME
WEIGHTS_NAME            = MODEL_NAME+"weights"
PROP_INPUT              = True
RESCALING_FACTOR        = 7
PADDING_FACTOR          = 0
AMP_MODULATION          = True
PHASE_MODULATION        = True


### Create Model ###
model = CustomModel(
    model_name = MODEL_NAME,
    inputs_path = INPUTS_PATH,
    targets_path = TARGETS_PATH,
    num_of_samples = NUM_OF_SAMPLES,
    num_of_layers  = NUM_OF_LAYERS,
    layers_distance_pxls = LAYERS_DISTANCE,
    wavelen_in_pxls = WAVELENGTH_IN_PIXELS,
    nm = NM,
    learning_rate = LEARNING_RATE,
    batch_size   = BATCH_SIZE,
    epochs = EPOCHS,
    dir_to_save = DIR_TO_SAVE,
    name_to_save = NAME_TO_SAVE,
    weights_name = WEIGHTS_NAME,
    prop_input   = PROP_INPUT,
    rescaling_factor = RESCALING_FACTOR,
    padding_factor   = PADDING_FACTOR,
    amp_modulation   = AMP_MODULATION,
    phase_modulation = PHASE_MODULATION
)



### Train Model ###
# model.trainModel()

### Test Model ###
LOW_IDX = 0
HIGH_IDX = 60000
NUM_OF_SAMPLES_TO_TEST = 100
NUMERIC_TARGETS = None

# model.testModel()

```