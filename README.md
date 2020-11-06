# Deep Defractive Neural Networks - Custom Simulations

This project target is to supply a (base) module that can train a Neural Net - mimicking a physical 
diffracting 3D Neural Network.

## How to use
The flow is as follows:
* create a model using the *CustomModel* class in *project_model* with the proper parameters.
* call *model.trainModel()*
* call *model.testModel with the right parameters

## Notes

* Any input is more than welcome!
* there are 2 ways to train a NN: 
    * with a pre-exisitng and pre-processed inputs and targets:
        * this means that inputs are converted to greyscale and only have 1 layer of color depth.
        * targets (integer labeles - 0, 1, ...) are mapped to 2D image where the desired label is
         a light up square and the rest of the 2D image is 0's.
    * in trainModel() feed a data set name (one of TensorFlows data-base) - and the NN will
     convert a batch of the data every time to fit the system - mid training. 
     
Generaly - the first option is much faster - since converting each batch mid flight takes
 too long.
 
At the end of the training phase, the model and the weights would be saved at DIR_TO_SAVE.

After the testing phase, a confusion matrix will pop, showing biases and the like.


## TODO
This system could always use more work - both from the design and flow of the module (SW) and also 
the engineering (Optical) part of it.

## Constructing the model 
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
MODEL_NAME              = "newMnist"
INPUTS_PATH             = MNIST_TRAIN
TARGETS_PATH            = MNIST_28x28_LABLES
NUM_OF_SAMPLES          = 60000
NUM_OF_LAYERS           = 4
LAYERS_DISTANCE         = 100
WAVELENGTH_IN_PIXELS    = 0.85
NM                      = 1.0
LEARNING_RATE           = 0.0001
BATCH_SIZE              = 4
EPOCHS                  = 10
DIR_TO_SAVE             = "/content/drive/My Drive/Colab Notebooks/project_custom_models/models"
NAME_TO_SAVE            = MODEL_NAME
WEIGHTS_NAME            = MODEL_NAME+"weights"
PROP_INPUT              = True
RESCALING_FACTOR        = 7
PADDING_FACTOR          = 0
AMP_MODULATION          = True
PHASE_MODULATION        = True
TEST_INPUTS_PATH        = MNIST_TRAIN
TEST_LABELS_PATH        = MNIST_TRAIN_LABLES

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
    phase_modulation = PHASE_MODULATION,
    test_inputs_path = TEST_INPUTS_PATH,
    test_lables_path = TEST_LABELS_PATH,
    force_shape = 28
)



### Train Model ###
# model.trainModel()

### Test Model ###
LOW_IDX = 0
HIGH_IDX = 60000
NUM_OF_SAMPLES_TO_TEST = 100
NUMERIC_TARGETS = None
```

## Training the model:
One can either train the model on data on local machine, or use any of the TF datasets.
In order to train on local data set - just to:

```python
model.trainModel()
```

To use TF data set package:
```python
model.trainModel(
    local_training_set = False, 
    tf_dataset_name = 'cifar10'
    )
```

## Loading a trained model:
```python
!pip install -q pyyaml h5py  # Required to save models in HDF5 format
model.model.build((None, 224, 224)) # give the shape of the layer
model.model.load_weights("/content/drive/My Drive/Colab Notebooks/project_custom_models/models/newCifar/")
```

## Testing the Model:
```python
model.shape = (196,196) # see internal function docu for args 
model.testModel(0, 60000, 500, TEST_LABELS_PATH,  True, None, TEST_INPUTS_PATH,
                monte_carlo = False, monte_carlo_variance = 0.05, 
                input_shift = True, input_shift_percentrage = 0.1)
```