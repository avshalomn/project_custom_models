"""
In here is a general module to build what ever kinf of layers we want
"""
from project_defs import *
import tensorflow as tf


PI = np.pi
FAILURE = 1
LAYER_PHASE_MAX = 2*PI
LAYER_PHASE_MIN = 0.0
LAYER_AMP_MAX = 1
LAYER_AMP_MIN = 0

class ModularLayer(tf.keras.layers.Layer):
    def __init__(self, layer_name,
                 distance_in_pixels,
                 shape,
                 wavelength_in_pixels,
                 nm,
                 amp_modulation = False,
                 phase_modulation = False):
        """
        this is a modular layer. we can choose the distnace to next layer to propegate,
        the refractive index of the medium (1 = air), shape of the layer, and wether or not we
        want amplitude modulation or phase modulation. (or a combination of both)

        :param layer_name:
        :param distance_in_pixels: distance in pixels to next layer
        :param shape: shape of layer
        :param wavelength_in_pixels: wavelength in pixles
        :param nm:  refractive index of medium (air == 1)
        :param amp_modulation: bool
        :param phase_modulation: bool
        """

        super(ModularLayer, self).__init__()
        self.layer_name = layer_name
        self.shape = shape
        self.distance_in_pixels = distance_in_pixels
        self.shape = shape
        self.wave_len = wavelength_in_pixels
        self.nm = nm
        self.units = 0
        self.amp_mod = amp_modulation
        self.phase_mod = phase_modulation
        self.phase_w = None
        self.amp_w = None
        self.js_matrix = None

        # update self parameters
        self.checkParams()

        # build layer
        self.buildLayer()


    def call(self, inputs):
        """
        this is an override function of the Layer.call() module.
        :param inputs:
        :return:
        """
        # break input to real and imaginary
        x_r = tf.reshape(tf.cast(tf.math.real(inputs), tf.float32), [-1, self.units])
        x_i = tf.reshape(tf.cast(tf.math.imag(inputs), tf.float32), [-1, self.units])


        y_r = (self.amp_w) * (x_r * tf.cast(tf.cos(self.phase_w), tf.float32) + x_i * tf.cast(
            tf.sin(self.phase_w), tf.float32))
        y_i = (self.amp_w) * (x_i * tf.cast(tf.cos(self.phase_w), tf.float32) - x_r * tf.cast(
            tf.sin(self.phase_w), tf.float32))

        # print("yr shape",y_r.shape)
        res = tf.complex(y_r, y_i)
        res = tf.reshape(res, [-1, self.shape[0], self.shape[1]])
        res = project_prop.my_fft_prop(field = res, d = self.distance_in_pixels, nm = self.nm,
                                       res = self.wave_len, method = "helmholtz", ret_fft=None,
                                       padding = True)
        return tf.cast(res, tf.complex128)

    def checkParams(self):
        assert (len(self.shape) == 2)

        if (self.amp_mod == False and self.phase_mod == False):
            print("ERROR: in layer :{} - amp mdulation && phase modulation cannot be False "
                  "both".format(self.name))
            exit(FAILURE)

    def buildLayer(self):
        if self.amp_mod or self.phase_mod:
            self.units = self.shape[0] * self.shape[1]

        self.js_matrix = tf.constant(np.zeros(shape=(self.units,)) + 1j)

        if self.phase_mod:
            self.phase_w = tf.Variable(initial_value=tf.random.uniform(shape=(self.units,),
                                                                       minval=LAYER_PHASE_MIN,
                                                                       maxval=LAYER_PHASE_MAX),
                                                                       trainable=True)

        if self.amp_mod:
            self.amp_w = tf.Variable(initial_value=tf.random.uniform(shape=(self.units,),
                                                                     minval=LAYER_AMP_MIN,
                                                                     maxval=LAYER_AMP_MAX),
                                                                     trainable=True)
        # in case we do not want Amplitude modulation, we assume "see thorugh" layer in terms of
        # Amplitude
        else:
            self.amp_w = tf.Variable(initial_value=tf.ones(shape = (self.units,)))


    def get_config(self):
        config = super(ModularLayer, self).get_config()
        # config.update({'amp_weights':self.amp_w})
        # config.update({'phase_weights':self.phase_w})
        return config

###################
### prop layer ####
###################

class PropLayer(tf.keras.layers.Layer):

    def __init__(self, layer_name,
                 distance_in_pixels,
                 shape,
                 wavelength_in_pixels,
                 nm):
        """
        this is a simple Proppagation layer - no variables to train.
        :param layer_name:
        :param distance_in_pixels:
        :param shape:
        :param wavelength_in_pixels:
        :param nm:
        """
        super(PropLayer, self).__init__()

        self.layer_name = layer_name
        self.shape = shape
        self.distance_in_pixels = distance_in_pixels
        self.shape = shape
        self.wave_len = wavelength_in_pixels
        self.nm = nm
        self.units = self.shape[0] * self.shape[1]
        self.js_matrix = tf.constant(np.zeros(shape=(self.units,)) + 1j)


    def call(self, inputs):
        res = project_prop.my_fft_prop(field = inputs, d = self.distance_in_pixels, nm = self.nm,
                                       res = self.wave_len, method = "helmholtz", ret_fft=None,
                                       padding = True)
        return tf.cast(res, tf.complex128)


    def get_config(self):
        config = super(PropLayer, self).get_config()
        return config


################
# Output Layer #
################

@tf.function
def output_func(inp):
    assert (inp.dtype == tf.complex128)
    real = tf.math.real(inp)
    imag = tf.math.imag(inp)
    amp = tf.sqrt(real**2 + imag**2)
    amp = amp / (tf.reduce_mean(amp) + 1e-14)
    return amp

output_layer = tf.keras.layers.Lambda(output_func)