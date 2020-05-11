"""
In here is a general module to build what ever kinf of layers we want
"""
from project_defs import *
import tensorflow as tf



class ModularLayer(tf.keras.layers.Layer):
    def __init__(self, _name,
                 distance_in_pixels,
                 shape,
                 wavelength_in_pixels,
                 nm,
                 amp_modulation = False,
                 phase_modulation = False):

        super(ModularLayer, self).__init__()
        self._name = _name
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

        self.checkParams()
        self.buildLayer()

    def call(self, inputs):
        x_r = tf.reshape(tf.cast(tf.math.real(inputs), tf.float32), [-1, self.units])
        x_i = tf.reshape(tf.cast(tf.math.imag(inputs), tf.float32), [-1, self.units])

        y_r = (self.amp_w) * (x_r * tf.cast(tf.cos(self.phase_w), tf.float32) + x_i * tf.cast(
            tf.sin(self.phase_w), tf.float32))
        y_i = (self.amp_w) * (x_i * tf.cast(tf.cos(self.phase_w), tf.float32) - x_r * tf.cast(
            tf.sin(self.phase_w), tf.float32))

        res = tf.complex(y_r, y_i)
        res = tf.reshape(res, [-1, self.shape[0], self.shape[1]])
        res = project_prop.my_fft_prop(res, self.distance_in_pixels, self.nm, self.wave_len)
        return tf.cast(res, tf.complex128)

    def checkParams(self):
        assert (len(self.shape) == 2)

        if (self.amp_mod == False and self.phase_mod == False):
            print("ERROR: in layer :{} - amp mdulation && phase modulation cannot be False "
                  "both".format(self.name))
            exit(FAILURE)

    def buildLayer(self):
        if self.amp_units or self.units:
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