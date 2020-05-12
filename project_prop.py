from project_defs import *
import numpy as np
PI = np.pi

@tf.function
def pad_add(av,size=None,stlen=10):
    if size is None:
        size = list()
        for s in av.shape[1:]:
            size.append(int(2*s))
    elif not hasattr(size, "__len__"):
        size = [size]


    assert len(av.shape) in [1, 2, 3], "Only 1D and 2D arrays!"
    assert len(av.shape[1:]) == len(
        size), "`size` must have same length as `av.shape`!"


    if len(av.shape) in [2,3]:
        return _pad_add_2d(av, size, stlen)
    else:
        return None

    return


def _get_pad_left_right(small, large):
    """ Compute left and right padding values.

    Here we use the convention that if the padding
    size is odd, we pad the odd part to the right
    and the even part to the left.

    Parameters
    ----------
    small : int
        Old size of original 1D array
    large : int
        New size off padded 1D array

    Returns
    -------
    (padleft, padright) : tuple
        The proposed padding sizes.
    """
    assert small < large, "Can only pad when new size larger than old size"

    padsize = large - small
    if padsize % 2 != 0:
        leftpad = (padsize - 1)/2
    else:
        leftpad = padsize/2
    rightpad = padsize-leftpad

    return [int(leftpad),int(rightpad)]
    # return int(leftpad), int(rightpad)


@tf.function
def _pad_add_2d(av, size, stlen):
    """ 2D component of `pad_add`
    """
    assert len(size)  in [2, 3]

    padx = _get_pad_left_right(av.shape[1], size[0])
    pady = _get_pad_left_right(av.shape[2], size[1])

    mask = np.zeros((av.shape[1],av.shape[2]), dtype=bool)
    mask[stlen:-stlen, stlen:-stlen] = True
    if(av.shape[0] is None):
        tensor_mask = tf.constant(np.array([~mask]))
    else:
        tensor_mask = tf.constant(np.array([~mask for i in range(av.shape[0])]))

    border  = tf.where(tensor_mask,av,0)
    phase = tf.math.atan2(tf.math.imag(border),tf.math.real(border))

    if av.dtype.name.count("complex"):
        padval = tf.cast(tf.reduce_mean(tf.abs(border)), tf.complex128)* tf.exp(1j*tf.cast(tf.reduce_mean(phase),tf.complex128))
    else:
        padval = tf.reduce_mean(border)

    end_values = padval
    ## this part uses a value different than  0
    # bv = tf.pad(av,paddings=[[0,0],padx,pady],
    #             mode="CONSTANT",
    #             constant_values=end_values)
    #
    ## this verison pads with 0
    bv = tf.pad(av,paddings=[[0,0],padx,pady],
                mode="CONSTANT",
                constant_values=0)


    # bv = np.pad(av,
    #             (padx, pady),
    #             mode="linear_ramp",
    #             end_values=end_values)


    # roll the array so that the padding values are on the right
    # bv = np.roll(bv[0], -padx[0], 0)
    # bv = np.roll(bv, -pady[0], 1)
    # bv = tf.roll(bv,-padx[0],1)
    # bv = tf.roll(bv,-pady[0],2)

    return bv


def pad_rem(pv, size=None):
    """ Removes linear padding from array

    This is a convenience function that does the opposite
    of `pad_add`.

    Parameters
    ----------
    pv : 1D or 2D ndarray
        The array from which the padding will be removed.
    size : tuple of length 1 (1D) or 2 (2D), optional
        The final size of the un-padded array. Defaults to half the size
        of the input array.


    Returns
    -------
    pv : 1D or 2D ndarray
        Padded array `av` with pads appended to right and bottom.
    """
    if size is None:
        size = list()
        for s in pv.shape[1:]:
            assert s % 2 == 0, "Uneven size; specify correct size of output!"
            size.append(int(s/2))
    elif not hasattr(size, "__len__"):
        size = [size]
    assert len(pv.shape) in [1, 2, 3], "Only 1D and 2D arrays!"
    assert len(pv.shape[1:]) == len(
        size), "`size` must have same length as `av.shape`!"

    x_margin = (pv.shape[1] - size[0])//2
    y_margin = (pv.shape[2] - size[1])//2


    if len(pv.shape) in [2, 3]:
        # return pv[:,size[0]:pv.shape[1] - size[0], size[1]:pv.shape[2] - size[1]]
        return pv[:,
               x_margin:pv.shape[1] - x_margin,
               y_margin : pv.shape[2] - y_margin
               ]

    else:
        return pv[:,size[0]]

@tf.function
def my_fftfreq(n,d=1):

    val = tf.cast(1.0 / (n*d),tf.complex128)
    result = [i for i in range(0,n)]
    N = (n-1)//2 + 1

    intervale1 = [i for i in range(0,N)]
    result[:N] = intervale1

    intervale2 = [i for i in range(-n//2 , 0)]
    result[N:] = intervale2
    result = tf.cast(result,tf.complex128)

    return tf.multiply(result,val)

@tf.function
def my_fft_prop(field,
                d = 75,
                nm = 1.0,
                res = 1.3,
                method = 'helmholtz',
                ret_fft = None,
                padding = True):

    field = tf.cast(field,tf.complex128)
    if padding:
        field = pad_add(field)

    fft_field = tf.cast(tf.signal.fft2d(field),tf.complex128)      # first convert to fft
    print(field)

    km = (2*np.pi*nm)/res

    kx = tf.reshape((my_fftfreq(fft_field.shape[1])*2*PI),[-1,1])
    ky = tf.reshape((my_fftfreq(fft_field.shape[2])*2*PI),[1,-1])

    root_km = km**2 - kx**2 - ky**2

    fstemp = tf.cast(tf.exp(1j * (tf.sqrt(root_km) - km ) * d),tf.complex128)
    fft_field = tf.cast(fft_field,tf.complex128)

    pre_ifft = tf.cast(tf.multiply(fstemp, fft_field),tf.complex128)
    result = tf.cast(tf.signal.ifft2d(pre_ifft),tf.complex128)
    if padding:
        result = pad_rem(result)
    print("post prop shape :",result.shape)
    return result



D = [0,1,10,30,50,90,500,750,1000]
COMPARE = 0
TEST_FUNC_TENSOR = 0
if COMPARE:

    import matplotlib.pyplot as plt

    DB_PATH = '/cs/usr/avshalomnaor/Documents/Project/DBs/'

    MNIST_DIGIT_TRAIN_IMGS_PATH = DB_PATH + 'train-images.idx3-ubyte'

    num_input = idx2numpy.convert_from_file(MNIST_DIGIT_TRAIN_IMGS_PATH)[0]/255
    ten_input = tf.reshape(tf.convert_to_tensor(num_input),(1,28,28))

    rect_input_no_pad = np.ones((28,28))
    ten_rect_input_no_pad = tf.reshape(tf.convert_to_tensor(rect_input_no_pad),(1,28,28))

    rect_input = np.zeros((28,28))
    rect_input[10:18,10:18] = 1
    ten_rect_inputs = tf.reshape(tf.convert_to_tensor(rect_input),(1,28,28))


    import nrefocus


    for d in D:
        tensor_out = my_fft_prop(ten_rect_inputs,d)
        out = nrefocus.refocus(rect_input,d,1,1)

        plt.subplot(311)
        plt.imshow(tf.abs(tensor_out[0])**2)
        plt.title("my func")

        plt.subplot(312)
        plt.imshow(np.abs(out**2))
        plt.title("NREFOCUS")

        plt.subplot(313)
        distance = tf.abs(tf.abs(tensor_out[0])**2 - np.abs(out)**2)
        max = np.max(distance)
        mean = np.mean(distance)
        plt.imshow(distance)
        plt.title("Distance, max = {},mean err = {}".format(max,mean))
        plt.show()

if TEST_FUNC_TENSOR:
    inputs = idx2numpy.convert_from_file(MNIST_DIGIT_TRAIN_IMGS_PATH)[3:6]/255
    inputs = tf.convert_to_tensor(inputs)
    for d in D:
        ten_out = tf.abs(my_fft_prop(inputs,d))
        out     = [nrefocus.refocus(inputs[0],d,1,1),nrefocus.refocus(inputs[1],d,1,1),nrefocus.refocus(inputs[2],d,1,1)]

        for i in range(ten_out.shape[0]):

            plt.subplot(311)
            plt.imshow((ten_out[i]) ** 2)
            plt.title("my func")

            plt.subplot(312)
            plt.imshow(np.abs(out[i] ** 2))
            plt.title("NREFOCUS")

            plt.subplot(313)
            distance = tf.abs((ten_out[i]) ** 2 - np.abs(out[i]) ** 2)
            max = np.max(distance)
            mean = np.mean(distance)
            plt.imshow(distance)
            plt.title("Distance, max = {},mean err = {}".format(max, mean))
            plt.show()
## TODO: check that function also works for tensor of 8 items

