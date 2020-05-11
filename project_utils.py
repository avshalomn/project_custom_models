from project_defs import *

def compare_imgs(label, output, threshold):
    """
    :param label: the "real" data we expect
    :param output: the output of the system
    :threshold: how much of the area is 1?
    :return: True, False, based on a
    """
    assert (label.shape == output.shape)

    label_max_mean, label_cords = find_max_area(label)
    output_max_mean, output_cords = find_max_area(output)

    lx = np.arange(label_cords[0][0], label_cords[0][1])
    ly = np.arange(label_cords[1][0], label_cords[1][1])

    ox = np.arange(output_cords[0][0], output_cords[0][1])
    oy = np.arange(output_cords[1][0], output_cords[1][1])

    X = [i for i in ox if i in lx]
    Y = [i for i in oy if i in ly]

    label_area = len(lx) * len(ly)
    cover_area = len(X) * len(Y)

    if cover_area / label_area > threshold:
        return True
    return False


class Truth_table:
    def __init__(self):
        self.d = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

        self.hits = self.misses = self.d
        self.trues = self.falses = 0
        self.mat = np.zeros((10, 10))

    def add_true(self, n):
        self.trues += 1
        self.hits[n] += 1
        self.mat[n][n] += 1

    def add_false(self, expected):
        self.falses += 1
        self.misses[expected] += 1
        # self.mat[expected][actual] += 1

    def clear(self):
        self.hits = self.misses = self.d
        self.trues = self.falses = 0
        self.mat = np.zeros((10, 10))


# @tf.function
def find_max_area(img):
    W = img.shape[1]
    H = img.shape[0]
    mean = np.mean(img)
    max_mean = 0
    max_mean_cords = [(0), (0)]
    for row in range(1, W - W // 4):
        for col in range(1, H - H // 4):
            temp_area = img[row:row + H // 4, col:col + W // 4]
            if np.mean(temp_area) >= max_mean:
                max_mean = np.mean(temp_area)
                max_mean_cords[0] = (row, row + H // 4)
                max_mean_cords[1] = (col, col + W // 4)
    if max_mean != 0:
        return max_mean, max_mean_cords
    else:
        return None


# @tf.function
def show_max_area(img):
    max_mean, cords = find_max_area(img)
    overlay = np.zeros(shape=img.shape)
    overlay[cords[0][0]:cords[0][1], cords[1][0]:cords[1][1]] = 1
    plt.imshow(img, cmap='gray')
    plt.imshow(overlay, cmap='jet', alpha=0.5)


@tf.function
def rescale_batch(batch, new_size):
    tmp_b = batch
    tmp_b = tf.reshape(tmp_b, (tmp_b.shape[0], tmp_b.shape[1], tmp_b.shape[2], 1))
    with_pad = tf.image.resize_with_pad(tmp_b, new_size, new_size, antialias=True)
    return tf.reshape(with_pad, (tmp_b.shape[0], with_pad.shape[1], with_pad.shape[2]))


@tf.function
def save_model_to_json(model, model_shape:tuple, dir_to_save:str, name_to_save:str):

    res = tf.keras.models.Model.to_json(model)
    res = json.loads(res)

    # res["model_name"]       = model_name
    # res["inputs_path"]      = inputs_path
    # res["targets_path"]     = targets_path
    # res["num_of_samples"] = num_of_samples
    # res["inputs"] = None
    # res["targets"] = None
    # res["num_of_layers"] = num_of_layers
    # res["z"] = layers_distance_pxls
    # res["wavelen"] = wavelen_in_pxls
    # res["amp_mod"] = amp_modulation
    # res["phase_mod"] = phase_modulation
    # res["rescaling_f"] = rescaling_factor
    # res["padding_f"] = padding_factor
    # res["nm"] = nm,
    # res["lr"] = learning_rate
    # res["batch_size"] = batch_size
    # res["epochs"] = epochs
    # res["prop_input"] = prop_input
    # res["padz"] = 0
    # res["padding"] = None
    # res["dir_to_save"] = dir_to_save
    # res["name_to_save"] = name_to_save
    # res["running_model"] = False
    res["model_shape"] = model_shape
    res["weights"] = {}

    for w in model.weights:
        res["weights"][w.name] = w.numpy().tolist()

    assert ".txt" in name_to_save, "ERROR: add .txt to name_to_save"
    with open(dir_to_save + '/' + name_to_save, "w") as fp:
        json.dump(res, fp)

    print("Success! - save model @ : ",dir_to_save+'/'+name_to_save)


@tf.function
def save_weights(model, dir_path, weights_name):
    print("**** Make sure you did: pip install -q pyyaml h5py ****")
    model.save_weights(dir_path + '/' + weights_name)



## TODO finish this function!
## NOTE - THERE ARE NOT IN USE CURRENTLY ##
@tf.function
def fit_model(model,
              inputs,
              targets,
              samples_num,
              inputs_shape,
              rescaling=False,
              padding_inputs=False,
              rescale_muli=None,
              padding_size=None,
              batch_size=8,
              EPOCHS=1,
              weights_path=None):
    # main_inputs = idx2numpy.convert_from_file(inputs_path)[:samples_num]/ 255
    # main_targets = idx2numpy.convert_from_file(targets_path)[:samples_num]

    inputs = inputs[:samples_num]
    targets = targets[:samples_num]
    print("inputs shape ", inputs.shape)
    if (rescaling == True):
        rescaled_HEIGHT, rescaled_WIDTH = inputs_shape[0] * rescale_muli, inputs_shape[
            1] * rescale_muli
        rescaled_SHAPE = (rescaled_HEIGHT, rescaled_WIDTH)
        inputs = rescale_batch(inputs, rescaled_HEIGHT)
        weights_name = "/rescaled_to_{}X{}.h5".format(rescaled_HEIGHT, rescaled_WIDTH)

        if (padding_inputs == True):
            inputs = tf.pad(inputs, padding_size)
            padded_SHAPE = inputs.shape[1], inputs.shape[2]
            weights_name = "/rescaled_and_padded_to_{}X{}.h5".format(
                rescaled_SHAPE[0] + padded_SHAPE[0],
                rescaled_SHAPE[1] + padded_SHAPE[1])

        print(samples_num)
        print(batch_size)
        new_inp_size = samples_num // batch_size

        for e in range(EPOCHS):
            print("Epoch num #{}".format(e))
            for i in range(new_inp_size):
                ## make a batch
                tmp_inputs = tf.cast(inputs[i:i + batch_size], tf.float32)
                tmp_targets = tf.reshape(targets[i:i + batch_size],
                                         (batch_size, targets.shape[1], targets.shape[2]))
                tmp_targets = tf.cast(tmp_targets, tf.float32)

                ## rescale
                tmp_inputs = tf.cast(rescale_batch(tmp_inputs, rescaled_HEIGHT), tf.complex128)
                if (padding_inputs == True):

                    tmp_targets = tf.cast(
                        rescale_batch(tmp_targets, rescaled_HEIGHT + padded_SHAPE[0]),
                        tf.complex128)
                else:
                    tmp_targets = tf.cast(rescale_batch(tmp_targets, rescaled_HEIGHT),
                                          tf.complex128)

                ## reshape targets
                tmp_targets = tf.reshape(tmp_targets,
                                         (batch_size, tmp_targets.shape[1] * tmp_targets[2]))

                ## fit
                model.fit(tmp_inputs, tmp_targets, epochs=1, use_multiprocessing=True, verbose=0)
        assert (weights_path != None)
        model.save_weights(weights_path + weights_name)

    else:  ## if not rescaling
        if (padding_inputs == True):
            inputs = tf.pad(inputs, padding_size)
            weights_name = "/only_padded_{}X{}.5h".format(inputs.shape[1], inputs.shape[2])
            new_inp_size = inputs.shape[0] // batch_size
            for e in range(EPOCHS):
                print("Epoch num #{}".format(e))
                for i in range(new_inp_size):
                    ## make a batch
                    tmp_inputs = tf.cast(inputs[i:i + batch_size], tf.float32)
                    tmp_targets = tf.reshape(targets[i:i + batch_size],
                                             (batch_size, targets.shape[1], targets.shape[2]))
                    tmp_targets = tf.cast(tmp_targets, tf.float32)
                    tmp_targets = tf.cast(rescale_batch(tmp_targets, tmp_inputs.shape[1]),
                                          tf.complex128)
                    tmp_targets = tf.reshape(tmp_targets,
                                             (batch_size, tmp_targets.shape[1] * tmp_targets[2]))

                    ## fit
                    model.fit(tmp_inputs, tmp_targets, epochs=1, use_multiprocessing=True,
                              verbose=0)
            assert (weights_path != None)
            model.save_weights(weights_path + weights_name)
        else:
            print("inputs shape and type : ", inputs.shape, type(inputs))
            print("targets shape and type: ", targets.shape, type(targets))
            weights_name = "/basic_{}X{}.h5".format(inputs.shape[1], inputs.shape[2])
            steps_per_epoch = inputs.shape[0] // batch_size
            model.fit(inputs, targets, batch_size=batch_size, epochs=EPOCHS,
                      use_multiprocessing=True, steps_per_epoch=steps_per_epoch)
            model.save_weights(weights_path + weights_name)

            print(model.summery())


@tf.function
def predict(model, input_batch_to_predict, targets_batch, weights_path, rescale=False,
            rescale_multi=None, pad=False, padding=None, plot=False):
    model.build(input_batch_to_predict.shape)
    model.load_weights(weights_path)
    original_shape = input_batch_to_predict.shape[1:]
    if rescale == True:
        input_batch_to_predict = rescale_batch(input_batch_to_predict,
                                               input_batch_to_predict.shape[1] * rescale_multi)
    if pad == True:
        input_batch_to_predict = tf.pad(input_batch_to_predict, padding)

    if (len(targets_batch.shape) == 2):
        targets_batch = tf.reshape(targets_batch,
                                   (targets_batch.shape[0], original_shape[0], original_shape[1]))
    targets_batch = rescale_batch(targets_batch, input_batch_to_predict.shape[1])

    y = model.predict(input_batch_to_predict)

    if plot == True:
        for i in range(y.shape[0]):
            plt.subplot(311)
            plt.imshow(tf.abs(input_batch_to_predict[i]))
            plt.title("Input")

            plt.subplot(312)
            plt.imshow((tf.abs(targets_batch[i])))
            plt.title("Target")

            plt.subplot(313)
            plt.imshow(tf.abs(y[i]) ** 2)
            plt.colorbar()
            plt.title("Output")

            plt.show()

    return
