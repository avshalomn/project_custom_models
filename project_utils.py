from project_defs import *
import matplotlib.pyplot as plt


# @tf.function
def get_actual_square_coords(coords, sqaure_hight, square_third_width, square_fourths_width,
                             second_row: bool):
    h, w = coords
    if (second_row == False):
        return [coords[0] - sqaure_hight // 2, coords[0] + sqaure_hight // 2,
                coords[1] - square_third_width // 2, coords[1] + square_third_width // 2]
    else:
        return [coords[0] - sqaure_hight // 2, coords[0] + sqaure_hight // 2,
                coords[1] - square_fourths_width // 2, coords[1] + square_fourths_width // 2]


# @tf.function
def get_sqaures_corrs(label, original_input_shape: tuple):
    import math
    SQAURE_RATIO = 0.8
    assert (len(original_input_shape) == 2)
    h, w = original_input_shape

    h_third = h // 3
    w_third = h // 3
    w_fourth = h // 4

    first_row = int(h_third / 2)
    second_row = first_row + h_third
    third_row = second_row + h_third

    first_thirds_col = int(w_third / 2)
    second_thirds_col = first_thirds_col + w_third
    third_thirds_col = second_thirds_col + w_third

    first_fourths_col = int(w_fourth / 2)
    second_fourths_col = first_fourths_col + w_fourth
    third_fourths_col = second_fourths_col + w_fourth
    fourth_fourths_col = third_fourths_col + w_fourth

    # calculate the squares' width
    f_t_rows_sqarue_width = math.floor(SQAURE_RATIO * w_third)
    s_row_square_width = math.floor(SQAURE_RATIO * w_fourth)

    # calculate the squares' hight
    sqaure_hight = math.floor(SQAURE_RATIO * h_third)

    #   coords, sqaure_hight, square_third_width, square_fourths_width, second_row : bool

    ## first row
    if label == 0:
        return get_actual_square_coords((first_row, first_thirds_col), sqaure_hight,
                                        f_t_rows_sqarue_width, s_row_square_width, False)
    if label == 1:
        return get_actual_square_coords((first_row, second_thirds_col), sqaure_hight,
                                        f_t_rows_sqarue_width, s_row_square_width, False)
    if label == 2:
        return get_actual_square_coords((first_row, third_thirds_col), sqaure_hight,
                                        f_t_rows_sqarue_width, s_row_square_width, False)

    ## second row
    if label == 3:
        return get_actual_square_coords((second_row, first_fourths_col), sqaure_hight,
                                        f_t_rows_sqarue_width, s_row_square_width, True)
    if label == 4:
        return get_actual_square_coords((second_row, second_fourths_col), sqaure_hight,
                                        f_t_rows_sqarue_width, s_row_square_width, True)
    if label == 5:
        return get_actual_square_coords((second_row, third_fourths_col), sqaure_hight,
                                        f_t_rows_sqarue_width, s_row_square_width, True)
    if label == 6:
        return get_actual_square_coords((second_row, fourth_fourths_col), sqaure_hight,
                                        f_t_rows_sqarue_width, s_row_square_width, True)

    ## third row
    if label == 7:
        return get_actual_square_coords((third_row, first_thirds_col), sqaure_hight,
                                        f_t_rows_sqarue_width, s_row_square_width, False)
    if label == 8:
        return get_actual_square_coords((third_row, second_thirds_col), sqaure_hight,
                                        f_t_rows_sqarue_width, s_row_square_width, False)
    if label == 9:
        return get_actual_square_coords((third_row, third_thirds_col), sqaure_hight,
                                        f_t_rows_sqarue_width, s_row_square_width, False)


# @tf.function
def make_2D_label(label: int, original_input_shape: tuple):
    assert (label >= 0 and label < 10)
    new_label = np.zeros(shape=original_input_shape)
    coors = get_sqaures_corrs(label, original_input_shape)
    new_label[coors[0]:coors[1], coors[2]:coors[3]] = 1

    new_label = tf.convert_to_tensor(new_label)

    return new_label


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def make_inputs_targets_batch(input_and_targets_batch):

    original_shape = input_and_targets_batch["image"][0].shape[:2]

    if (input_and_targets_batch["image"][0].shape[2] == 3):
        inp_batch = np.array([rgb2gray(i) for i in input_and_targets_batch["image"]])
    else:
        inp_batch = np.array([i for i in input_and_targets_batch["image"]])
    numeric_lables = [i for i in input_and_targets_batch["label"]]

    # print("the numeric labels are: ", numeric_lables)
    targets_batch = np.array([make_2D_label(l.numpy(), original_shape) for l in numeric_lables])

    return inp_batch, targets_batch




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

    def add_false(self, expected, actual):
        self.falses += 1
        self.misses[expected] += 1
        self.mat[expected][actual] += 1

    def clear(self):
        self.hits = self.misses = self.d
        self.trues = self.falses = 0
        self.mat = np.zeros((10, 10))


    def print_table(self,
                    data_set_name : str,
                    x_title : str,
                    y_title : str
                    ):
        fig, ax = plt.subplots()
        im = ax.imshow(self.mat, cmap='YlGn')
        for i in range(10):
            for j in range(10):
                text = ax.text(j, i, round(self.mat[i, j],2),
                               ha="center", va="center", color="black")

        ax.set_title("Confusion Matrix for: {}".format(data_set_name))
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        fig.tight_layout()
        plt.show()



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


def result_to_label(result_coors, img):


    for i in range(10):
        local_coors = [result_coors[0][0],result_coors[0][1], result_coors[1][0], result_coors[
            1][1]]
        two_d_label = get_sqaures_corrs(i, img.shape)

        # fit the area to the label
        while(abs(local_coors[1] - local_coors[0]) > abs(two_d_label[1] - two_d_label[0])):
            local_coors[0]+=1
            local_coors[1]-=1

        while(abs(local_coors[3] - local_coors[2]) > abs(two_d_label[3] - two_d_label[2])):
            local_coors[2]+=1
            local_coors[3]-=1


        if (
            get_precentage_in_range((local_coors[0], local_coors[1]),
                                    (two_d_label[0], two_d_label[1])) > 0.7 and
            get_precentage_in_range((local_coors[2],local_coors[3]),
                                    (two_d_label[2],two_d_label[3])) > 0.7):

            print("actual label was: {}".format(i))
            return i
    return None


def get_precentage_in_range(tup1 : tuple,
                            tup2 : tuple):
    sum = 0

    if (abs(tup1[1] - tup1[0]) > abs(tup2[1] - tup2[0])):
        for i in range(tup2[0], tup2[1]):
            if i >= tup1[0] and i <= tup1[1]:
                sum+=1
        return sum/len(tup1)

    for i in range(tup1[0], tup1[1]):
        if i >= tup2[0] and i <= tup2[1]:
            sum+=1
    return sum/len(tup2)






@tf.function
def rescale_batch(batch, new_size, shift_input = False, shift_per = None):
    tmp_b = batch
    tmp_b = tf.reshape(tmp_b, (tmp_b.shape[0], tmp_b.shape[1], tmp_b.shape[2], 1))
    with_pad = tf.image.resize_with_pad(tmp_b, new_size, new_size, antialias=True)
    if (shift_input):
        fnc = lambda x : shift_pic_rand(x, shift_per, True)
        tf.map_fn(fnc, with_pad)
        return tf.reshape(with_pad, (tmp_b.shape[0], with_pad.shape[1], with_pad.shape[2]))
    else:
        return tf.reshape(with_pad, (tmp_b.shape[0], with_pad.shape[1], with_pad.shape[2]))

@tf.function
def shift_pic_rand(image , max_percent : float, rand_mode = False):
    """

    :param image:
    :param max_percent: maximum percent of shift
    :param rand_mode:
    :return:
    """
    assert (max_percent <= 0.1), "max percent cannot be bigger than 10% - in risk of loosing data"

    if (rand_mode == True):
        pixels_to_shift_y_ax = int(np.random.uniform(0, max_percent) * (image.shape[0]))
        pixels_to_shift_x_ax = int(np.random.uniform(0, max_percent) * (image.shape[1]))
    else:
        pixels_to_shift_y_ax = int(max_percent * (image.shape[0]))
        pixels_to_shift_x_ax = int(max_percent * (image.shape[1]))

    add_to_top = np.random.randint(2)
    add_to_left = np.random.randint(2)

    # new_img = np.asarray(image)
    new_img = tf.convert_to_tensor(image)


    res = np.zeros(image.shape)

    if (add_to_top):
        if(add_to_left):
            pads    = tf.constant([[pixels_to_shift_y_ax,0],[pixels_to_shift_x_ax,0]])
            # new_img = np.pad(new_img, [[pixels_to_shift_y_ax,0],[pixels_to_shift_x_ax,0]])
            new_img = tf.pad(new_img, pads)
            res     = new_img[0 : res.shape[0], 0: res.shape[1]]
        else:
            pads    = tf.constant( [[pixels_to_shift_y_ax,0],[0,pixels_to_shift_x_ax]])
            # new_img = np.pad(new_img, [[pixels_to_shift_y_ax,0],[0,pixels_to_shift_x_ax]])
            new_img = tf.pad(new_img, pads)
            res     = new_img[0: res.shape[0],pixels_to_shift_x_ax : pixels_to_shift_x_ax + res.shape[0]]
    else:
        if (add_to_left):
            pads    = tf.constant([[0, pixels_to_shift_y_ax],[pixels_to_shift_x_ax, 0]])
            # new_img = np.pad(new_img, [[0, pixels_to_shift_y_ax],[pixels_to_shift_x_ax, 0]])
            new_img = tf.pad(new_img, pads)
            res     = new_img[pixels_to_shift_y_ax : pixels_to_shift_y_ax + res.shape[0],
                      0 : res.shape[1]]
        else:
            pads    = tf.constant([[0, pixels_to_shift_y_ax],[0,pixels_to_shift_x_ax]])
            # new_img = np.pad(new_img, [[0, pixels_to_shift_y_ax],[0,pixels_to_shift_x_ax]])
            new_img = tf.pad(new_img, pads)
            res     = new_img[pixels_to_shift_y_ax : pixels_to_shift_y_ax + res.shape[0],
                      pixels_to_shift_x_ax : pixels_to_shift_x_ax + res.shape[1]]

    return res

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

            print(model.sumery())


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
