from project_defs import *
import matplotlib.pyplot as plt

class CustomModel():
    def __init__(self,
                 model_name,
                 inputs_path,
                 targets_path,
                 test_inputs_path,
                 test_lables_path,
                 num_of_samples,
                 num_of_layers,
                 layers_distance_pxls,
                 wavelen_in_pxls,
                 nm,
                 learning_rate,
                 batch_size,
                 epochs,
                 dir_to_save,
                 name_to_save,
                 weights_name,
                 prop_input = True,
                 rescaling_factor = None,
                 padding_factor = None,
                 amp_modulation = False,
                 phase_modulation = False,
                 force_shape = None,
                 **kwargs):

        self.model_name     = model_name
        self.inputs_path    = inputs_path
        self.targets_path   = targets_path
        self.test_inputs_path = test_inputs_path
        self.test_labels_path = test_lables_path
        self.num_of_samples = num_of_samples
        self.inputs         = None
        self.targets        = None
        self.tests_inputs   = None
        self.test_targets_labels   = None
        self.num_of_layers  = num_of_layers
        self.z              = layers_distance_pxls
        self.wavelen        = wavelen_in_pxls
        self.amp_mod        = amp_modulation
        self.phase_mod      = phase_modulation
        self.rescaling_f    = rescaling_factor
        self.padding_f      = padding_factor
        self.nm             = nm
        self.lr             = learning_rate
        self.batch_size     = batch_size
        self.epochs         = epochs
        self.prop_input     = prop_input
        self.padz           = 0
        self.padding        = None
        self.shape          = None
        self.layers         = []
        self.weights        = []
        self.model          = None
        self.dir_to_save    = dir_to_save
        self.name_to_save   = name_to_save
        self.weights_name   = weights_name

        self.force_shape = force_shape
        self.running_model  = False


        self.printHeader()
        print("**** Processing Parameters ****\n")
        self.processParams()
        print("**** Finished Processing Parameters ****\n")
        print("**** Building Model ****")
        self.buildModel()
        print("**** Finished Building Model ****\n")
        # print("**** Training Model ****")
        # self.trainModel()
        # print("**** Finished Training Model ****")


    def processParams(self):
        print("---- processing model parameters ----\n")
        if "ubyte" in self.inputs_path:
            self.inputs = idx2numpy.convert_from_file(self.inputs_path)[:self.num_of_samples] / 255
            self.targets= idx2numpy.convert_from_file(self.targets_path)[:self.num_of_samples]
            self.tests_inputs = idx2numpy.convert_from_file(self.test_inputs_path) / 255
            self.test_targets_labels = idx2numpy.convert_from_file(self.test_labels_path) ## note
            # - these are the labels only! need to convert to 2D

        else:
            print("---- ERROR: currently not supporting input files type!! ----")
            exit(FAILURE)

        assert self.epochs > 0, "number of epochs should be > 0"
        assert len(self.inputs.shape) >= 3
        shape_1d = (self.inputs.shape)[1]

        if (self.force_shape != None):  # FIXME: currntly the force shape is the only way to give
            # the system the right shape
            shape_1d = self.force_shape

        if self.rescaling_f != None:
            shape_1d *= self.rescaling_f

        if self.padding_f != None:
            self.padz = math.floor(shape_1d*self.padding_f)
            shape_1d += self.padz
            assert len(self.inputs.shape) == 3, "shape of input (4D?) is not yet supported"
            self.padding = [[0,0], [self.padz, self.padz], [self.padz, self.padz]]

        self.shape = (shape_1d, shape_1d)
        print("---- model shape after processing is : {} ----".format(self.shape))

        line = "model : {}\nshape : {}\nlearning_rate : {}\nZ :{}\nlambda :{}\n" \
                           "input prop : {}\n# of layers_{}_phase_mode_{}_amp_mod_{}".format(
            self.model_name,
                                                                    self.shape,
                                                                    self.lr,
                                                                    self.z,
                                                                    self.wavelen,
                                                                    self.prop_input,
                                                                   self.num_of_layers,
                                                                    self.phase_mod,
                                                                    self.amp_mod)
        # print(line)

        self.weight_path = "model_{}_phase_modulate_{}_amp_modulate_{}_shape_{}_learning_rate_{}_Z_{}_lambda_{}_input prop{}_#_of_layers_{}".format(self.model_name,
                                                                 self.phase_mod,
                                                                 self.amp_mod,
                                                       self.shape,
                                                       self.lr,
                                                       self.z,
                                                       self.wavelen,
                                                       self.prop_input,
                                                       self.num_of_layers)


    def buildModel(self):
        ## build layers ##
        print("**** Make sure you did: pip install -q pyyaml h5py ****")
        for i in range(self.num_of_layers):
            layer = project_layers.ModularLayer("layer : {}".format(i),
                                                self.z,
                                                self.shape,
                                                self.wavelen,
                                                self.nm,
                                                self.amp_mod,
                                                self.phase_mod)


            self.layers.append(layer)
        print("---- model has {} layers ----".format(len(self.layers)))

        prop_layer = project_layers.PropLayer("prop_layer", self.z, self.shape, self.wavelen,
                                              self.nm)

        output_layer = project_layers.output_layer

        model = tf.keras.models.Sequential()
        if self.prop_input:
            model.add(prop_layer)
            model.add(tf.keras.layers.Flatten())

        for i in range(self.num_of_layers):
            model.add(self.layers[i])
            model.add(tf.keras.layers.Flatten())

        model.add(output_layer)

        model.compile(
            tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
            loss=tf.keras.losses.mse, metrics=['mae', 'mse']
        )
        # assert self.model == None, "---- ERROR : model already exists! ----"
        self.running_model = True
        self.model = model


    def trainModel(self,
                   local_training_set = True,
                   tf_dataset_name = None):

        print("**** Training Model ****")
        model = self.model

        if (local_training_set == False):

            assert tf_dataset_name != None, "please give a tf_dataset name as an argument"

            ds, info = tfds.load(tf_dataset_name, split='train', shuffle_files=True, with_info=True)
            print("**** Model Info ****")
            print(info)
            num_of_examples = int (info.splits["train"].num_examples)
            for_loops_num = num_of_examples//self.batch_size

            for epoch in self.epochs:
                print("Start Epoch num {}".format(epoch))

                ds = ds.shuffle(num_of_examples).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
                for i in (range(for_loops_num)):

                    if (i % 100 == 0):
                        print("train number : {}".format(i))

                    for example in ds.take(1):
                        # covert inputs to grey scale and targets labels to 2D labels
                        input_batch, target_batch = project_utils.make_inputs_targets_batch(example)

                    if ((i % (for_loops_num // 10)) == 0):
                        print("step #{} out of {}".format(i, for_loops_num))


                    print("fitting input")
                    if self.rescaling_f > 0:
                        input_batch = project_utils.rescale_batch(input_batch,
                                                                  input_batch.shape[1]*
                                                                  self.rescaling_f)

                    if self.padding_f > 0:
                        input_batch = tf.pad(input_batch, self.padding)

                    print("fitting target")
                    if (target_batch.shape != input_batch.shape):
                        target_batch = project_utils.rescale_batch(target_batch, input_batch.shape[1])

                    assert (target_batch.shape == input_batch.shape), "---- input batch shape != " \
                                                                      "targets " \
                                                                      "shape! ----"
                    # print("input batch shape: {}, target shape: {}".format(input_batch.shape,
                    #                                               target_batch.shape))

                    input_batch = tf.cast(input_batch, tf.complex128)

                    ## note - reshaping targets like so only right for 3D targets !##
                    target_batch = tf.reshape(target_batch,(self.batch_size,
                                                            target_batch.shape[1]*target_batch.shape[2]))

                    print("training")
                    model.fit(input_batch, target_batch, epochs=1, use_multiprocessing=True, verbose=0)

        else:
            new_inp_size = self.num_of_samples//self.batch_size
            for e in range(self.epochs):

                print("Epoch num #{} ot out {}".format(e + 1, self.epochs))
                for i in range(new_inp_size):

                    ## print what step were in
                    if ((i % (new_inp_size // 10)) == 0):
                        print("step #{} out of {}".format(i, new_inp_size))

                    ## process input/target batch ##
                    input_batch = tf.cast(self.inputs[i:i + self.batch_size], tf.float32)
                    target_batch = tf.cast(self.targets[i:i + self.batch_size], tf.float32)

                    # target_batch = tf.reshape(target_batch,
                    #                           (self.batch_size, self.shape[0], self.shape[1]))

                    if self.rescaling_f > 0:
                        input_batch = project_utils.rescale_batch(input_batch,
                                                                  input_batch.shape[1]*
                                                                  self.rescaling_f)

                    if self.padding_f > 0:
                        input_batch = tf.pad(input_batch, self.padding)

                    if (target_batch.shape != input_batch.shape):
                        target_batch = project_utils.rescale_batch(target_batch, input_batch.shape[1])

                    assert (target_batch.shape == input_batch.shape), "---- input batch shape != " \
                                                                      "targets " \
                                                                      "shape! ----"

                    input_batch = tf.cast(input_batch, tf.complex128)

                    ## note - reshaping targets like so only right for 3D targets !##
                    target_batch = tf.reshape(target_batch,(self.batch_size,
                                                            target_batch.shape[1]*target_batch.shape[2]))

                    model.fit(input_batch, target_batch, epochs=1, use_multiprocessing=True, verbose=0)


        ## after training finished
        print(model.summary())
        self.model = model

        ## TODO - fix saveing model!!!
        # model.save(self.dir_to_save + '/' + self.name_to_save)
        model.save("{}/{}".format(self.dir_to_save,self.model_name))
        model.save_weights("{}/{}/".format(self.dir_to_save, self.model_name))

        print("*** Finished training - saved weights @ {} ***".format(self.dir_to_save +'/'+self.weights_name))


    def rebuildModel(self, dir_to_save, model_name, input_shape):
        """
        This function gets a path for a JSON file, desribing a model, an rebuild it before
        testing the model
        :return:
        """
        # TODO: for now input_shape must be forced via arguemt, next step is to save it with a
        #  JSON representaion of the network in the same dir as the weights, then pre-load that
        #  JSON file to get all the relevant data in ordder to load the weights propely
        try:
            custom_objects = {"output_func": project_layers.output_func,
                              "output_layer": project_layers.output_layer,
                              "my_fft_prop": project_prop.my_fft_prop,
                              "ModularLayer": project_layers.ModularLayer}

            self.model = tf.keras.models.load_model("{}/{}/".format(dir_to_save,model_name),
                                                    custom_objects = custom_objects)
            print("**** Full model loaded ****")
        except:
            self.model.build(input_shape = (None, ) + input_shape)
            self.model.load_weights("{}/{}/".format(dir_to_save,model_name))
            print("**** Weights Loaded ****")


    def testModel(self,
                  test_low_idx,
                  test_high_idx,
                  num_of_samples_to_tests,
                  numeric_targets,
                  local_training_set,
                  tf_dataset_name = None,
                  inputs_test_set = None,
                  monte_carlo = False,
                  monte_carlo_variance = None,
                  input_shift = False,
                  input_shift_percentrage = None
                  ):

        """

        :param test_low_idx: min for random index
        :param test_high_idx: max for random index
        :param num_of_samples_to_tests: number of samples
        :param numeric_targets: path to the array with the labels as numeric values
        :param inputs_test_set: path to the 2D test set
        :return:
        """
        if self.running_model == False:
            print("ERROR: cannot test model, 'running_model' if False")
            exit(FAILURE)

        assert (self.model != None), "Model cannot be None when testing!"
        # get truth table
        TT = project_utils.Truth_table()
        TT.clear()

        # # assert (local_training_set == False and tf_dataset_name != None), "If dataset is not " \
        #                                                                   "local, tf_dataset_name arg should be != None"

        true_count = 0
        flase_count = 0


        # Deal with tf_datasets:
        if (tf_dataset_name != None and local_training_set == False):
            ds, info = tfds.load(tf_dataset_name, split='test', shuffle_files=True, with_info=True)
            num_of_samples = int (info.splits["test"].num_examples)

            if monte_carlo == True:
                test_model = self.monteCarlo(monte_carlo_variance)
            else:
                test_model = self.model

            ds = ds.shuffle(num_of_samples).batch(1).prefetch(tf.data.experimental.AUTOTUNE)
            count = 0
            for i in range(num_of_samples_to_tests):

                for example in ds.take(1):
                    input_batch, target_batch = project_utils.make_inputs_targets_batch(example)
                    expected = example["label"].numpy()[0]
                    # print(expected)
                    # print(example)
                    # print(example["label"].numpy)

                inp_shape = input_batch.shape
                if len(inp_shape) == 2:
                    inp_shape = (1, inp_shape[0], inp_shape[1], 1)
                if len(inp_shape) == 3:
                    inp_shape = (1, inp_shape[0], inp_shape[1], inp_shape[2])


                if self.rescaling_f > 0:
                    input_batch = project_utils.rescale_batch(input_batch,
                                                              input_batch.shape[1]*
                                                              self.rescaling_f,
                                                              input_shift,
                                                              input_shift_percentrage)

                if self.padding_f > 0:
                    input_batch = tf.pad(input_batch, self.padding)

                if (target_batch.shape != input_batch.shape):
                    target_batch = project_utils.rescale_batch(target_batch, input_batch.shape[1])

                assert (target_batch.shape == input_batch.shape), "---- input batch shape != " \
                                                                  "targets " \
                                                                  "shape! ----"

                input_batch = tf.cast(input_batch, tf.complex128)

                ## note - reshaping targets like so only right for 3D targets !##
                # target_batch = tf.reshape(target_batch,(self.batch_size,
                #                                         target_batch.shape[1]*target_batch.shape[2]))
                # if self.rescaling_f > 0:
                #     inp = tf.image.resize_with_pad(tf.reshape(input_batch, inp_shape), self.shape[0],
                #                                    self.shape[1])
                #
                # if self.padding_f > 0:
                #     inp = tf.pad(inp, self.padding)
                #
                # target = tf.image.resize_with_pad(tf.reshape(target_batch, inp_shape), self.shape[0],
                #                                   self.shape[1])
                #
                # inp = tf.reshape(inp, (1, self.shape[0], self.shape[1]))
                # target = tf.reshape(target, (self.shape[0], self.shape[1]))
                #
                # inp = tf.cast(inp, tf.complex128)
                # assign Monte Carlo model or the clean model

                y = test_model.predict(input_batch)

                WIDTH, HEIGHT = self.shape

                if (project_utils.compare_imgs(tf.reshape(target_batch, (WIDTH, HEIGHT)),
                                               tf.reshape(y, (WIDTH, HEIGHT)),
                                               0.3) == True):
                    # TT.add_true(numeric_targets[idx])  // TODO - uncomment and debug
                    print("TRUE")
                    # plt.subplot(311)
                    # plt.imshow(tf.abs(tf.reshape(input_batch, (WIDTH, HEIGHT))))
                    # plt.subplot(312)
                    # plt.imshow(tf.reshape(target_batch, (WIDTH, HEIGHT)))
                    # plt.subplot(313)
                    # project_utils.show_max_area(tf.reshape(y, (WIDTH, HEIGHT)))
                    # project_utils.show_max_area((y.reshape(WIDTH, HEIGHT)))
                    # plt.show()
                    true_count+=1
                    TT.add_true(expected)

                else:
                    print("FALSE")
                    plt.subplot(311)
                    plt.imshow(tf.abs(tf.reshape(input_batch, (WIDTH, HEIGHT))))
                    plt.subplot(312)
                    plt.imshow(tf.reshape(target_batch, (WIDTH, HEIGHT)))
                    plt.subplot(313)
                    project_utils.show_max_area(tf.reshape(y, (WIDTH, HEIGHT)))
                    project_utils.show_max_area((y.reshape(WIDTH, HEIGHT)))
                    plt.show()
                    flase_count+=1
                    actual_label = project_utils.result_to_label(project_utils.find_max_area(
                        y.reshape(HEIGHT, WIDTH))[1], y.reshape(HEIGHT, WIDTH))

                    print("expected label: {}, got label: {}".format(expected, actual_label))
                    TT.add_false(expected, actual_label)

                count += 1

                # print("# of True: ", TT.trues)
                # print("# of False: ", TT.falses)

            print("Total # of True: ", true_count)
            print("Total # of False: ", flase_count)
            TT.print_table("Mnist", "Got", "Expected")

        else:
            #create a list of indexes for tests
            test_idx_list = np.random.randint(test_low_idx, test_high_idx, num_of_samples_to_tests)

            # assign Monte Carlo model or the clean model
            if monte_carlo == True:
                test_model = self.monteCarlo(monte_carlo_variance)
            else:
                test_model = self.model

            count = 1
            for idx in test_idx_list:
                print("running test #{} out of #{}, with idx = {}".format(count,
                                                                          num_of_samples_to_tests, idx))

                # TODO - update this work with other inputs
                if (inputs_test_set != None):
                    inp = self.tests_inputs[idx]
                    assert (len(inp.shape) == 2)
                    target = project_utils.make_2D_label(self.test_targets_labels[idx], inp.shape)
                    expected = self.test_targets_labels[idx]

                else:
                    inp = self.inputs[idx]
                    target = self.targets[idx]

                # deal with rescaling
                if (input_shift):
                    inp = project_utils.shift_pic_rand(inp, input_shift_percentrage, True)

                inp_shape = inp.shape
                if len(inp_shape) == 2:
                    inp_shape = (1, inp_shape[0], inp_shape[1], 1)
                if len(inp_shape) == 3:
                    inp_shape = (1, inp_shape[0], inp_shape[1], inp_shape[2])

                if self.rescaling_f > 0:
                    inp = tf.image.resize_with_pad(tf.reshape(inp, inp_shape),self.shape[0],
                                                   self.shape[1])

                if self.padding_f > 0:
                    inp = tf.pad(inp, self.padding)


                target = tf.image.resize_with_pad(tf.reshape(target, inp_shape), self.shape[0],
                                                  self.shape[1])

                inp     = tf.reshape(inp, (1, self.shape[0], self.shape[1]))
                target  = tf.reshape(target, (self.shape[0], self.shape[1]))

                inp = tf.cast(inp, tf.complex128)


                y = test_model.predict(inp)

                WIDTH, HEIGHT = self.shape

                if (project_utils.compare_imgs(tf.reshape(target, (WIDTH, HEIGHT)),
                                 tf.reshape(y, (WIDTH, HEIGHT)),
                                 0.3) == True):
                    # TT.add_true(numeric_targets[idx])  // TODO - uncomment and debug
                    print("TRUE")
                    true_count += 1
                    TT.add_true(expected)
                else:
                    print("FALSE")
                    plt.subplot(311)
                    plt.imshow(tf.abs(tf.reshape(inp, (WIDTH, HEIGHT))))
                    plt.subplot(312)
                    plt.imshow(tf.reshape(target, (WIDTH, HEIGHT)))
                    plt.subplot(313)
                    project_utils.show_max_area(tf.reshape(y, (WIDTH, HEIGHT)))
                    project_utils.show_max_area((y.reshape(WIDTH, HEIGHT)))
                    plt.show()
                    flase_count += 1
                    actual_label = project_utils.result_to_label(project_utils.find_max_area(
                        y.reshape(HEIGHT, WIDTH))[1], y.reshape(HEIGHT, WIDTH))

                    print("expected label: {}, got label: {}".format(expected, actual_label))
                    TT.add_false(expected, actual_label)

                count+=1

                # print("# of True: ", TT.trues)
                # print("# of False: ", TT.falses)

            print("Total # of True: ", TT.trues)
            print("Total # of False: ", TT.falses)
            TT.print_table("Mnist", "Got", "Expected")



    def printHeader(self):
        print(header.format(self.model_name, self.num_of_layers, self.z,
                                          self.wavelen, self.nm))

    def monteCarlo(self, variance):

        assert (variance < 1), "Variance should be < 1 (in %)"

        assert (self.model != None), "Model cannot be None!"

        monte_carlo_model = self.model

        for i in range(len(monte_carlo_model.layers)):
            weights = monte_carlo_model.layers[i].get_weights()
            num_of_weights = len(weights)

            if (num_of_weights > 0):
                for w in range(num_of_weights):
                    weights[w] *= (1 + np.random.uniform(0, variance, weights[w].shape))
                    monte_carlo_model.layers[i].set_weights(weights)
        return monte_carlo_model