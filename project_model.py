from project_defs import *


class CustomModel():
    def __init__(self,
                 model_name,
                 inputs_path,
                 targets_path,
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
                 phase_modulation = False):

        self.model_name     = model_name
        self.inputs_path    = inputs_path
        self.targets_path   = targets_path
        self.num_of_samples = num_of_samples
        self.inputs         = None
        self.targets        = None

        self.num_of_layers  = num_of_layers
        self.z              = layers_distance_pxls
        self.wavelen        = wavelen_in_pxls
        self.amp_mod        = amp_modulation
        self.phase_mod      = phase_modulation
        self.rescaling_f    = rescaling_factor
        self.padding_f      = padding_factor
        self.nm             = nm,
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

        self.running_model  = False

        print("**** Processing Parameters ****")
        self.processParams()
        print("**** Finished Processing Parameters ****")
        print("**** Building Model ****")
        self.buildModel()
        print("**** Finished Building Model ****")
        # print("**** Training Model ****")
        # self.trainModel()
        # print("**** Finished Training Model ****")


    def processParams(self):
        print("---- processing model parameters ----")
        if "ubyte" in self.inputs_path:
            self.inputs = idx2numpy.convert_from_file(self.inputs_path)[:self.num_of_samples] / 255
            self.targets= idx2numpy.convert_from_file(self.targets_path)[:self.num_of_samples]

        else:
            print("---- ERROR: currently not supporting input files type!! ----")
            exit(FAILURE)

        assert self.epochs > 0

        shape_1d = (self.inputs.shape)[0]

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
                           "input prop : {}\n# of layers{}".format(self.model_name,
                                                                    self.shape,
                                                                    self.lr,
                                                                    self.z,
                                                                    self.wavelen,
                                                                    self.prop_input,
                                                                   self.num_of_layers)
        print(line)

        self.weight_path = "model_{}_phase_modulate_{}_amp_modulate_{}_shape_{}_learning_rate_{}_Z_{}_lambda_{}_input prop{}_#_of_layers{}".format(self.model_name,
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

        flat_layer = tf.keras.layers.Flatten
        prop_layer = tf.keras.layers.Lambda(project_prop.my_fft_prop)
        output_layer = project_layers.output_layer

        model = tf.keras.models.Sequential()
        if self.prop_input:
            model.add(prop_layer)
            model.add(flat_layer())

        for i in range(self.num_of_layers):
            model.add(self.layers[i])
            model.add(flat_layer())

        model.add(output_layer)

        model.compile(
            tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
            loss=tf.keras.losses.mse, metrics=['mae', 'mse']
        )
        assert self.model == None, "---- ERROR : model already exists! ----"
        self.running_model = True
        self.model = model


    def trainModel(self):

        print("**** Training Model ****")
        model = self.model
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
                target_batch = tf.reshape(target_batch,
                                          (self.batch_size, self.shape[1], self.shape[2]))

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
        print(model.summery())
        self.model = model
        project_utils.save_model_to_json(self.model,
                                         self.shape,
                                         self.dir_to_save,
                                         self.name_to_save)

        self.model.save_weights(self.dir_to_save+'/'+self.weights_name)
        print("*** Finished training - saved weights @ {} ***".format(self.dir_to_save +'/'+self.weights_name))

    def rebuildModel(self):
        """
        This function gets a path for a JSON file, desribing a model, an rebuild it before
        testing the model
        :return:
        """

        # only rebuild if the model is imported and not already run
        if self.running_model == False:
            self.processParams()
            self.buildModel()
            self.model = self.model.build((1, self.shape[0], self.shape[1]))
            self.model.load_weights(self.dir_to_save+'/'+self.weights_name)
            self.running_model = True
            print("**** Loaded Weights ****")


    def testModel(self, test_low_idx, test_high_idx, num_of_samples_to_tests,
                  numeric_targets):

        if self.running_model == False:
            print("ERROR: cannot test model, 'running_model' if False")
            exit(FAILURE)

        # get truth table
        TT = project_utils.Truth_table()
        TT.clear()

        #create a list of indexes for tests
        test_idx_list = np.random.randint(test_low_idx, test_high_idx, num_of_samples_to_tests)

        count = 1
        for idx in test_idx_list:
            print("running test #{} out of #{}, with idx = {}".format(count,
                                                                      num_of_samples_to_tests, idx))

            # TODO - update this work with other inputs
            inp = self.inputs[idx] / 255
            target = self.targets[idx]

            # deal with rescaling

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


            y = self.model.predict(inp)

            WIDTH, HEIGHT = self.shape

            if (project_utils.compare_imgs(tf.reshape(target, (WIDTH, HEIGHT)),
                             tf.reshape(y, (WIDTH, HEIGHT)),
                             0.3) == True):
                TT.add_true(numeric_targets[idx])
                print("TRUE")
            else:
                TT.add_false(numeric_targets[idx])
                print("FALSE")
                plt.subplot(311)
                plt.imshow(tf.abs(tf.reshape(inp, (WIDTH, HEIGHT))))
                plt.subplot(312)
                plt.imshow(tf.reshape(target, (WIDTH, HEIGHT)))
                plt.subplot(313)
                project_utils.show_max_area(tf.reshape(y, (WIDTH, HEIGHT)))
                project_utils.show_max_area((y.reshape(WIDTH, HEIGHT)))
                plt.show()

            print("# of True: ", TT.trues)
            print("# of False: ", TT.falses)

        print("Total # of True: ", TT.trues)
        print("Total # of False: ", TT.falses)
