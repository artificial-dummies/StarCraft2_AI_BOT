import tensorflow as tf
import keras.backend.tensorflow_backend as backend
import keras  # Keras 2.1.2 and TF-GPU 1.9.0
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import numpy as np
import os
import random
import cv2
import time

# This fractions up the memory of GPU so it can be be allocated for training
def get_session(gpu_fraction=0.75):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
backend.set_session(get_session())

# adding hidden layers for the network, use padding = same for less loss (pads zeroes along the edge of the image)
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(176, 200, 1),
                 activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Here we gather the output to apply on the model.
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='softmax'))

# setting the learning rate, can be adjusted to influence the model
learning_rate = 0.001
opt = keras.optimizers.adam(lr=learning_rate)#, decay=1e-6)

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# output for tensorboard to view the loss and accuracy
tensorboard = TensorBoard(log_dir="logs/STAGE1-{}-{}".format(int(time.time()), learning_rate))

train_data_dir = "train_data"

# loads up previous model, uncomment on the first go
model = keras.models.load_model('AD-10-epochs-0.001-LR-STAGE1.model')

# checks the choises made in the data and see the length
def check_data(choices):
    total_data = 0

    lengths = []
    for choice in choices:
        print("Length of {} is: {}".format(choice, len(choices[choice])))
        total_data += len(choices[choice])
        lengths.append(len(choices[choice]))

    print("Total data length now is:", total_data)
    return lengths

# setting epochs for training, increase this for more accurate training results, beware of GPU and CPU capabilities
hm_epochs = 1

for i in range(hm_epochs):
    current = 0     # starting point in data, changes in itterations
    increment = 50  # increment per itteration
    not_maximum = True  # Checks if you have reached the end of the training data
    all_files = os.listdir(train_data_dir)      # gets all training files
    maximum = len(all_files)        # sets the total number of training files
    random.shuffle(all_files)       # shuffle up the files for random selection of training files

    while not_maximum:
        try:
            print("WORKING ON {}:{}, EPOCH:{}".format(current, current+increment, i))

            # choices the agent has
            choices = {0: [],
                       1: [],
                       2: [],
                       3: [],
                       4: [],
                       5: [],
                       6: [],
                       7: [],
                       8: [],
                       9: [],
                       10: [],
                       11: [],
                       12: [],
                       13: [],
                       14: [],
                       15: [],
                       16: [],
                       17: [],
                       18: [],
                       19: []
                       }

            # increment through each file and select it
            for file in all_files[current:current+increment]:
                try:
                    full_path = os.path.join(train_data_dir, file)
                    data = np.load(full_path)
                    data = list(data)
                    for d in data:
                        choice = np.argmax(d[0])
                        choices[choice].append([d[0], d[1]])
                except Exception as e:
                    print(str(e))

            lengths = check_data(choices)

            lowest_data = min(lengths)

            # shuffle up the choices so that they do not line up in order
            for choice in choices:
                random.shuffle(choices[choice])
                choices[choice] = choices[choice][:lowest_data]

            check_data(choices)

            train_data = []

            # append choices in the training file to an array
            for choice in choices:
                for d in choices[choice]:
                    train_data.append(d)

            # shuffle up the choices so that they do not line up in order
            random.shuffle(train_data)
            print(len(train_data))

            # determine the test and batch size, modified depending on your hardware
            test_size = 100
            batch_size = 64  # 128 best so far.

            # create input variables to send into the model from the first 100 if the data + the CV2 we have in the data
            x_train = np.array([i[1] for i in train_data[:-test_size]]).reshape(-1, 176, 200, 1)
            y_train = np.array([i[0] for i in train_data[:-test_size]])

            # create input variables for validation to send into the model from the first 100 if the data + the CV2 we have in the data
            x_test = np.array([i[1] for i in train_data[-test_size:]]).reshape(-1, 176, 200, 1)
            y_test = np.array([i[0] for i in train_data[-test_size:]])

            # send in to the modal for training
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      validation_data=(x_test, y_test),
                      shuffle=True,
                      epochs=1,
                      verbose=1, callbacks=[tensorboard])

            # save the model
            model.save("AD-10-epochs-0.001-LR-STAGE1.model")
        except Exception as e:
            print(str(e))
        current += increment # increment for next itteration
        if current > maximum: # check if end of test files has been reached
            not_maximum = False