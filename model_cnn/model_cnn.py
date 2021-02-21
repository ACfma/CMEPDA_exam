# -*- coding: utf-8 -*-
"""
Example of tool used to create, fit 3D CNN model, and then uses it to
evaluate test set data in model_cnn.ipynb.

"""

from matplotlib import pyplot as plt
from keras.layers import Conv3D, MaxPooling3D, Dense, Flatten,\
    BatchNormalization, Dropout, InputLayer
from keras.models import Sequential

def model_build(shape):
    '''
    This function returns the model used to classificate
    AD/CTRL subject, it involved differents convolutional
    layers, pooling layer, fully connected layer and one
    output with sigmoid activation.

    Parameters
    ----------
    shape : class 'tuple'
        shape of images we use to feed the network.

    Returns
    -------
    model : class 'tensorflow.python.keras.\
            engine.Model.Sequential'
        Model, Convolutional NN.

    '''
    model = Sequential([

        InputLayer(input_shape=shape),
        Conv3D(6, (4,4,4), strides=2, activation = 'relu'),
        BatchNormalization(),
        MaxPooling3D((2,2,2), strides=(2,2,2)),
        Conv3D(8, (3,3,3),strides=2, activation = 'relu'),
        BatchNormalization(),
        Dropout(0.5),
        MaxPooling3D((2,2,2), strides=(2,2,2)),
        Conv3D(16, (3,3,3),strides=1, activation = 'relu'),
        BatchNormalization(),
        Dropout(0.7),
        MaxPooling3D((2,2,2), strides=1, padding='same'),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')

    ])
    return model

def model_fit(model, train_x, train_y, test_x, test_y, btc_s):
    '''
    This function first summarize the model previous created and then
    compiles it automatically with Adam optimizer, binary cross entropy
    loss and set metrics as accuracy.
    At last, afterd the model is trained, returns a plot of model losses
    and metrics choosen of train and validation test set data and evaluates
    the performances of the model to the test set data.
    Parameters:

    Parameters
    ----------
    model : class 'tensorflow.python.keras.\
            engine.sequential.Sequential'
        the model previous created.
    train_x : Vector, matrix, or array
        training data.
    train_y : Vector, matrix, or array
        target data.
    test_x : Vector, matrix, or array
        test data.
    test_y : Vector, matrix, or array
        target test data.
    btc : int
        select the batch size to control the number of
        samples per gradient update.

    Returns
    -------
    history : class 'tensorflow.python.keras.\
            engine.Model'
        Model, Convolutional NN.
    score_train : class 'ndarray'
        Loss and accuracy values for train test.
    score_test : class 'ndarray'
        Loss and accuracy values for test test.

    '''
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    history = model.fit(train_x, train_y, validation_split=0.2, epochs=100, batch_size = btc_s)
    score_train = model.evaluate(train_x, train_y, verbose=0)
    print(f'Train loss: {score_train[0]} / Train accuracy: {score_train[1]}')
    score_test = model.evaluate(test_x, test_y, verbose=0)
    print(f'Test loss: {score_test[0]} / Test accuracy: {score_test[1]}')
    plt.figure('loss')
    plt.title('Loss and Val_Loss vs epochs = {m}/100'.format(m = len(history.history['loss']) ))
    plt.plot(history.history['loss'], label ='loss')
    plt.plot(history.history['val_loss'], label ='val_loss')
    plt.grid()
    plt.legend()
    plt.show()
    plt.figure('acc')
    plt.title('Accuracy and val_accuracy\
              vs epochs = {m}/100'.format(m = len(history.history['accuracy']) ))
    plt.plot(history.history['accuracy'], label ='accuracy')
    plt.plot(history.history['val_accuracy'], label ='val_accuracy')
    plt.legend()
    plt.grid()
    plt.show()
    return history, score_train, score_test
