# -*- coding: utf-8 -*-
"""
Tool to create a 3D CNN model

"""

from matplotlib import pyplot as plt
from keras.layers import Conv3D, MaxPooling3D, Dense, Flatten,\
    BatchNormalization, Dropout, LeakyReLU, InputLayer, Softmax
from keras.models import Sequential

def model_build(shape):
    '''
    This function returns the model used to classificate
    AD/CTRL subject, it involved differents convolutional
    layers, pooling layer, fully connected layer and two
    output with softmax activation.

    Parameters
    ----------
    shape : class 'tuple'
        shape of images we use to feed the network.

    Returns
    -------
    model : class 'tensorflow.python.keras.\
            engine.Model'
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
        MaxPooling3D((2,2,2), strides=(2,2,2),padding = 'same'),
        Flatten(),
        Dense(32),
        LeakyReLU(alpha=0.1),
        Dense(2),
        Softmax()

    ])
    return model

def model_compiler(model, optimizer, loss, metrics):
    '''
    This function returns the model summary and the losses and metrics with
    model.compile()

    Parameters
    ----------
    model : class 'tensorflow.python.keras.\
            engine.Model'
        Model, Convolutional NN.
    optimizer : class 'tensorflow.python.keras.optimizer'
        select the gradient descent to minimize an objective
        function parameterized by the model's parameters by updating the
        parameters in the opposite direction of the gradient of the
        objective function.
    loss : str, binary_crossentropy
        select the loss function to calculate how poorly the model
        is performing by comparing what the model is predicting
        with the actual value it is supposed to output.
    metrics : list
        select the metrics (accuracy) for estimate the model.

    Returns
    -------
    model : class 'tensorflow.python.keras.\
            engine.Model'
        Model, Convolutional NN.

    '''
    model.summary()
    model.compile(optimizer=optimizer, loss=loss,metrics=metrics)
    return model

def model_fit(model, train_x, train_y, test_x, test_y, epochs, check):
    '''
    This function returns a plot of model losses and metrics choosen
    of train and validation test set data and evaluate the performance
    of the model to the test set data.
    Parameters:

    Parameters
    ----------
    model : class 'tensorflow.python.keras.\
            engine.sequential.Sequential'
    train_x : Vector, matrix, or array 
        training data.
    train_y : Vector, matrix, or array
        target data.
    test_x : Vector, matrix, or array
        test data.
    test_y : Vector, matrix, or array
        target test data.
    epochs : int
        select the epochs to control the number of complete
        passes through the training dataset.
    check : list
        callbacks to be called during training.

    Returns
    -------
    history : class 'tensorflow.python.keras.\
            engine.Model'
        Model, Convolutional NN.
    score : class 'ndarray'
        Loss and accuracy values.
    preds : class 'ndarray'
        prediction on test data.

    '''
    history = model.fit(train_x, train_y, validation_split = 0.4, epochs = epochs, callbacks=check)
    score = model.evaluate(test_x,test_y, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    preds = model.predict(test_x)
    plt.figure()
    plt.title('Loss and Val_Loss vs epochs = {epochs}'.format(epochs=epochs))
    plt.plot(history.history['loss'], label ='loss')
    plt.plot(history.history['val_loss'], label ='val_loss')
    plt.grid()
    plt.legend()
    plt.figure()
    plt.title('Accuracy and Val_Accuracy vs epochs = {epochs}'.format(epochs=epochs))
    plt.plot(history.history['accuracy'], label ='accuracy')
    plt.plot(history.history['val_accuracy'], label ='val_accuracy')
    plt.legend()
    plt.grid()
    plt.show()
    return history, score, preds