import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense

from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras import optimizers

from tensorflow.keras.layers import InputLayer

def NN_model_2_layer(train_set, neurans_per_layer = 36):
    model=Sequential()
    model.add(Dense(neurans_per_layer, input_shape=(train_set.shape[1],)))
    model.add(Dense(neurans_per_layer, activation='sigmoid'))
    model.add(Dense(4, activation='softmax'))

    learning_rate = 0.001
    optimizer = optimizers.Adam(learning_rate)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

def NN_model_1_layer(train_set, neurans_per_layer = 36):
    tf.random.set_seed(124) 
    model=Sequential()
    model.add(Dense(neurans_per_layer, input_shape=(train_set.shape[1],)))   
    model.add(Dense(4, activation='softmax'))  

    learning_rate = 0.001
    optimizer = optimizers.Adam(learning_rate)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

def NN_model_4_layer(train_set, neurans_per_layer = 36):
    model=Sequential()
    model.add(Dense(neurans_per_layer, input_shape=(train_set.shape[1],)))   
    model.add(Dense(neurans_per_layer, activation='sigmoid'))
    model.add(Dense(neurans_per_layer, activation='sigmoid'))
    model.add(Dense(neurans_per_layer, activation='sigmoid'))
    model.add(Dense(4, activation='softmax'))      

    learning_rate = 0.001
    optimizer = optimizers.Adam(learning_rate)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

def run(train_set,
        train_labels,
        val_set,
        val_labels,
        model,
        Epochs = 500,
        Batch_size = 100):
    print('Here is a summary of this model: ')
    model.summary()

    with tf.device('/CPU:0'):
        history = model.fit(
            train_set,
            train_labels,
            batch_size=Batch_size,
            epochs=Epochs,
            verbose=0,
            shuffle=True,
            steps_per_epoch = int(train_set.shape[0]/Batch_size),
            validation_data = (val_set, val_labels))
    return model, history