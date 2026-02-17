from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Activation

def build_cnn_model():
    model = Sequential()
    model.add(Conv1D(64, 5, padding='same', input_shape=(40, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Activation('softmax'))
    return model
