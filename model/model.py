from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model
#Encoder model

#Image feature layers
def CNNtoRNN(vocab_size, max_length):

    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation="relu")(fe1)
    fe2 = BatchNormalization()(fe2)

    #Sequence feature layers

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.4)(se1)
    se3 = LSTM(256)(se2)
    se4 = Dropout(0.4)(se3)
    se5 = BatchNormalization()(se4)

    #decoder Model

    decoder1 = add([fe2,se5])
    decoder2 = Dense(256, activation="relu")(decoder1)
    decoder2 = BatchNormalization()(decoder2)
    outputs = Dense(vocab_size, activation="softmax")(decoder2)

    model = Model(inputs = [inputs1, inputs2], outputs = outputs)
    
    model.compile(loss = SparseCategoricalCrossentropy(), optimizer="adam")

    return model