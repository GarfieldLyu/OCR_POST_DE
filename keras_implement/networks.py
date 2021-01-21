import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import LSTM, Average,Reshape, Conv1D, Conv2D, Convolution2D, RepeatVector, Dropout, Bidirectional,\
    Embedding,Lambda, Permute, Flatten, Dense, merge, Input, Concatenate, Multiply, Activation, dot, TimeDistributed
from gated_cnn import GatedConvBlock


def custom_loss(input_tensor, output_tensor, alpha):
    beta = K.cast(K.equal(input_tensor, output_tensor), dtype='float32')

    def loss(y_true, y_pred):
        return K.categorical_crossentropy(y_true, y_pred) * (1-alpha*beta)
    return loss


class Encoder:
    def __init__(self, units, embedding_dim, n_features, max_length):
        inp = Input(shape=(max_length,), name='encoder_input')
        enc_embedding = Embedding(n_features, embedding_dim, trainable=True)(inp)
        encoder, f_h, f_c, b_h, b_c = Bidirectional(LSTM(int(units / 2), return_sequences=True, return_state=True))(
            enc_embedding)
        state_h = Concatenate()([f_h, b_h])
        state_c = Concatenate()([f_c, b_c])
        self.states = [state_h, state_c]
        self.enc_embedding = enc_embedding
        self.inp = inp
        self.enc_seq = encoder


def define_cnn_rnn(units, embedding_dim, n_features, max_length, kernel, custom=0, alpha=0):
    encoder = Encoder(units, embedding_dim, n_features, max_length)
    # decoder
    decoder_inp = Input(shape=(max_length,), name='decoder_input')
    embedding = Embedding(n_features, embedding_dim, trainable=True)(decoder_inp)
    decoder = LSTM(units, return_sequences=True)(embedding, initial_state=encoder.states)

    # attention
    score = []
    attention = dot([decoder, encoder.enc_seq], axes=[2, 2])
    attention = Activation(tf.nn.softmax)(attention)
    score.append(attention)
    for i, k in enumerate(kernel):
        cnnSeq = Conv1D(filters=units, kernel_size=k, input_shape=(max_length, n_features), padding='same',
                        activation='relu')(encoder.enc_embedding)
        if i == 0:  # first layer
            cnn_c = cnnSeq
        attention = dot([decoder, cnnSeq], axes=[2, 2])
        attention = Activation(tf.nn.softmax)(attention)
        score.append(attention)

    score = Average()(score)
    context_0 = dot([score, cnn_c], axes=[2, 1])
    context_1 = dot([score, encoder.enc_seq], axes=[2, 1])
    context = Concatenate()([context_0, context_1])
    decoder_contex = Concatenate()([context, decoder])
    # output
    decoder_outputs = TimeDistributed(Dense(units, activation='tanh'))(decoder_contex)
    decoder_outputs = TimeDistributed(Dense(n_features))(decoder_outputs)
    decoder_outputs = Activation(tf.nn.softmax)(decoder_outputs)
    model = Model([encoder.inp, decoder_inp], decoder_outputs)

    if custom == 0:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    else:
        model.compile(loss=custom_loss(encoder.inp, decoder_inp, alpha), optimizer='adam', metrics=['acc'])
    model.summary()
    return model


def define_simple_enc_dec(units, embedding_dim, n_features, max_length, custom=0, alpha=0):
    # a rnn attention encoder-decoder model without cnn unit.
    ''' encoder input'''
    main_input = Input(shape=(max_length,), name='main_input')
    embedding = Embedding(n_features, embedding_dim, trainable=True, mask_zero=True)(main_input)
    encoder, f_h, f_c, b_h, b_c = Bidirectional(LSTM(int(units / 2), return_sequences=True, return_state=True))(
        embedding)
    state_h = Concatenate()([f_h, b_h])
    state_c = Concatenate()([f_c, b_c])
    encoder_states = [state_h, state_c]

    '''decoder input '''
    decoder_input = Input(shape=(max_length,), name='decoder_input')
    embedding = Embedding(n_features, units, trainable=True)(decoder_input)
    decoder = LSTM(units, return_sequences=True)(embedding, initial_state=encoder_states)

    ''' attention'''
    attention = dot([decoder, encoder], axes=[2, 2])
    attention = Activation(tf.nn.softmax)(attention)
    contex = dot([attention, encoder], axes=[2, 1])
    decoder_contex = Concatenate()([contex, decoder])

    decoder_contex = Dropout(0.3)(decoder_contex)
    decoder_outputs = TimeDistributed(Dense(n_features))(decoder_contex)  # combined contex
    decoder_outputs = Activation(tf.nn.softmax)(decoder_outputs)
    model = Model([main_input, decoder_input], decoder_outputs)

    if custom:
        model.compile(loss=custom_loss(main_input, decoder_input, alpha), optimizer='adam', metrics=['acc'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model


def define_glu_rnn_single(units, embedding_dim, n_features, max_length, kernel, layer_num,
                          all_blocks=False, custom=False, alpha=0.1):  # without multi-layer attention
    encoder = Encoder(units, embedding_dim, n_features, max_length)
    states = encoder.states
    enc_embedding = encoder.enc_embedding
    inp = encoder.inp
    # decoder
    decoder_inp = Input(shape=(max_length,), name='decoder_input')
    embedding = Embedding(n_features, embedding_dim, trainable=True)(decoder_inp)
    decoder = LSTM(units, return_sequences=True)(embedding, initial_state=states)

    cnn_inp = Dense(units)(enc_embedding)
    cnn_inp = Reshape((1, max_length, units))(cnn_inp)
    cnnLayer = Conv2D(2 * units, (kernel, units), padding='same', activation='relu',
                      input_shape=(1, max_length, units))
    cnnGlu = GatedConvBlock(cnnLayer, conv_num=layer_num, return_blocks=all_blocks)(cnn_inp)
    cnnGlu = Reshape((max_length, units))(cnnGlu)
    atten = dot([decoder, cnnGlu], axes=[-1, -1])
    atten = Activation(tf.nn.softmax)(atten)
    contex = dot([atten, cnnGlu], axes=[2, 1])
    decoder_contex = Concatenate()([contex, decoder])

    # output
    decoder_outputs = TimeDistributed(Dense(units, activation='tanh'))(decoder_contex)
    decoder_outputs = TimeDistributed(Dense(n_features))(decoder_outputs)
    decoder_outputs = Activation(tf.nn.softmax)(decoder_outputs)

    model = Model([inp, decoder_inp], decoder_outputs)
    if custom == 0:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    else:
        model.compile(loss=custom_loss(inp, decoder_inp, alpha), optimizer='adam', metrics=['acc'])
    model.summary()
    return model


def define_glu_rnn(units, embedding_dim, n_features, max_length, kernel, layer_num, all_blocks, custom=0,
                   alpha=0.1):  # kernel should be integer
    encoder = Encoder(units, embedding_dim, n_features, max_length)
    states = encoder.states
    enc_embedding = encoder.enc_embedding
    inp = encoder.inp

    # decoder
    decoder_inp = Input(shape=(max_length,), name='decoder_input')
    embedding = Embedding(n_features, embedding_dim, trainable=True)(decoder_inp)
    decoder = LSTM(units, return_sequences=True)(embedding, initial_state=states)

    # attention
    cnn_inp = Dense(units)(enc_embedding)
    cnn_inp = Reshape((1, max_length, units))(cnn_inp)

    cnnLayer = Conv2D(2 * units, (kernel, units), padding='same', activation='relu',
                      input_shape=(1, max_length, units))
    cnnGlu = GatedConvBlock(cnnLayer, conv_num=layer_num, return_blocks=all_blocks)(cnn_inp)

    decoder_origin = decoder
    decoder = Lambda(lambda x: tf.stack([x] * (layer_num + 1)))(decoder)
    decoder = Lambda(lambda x: tf.transpose(x, [1, 0, 2, 3]))(decoder)  # batch, block, T, units
    # decoder = K.reshape(decoder, (layer_num+1, max_length, units))

    atten = dot([decoder, cnnGlu], axes=[-1, -1])
    atten = Activation(tf.nn.softmax)(atten)
    score = Lambda(lambda x: K.mean(x, axis=1))(atten)  # batch, timestep, timestep
    score = Lambda(lambda x: K.reshape(x, (-1, max_length, max_length)))(score)
    cnn_0 = Lambda(lambda x: x[:, 0, :, :])(cnnGlu)
    contex = dot([score, cnn_0], axes=[2, 1])
    decoder_contex = Concatenate()([contex, decoder_origin])
    decoder_outputs = TimeDistributed(Dense(units, activation='tanh'))(decoder_contex)
    decoder_outputs = TimeDistributed(Dense(n_features))(decoder_outputs)
    decoder_outputs = Activation(tf.nn.softmax)(decoder_outputs)

    model = Model([inp, decoder_inp], decoder_outputs)
    if custom == 0:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    else:
        model.compile(loss=custom_loss(encoder.inp, decoder_inp, alpha), optimizer='adam', metrics=['acc'])
    model.summary()
    return model