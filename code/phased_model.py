import tensorflow as tf

import sys
sys.path.append("/home/camilo/Desktop/plstm_tf2/models/layers")
from phased import PhasedLSTM

class PhasedSNForecastModel(tf.keras.Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.d1 = tf.keras.layers.Dropout(0.2)
        self.concat = tf.keras.layers.Concatenate()
        self.mask = tf.keras.layers.Masking(mask_value=-1.0)
        self._init_dense()
        self._init_recurrent()

    def normalization(self,inputs):
        max_all = tf.keras.backend.max(inputs)
        max_val = tf.keras.backend.max(inputs,axis=1)
        min_val = tf.keras.backend.min(tf.where(inputs > -1, inputs, max_all*tf.ones_like(inputs)),axis=1)

        stacked_min_val = tf.stack([min_val for i in range(inputs.shape[1])])
        stacked_min_val = tf.transpose(stacked_min_val, [1, 0, 2])
        stacked_max_val = tf.stack([max_val for i in range(inputs.shape[1])])
        stacked_max_val = tf.transpose(stacked_max_val, [1, 0, 2])

        inputs = (inputs - stacked_min_val) / (stacked_max_val-stacked_min_val)
        return min_val, max_val, inputs

    def denormalize(self,inputs, min_val, max_val):
        stacked_min_val = tf.stack([min_val for i in range(self.out_steps)])
        stacked_min_val = tf.transpose(stacked_min_val, [1, 0, 2])
        stacked_max_val = tf.stack([max_val for i in range(self.out_steps)])
        stacked_max_val = tf.transpose(stacked_max_val, [1, 0, 2])

        inputs = inputs * (stacked_max_val-stacked_min_val) + stacked_min_val
        return inputs

    def fowardpass(self, cells, states, denses, x, training=None):
        for i,cell in enumerate(cells):
            x, states[i] = cell(x, states=states[i],
                      training=training)
        for layer in denses:
            x = layer(x)

        return x, states

    def _warmup(self,rnn, denses, inputs):
        x, *state = rnn(inputs)
        for layer in denses:
            x = layer(x)
        return x, state

    def warmups(self,inputs):
        inputs = self.mask(inputs)
        prediction,states = self._warmup(self.rnn,self.denses,inputs)
        return prediction,states

    def _init_recurrent(self):
        cell1 = PhasedLSTM(self.units, activation="sigmoid", dropout=0.2)
        cell2 = PhasedLSTM(self.units//2, activation="sigmoid", dropout=0.2)

        self.cells = [cell1, cell2]
        self.rnn = tf.keras.layers.RNN(self.cells, return_state=True)

    def _init_dense(self):
        dense1 = tf.keras.layers.Dense(self.units//4, activation="sigmoid")
        # dense2 = tf.keras.layers.Dense(self.units//8, activation="sigmoid")
        # dense3 = tf.keras.layers.Dense(self.units//16, activation="sigmoid")

        out = tf.keras.layers.Dense(2, activation="sigmoid")
        self.denses = [dense1,self.d1, out ]#,dense2, self.d1,dense3, self.d1,out]

    def call(self, inputs, training=None):
        min_val, max_val, inputs =  self.normalization(inputs)
        #Creating empty tensors for predictions
        predictions = []
        prediction, states = self.warmups(inputs)

        #Saving first predictions
        predictions.append(prediction)

        for n in range(1, self.out_steps):
            prediction, states = self.fowardpass(self.cells, states, self.denses, prediction, training)
            predictions.append(prediction)

        #Stacking predictions
        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        predictions = self.denormalize(predictions, min_val, max_val)
        return predictions
