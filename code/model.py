import tensorflow as tf
from tensorflow import constant_initializer


def _exponential_initializer(min, max, dtype=None):
    def in_func(shape, dtype=dtype):
        initializer = tf.random_uniform_initializer(
                        tf.math.log(1.0),
                        tf.math.log(100.0)
                        )
        return tf.math.exp(initializer(shape))
    return in_func

class PhasedLSTM(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 leak_rate=0.001,
                 ratio_on=0.1,
                 period_init_min=0.0,
                 period_init_max=1000.0,
                 rec_activation = tf.math.sigmoid,
                 out_activation = tf.math.tanh,
                 name='plstm',
                 **kwargs):
        super(PhasedLSTM, self).__init__(name=name)

        self.state_size = [units,units] #This change
        self.output_size = units        #This change

        self.units = units
        self._leak = leak_rate
        self._ratio_on = ratio_on
        self._rec_activation = rec_activation
        self._out_activation = out_activation
        self.period_init_min = period_init_min
        self.period_init_max = period_init_max

        self.cell = tf.keras.layers.LSTMCell(units, **kwargs)

    def _get_cycle_ratio(self, time, phase, period):
        """Compute the cycle ratio in the dtype of the time."""
        phase_casted = tf.cast(phase, dtype=time.dtype)
        period_casted = tf.cast(period, dtype=time.dtype)
        time = tf.reshape(time, [tf.shape(time)[0],1]) #This change
        shifted_time = time - phase_casted
        cycle_ratio = (shifted_time%period_casted) / period_casted
        return tf.cast(cycle_ratio, dtype=tf.float32)

    def build(self, input_shape):
        self.period = self.add_weight(
                        name="period",
                        shape=[self.units],
                        initializer=_exponential_initializer(
                                            self.period_init_min,
                                            self.period_init_max),
                        trainable=True)

        self.phase = self.add_weight(name="phase",
                                     shape=[self.units],
                                     initializer=tf.random_uniform_initializer(
                                                         0.0,
                                                         self.period),
                                     trainable=True)
        self.ratio_on = self.add_weight(name="ratio_on",
                                        shape=[self.units],
                                        initializer=constant_initializer(self._ratio_on),
                                        trainable=True)

    def call(self, input, states):
        inputs, times = input[:,1:], input[:,0] #This change

        # =================================
        # CANDIDATE CELL AND HIDDEN STATE
        # =================================
        prev_hs, prev_cs = states
        output, (hs, cs) = self.cell(inputs, states)

        # =================================
        # TIME GATE
        # =================================
        cycle_ratio = self._get_cycle_ratio(times, self.phase, self.period)

        k_up = 2 * cycle_ratio / self.ratio_on
        k_down = 2 - k_up
        k_closed = self._leak * cycle_ratio

        k = tf.where(cycle_ratio < self.ratio_on, k_down, k_closed)
        k = tf.where(cycle_ratio < 0.5 * self.ratio_on, k_up, k)

        # =================================
        # UPDATE STATE USING TIME GATE VALUES
        # =================================
        new_h = k * hs + (1 - k) * prev_hs
        new_c = k * cs + (1 - k) * prev_cs

        return new_h, (new_h, new_c)


@tf.function
def sum_time(inputs):
    dts = inputs[:,:,0]
    dts = tf.where(dts < 0.0, 0.0, dts)
    times = tf.cumsum(dts,axis=1)
    times = tf.where(inputs[:,:,0] < 0.0, -1.0, times)
    times = tf.expand_dims(times,axis=-1)
    times_inputs = tf.concat([times,inputs], axis=-1)
    return times_inputs

class PhasedSNForecastModel(tf.keras.Model):
    def __init__(self, units, out_steps, features, dropout=0.5,name="autoencoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.out_steps = out_steps
        self.units = units
        self.features = features
        self.dropout = dropout
        self.concat = tf.keras.layers.Concatenate()
        self.mask = tf.keras.layers.Masking(mask_value=-1.0)
        self.sum_over_time = tf.keras.layers.Lambda(sum_time)
        self._init_dense()
        self._init_recurrent()



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
        prediction,states = self._warmup(self.rnn,self.denses,inputs)
        return prediction,states

    def _init_recurrent(self):
        cell1 = PhasedLSTM(self.units, dropout=self.dropout)
        self.cells = [cell1]
        self.rnn = tf.keras.layers.RNN(self.cells, return_state=True)

    def _init_dense(self):
        dense1 = tf.keras.layers.Dense(self.units//4, activation="tanh")
        out = tf.keras.layers.Dense(self.features, activation="linear")
        self.denses = [
                dense1,
                tf.keras.layers.Dropout(self.dropout),
                out]

    def call(self, inputs, training=None):
        inputs = self.sum_over_time(inputs)
        inputs = self.mask(inputs)
        
        last_times = inputs[:,-1,0]
        #Creating empty tensors for predictions
        predictions = []
        prediction, states = self.warmups(inputs)

        #Saving first predictions
        predictions.append(prediction)

        last_times = tf.expand_dims(last_times + prediction[:,0],axis=-1)
        prediction = tf.concat([prediction, last_times],axis=-1)

        for n in range(1, self.out_steps):
            prediction, states = self.fowardpass(self.cells, states, self.denses, prediction, training)
            predictions.append(prediction)
            last_times = tf.expand_dims(last_times[:,0] + prediction[:,0],axis=-1)
            prediction = tf.concat([prediction, last_times],axis=-1)


        #Stacking predictions
        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])

        # self.add_loss(self.mae(predicts, predictions))
        return predictions


class PhasedSNForecastProbabilisticIntervalModel(tf.keras.Model):
    def __init__(self, units, out_steps, model, dropout=0.5, penalization = 50):
        super().__init__()
        self.penalization = penalization
        self.out_steps = out_steps
        self.units = units
        self.dropout = dropout
        self.concat = tf.keras.layers.Concatenate()
        self.mask = tf.keras.layers.Masking(mask_value=-1.0)
        self.rnn = model.rnn
        self.cells = model.cells
        self.denses = model.denses
        for layer in self.cells:
            layer.traineable = False
        for layer in self.denses:
            layer.traineable = False
        self.rnn.traineable = False
        self.sum_over_time = tf.keras.layers.Lambda(sum_time)

        self._init_dense()

    def recurrent_pass(self,cells, states, x):
        for i,cell in enumerate(cells):
            x, states[i] = cell(x, states=states[i], training = False)
        return x, states

    def dense_pass(self, denses, x):
        for layer in denses:
            x = layer(x)
        return x

    def warmup(self,inputs):
        x, *states = self.rnn(inputs, training= False)
        return x,states


    def get_denses(self, features = 3):
        dense1 = tf.keras.layers.Dense(self.units//4, activation="tanh")
        dense2 = tf.keras.layers.Dense(self.units//8, activation="tanh")

        out = tf.keras.layers.Dense(features, activation="linear")
        return [
        dense1,
        tf.keras.layers.Dropout(self.dropout),
        dense2,
        tf.keras.layers.Dropout(self.dropout),
        out]

    def _init_dense(self):
        self.lower = self.get_denses(1)
        self.upper = self.get_denses(1)

    def get_PICW(self, lower, upper):
        penalization = tf.expand_dims(tf.cast(self.penalization, upper.dtype), 0)
        range = tf.where(upper > lower, upper-lower, self.penalization * (lower - upper))
        return tf.reduce_mean(range,axis=-1)

    def call(self, inputs, training = False):
        inputs = self.sum_over_time(inputs)
        inputs = self.mask(inputs)
        last_times = inputs[:,-1,0]        
       
        #Creating empty tensors for predictions
        predictions = []
        upper_preds = []
        lower_preds = []

        x, states = self.warmup(inputs)
        prediction = self.dense_pass(self.denses, x)
        upper_pred = self.dense_pass(self.upper, x)
        lower_pred = self.dense_pass(self.lower, x)

        #Saving first predictions
        predictions.append(prediction)
        upper_preds.append(upper_pred)
        lower_preds.append(lower_pred)

        
        last_times = tf.expand_dims(last_times + prediction[:,0],axis=-1)
        prediction = tf.concat([prediction, last_times],axis=-1)
        for n in range(1, self.out_steps):
            x, states = self.recurrent_pass(self.cells, states, prediction)
            prediction = self.dense_pass(self.denses, x)
            predictions.append(prediction)

            last_times = tf.expand_dims(last_times[:,0] + prediction[:,0],axis=-1)
            prediction = tf.concat([prediction, last_times],axis=-1)
            
            upper_pred = self.dense_pass(self.upper, x)
            lower_pred = self.dense_pass(self.lower, x)


            upper_preds.append(upper_pred)
            lower_preds.append(lower_pred)

        #Stacking predictions
        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])

        upper_preds = tf.stack(upper_preds)
        upper_preds = tf.transpose(upper_preds, [1, 0, 2])

        lower_preds = tf.stack(lower_preds)
        lower_preds = tf.transpose(lower_preds, [1, 0, 2])
        PICW = self.get_PICW(lower_preds,upper_preds)
        self.add_metric(PICW,name="PICW")
        self.add_loss(PICW)

        return {
            "prediction":predictions,
            "lower": lower_preds,
            "upper": upper_preds
            }
