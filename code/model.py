import tensorflow as tf

class SNForecastModel(tf.keras.Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self._create_cells()
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.recurrent_cells, return_state=True)
        self.dense_dt = tf.keras.layers.Dense(1, activation="tanh")
        self.dense_mag = tf.keras.layers.Dense(2, activation="tanh")
        self.dense_fid = tf.keras.layers.Dense(1, activation="sigmoid")
        self.concat = tf.keras.layers.Concatenate()


    def _create_cells(self):
        cell1 = tf.keras.layers.LSTMCell(self.units)
        self.recurrent_cells = [cell1]

    def call(self, inputs, training=None):
        #Splitting inputs
        dt_in = inputs["dt_in"]
        mag_in = inputs["mag_in"]
        fid_in = inputs["fid_in"]

        #Creating empty tensors for predictions
        predictions_dt = []
        predictions_mag = []
        predictions_fid = []

        #Concatenating inputs
        concat_inputs = self.concat([dt_in, mag_in, fid_in])
        x, *states =  self.lstm_rnn(concat_inputs)

        prediction_dt = self.dense_dt(x)
        prediction_mag = self.dense_mag(x)
        prediction_fid = self.dense_fid(x)

        #Saving first predictions
        predictions_dt.append(prediction_dt)
        predictions_mag.append(prediction_mag)
        predictions_fid.append(prediction_fid)

        #Using prediction to next horizon
        for n in range(1, self.out_steps):
            x = self.concat([prediction_dt,prediction_mag,prediction_fid])

            #Iterating over recurrent internal layers
            for i,cell in enumerate(self.recurrent_cells):
                x, states[i] = cell(x, states=states[i],
                                      training=training)

            #Making predictions
            prediction_dt = self.dense_dt(x)
            prediction_mag = self.dense_mag(x)
            prediction_fid = self.dense_fid(x)

            predictions_dt.append(prediction_dt)
            predictions_mag.append(prediction_mag)
            predictions_fid.append(prediction_fid)

        #Stacking predictions
        predictions_dt = tf.stack(predictions_dt)
        predictions_dt = tf.transpose(predictions_dt, [1, 0, 2])
        predictions_mag = tf.stack(predictions_mag)
        predictions_mag = tf.transpose(predictions_mag, [1, 0, 2])
        predictions_fid = tf.stack(predictions_fid)
        predictions_fid = tf.transpose(predictions_fid, [1, 0, 2])

        return {"dt_out": predictions_dt,"mag_out": predictions_mag,"fid_out":predictions_fid}
