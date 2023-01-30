import obspy.clients.fdsn
from obspy.core import read
from obspy.clients import seedlink
from obspy import UTCDateTime

import tensorflow as tf
import forecast_helpers as fh
import pandas as pd
import numpy as np

from datetime import datetime
import matplotlib.pyplot as plt

sec_in_hour = 60 * 60
year_to_start =  2008
WINDOW_SIZE= 30 * 24 * 2 # half-hours in a month
HORIZON=1 # next 30 minutes

def prepare_datasets():

    loaded_timeline = load_datasets("mags-1998-2022.csv")
    if loaded_timeline.empty:
        client=obspy.clients.fdsn.Client("EMSC")

        final_time = int(UTCDateTime.now().timestamp)
        initial_time = int(UTCDateTime(year_to_start, 1, 1, 0, 0).timestamp)
        mags_timeline = None
        step = int(sec_in_year / 6)

        for leap in range(initial_time, final_time, step):
          starttime = leap
          endtime = leap + step
          catalog = client.get_events(starttime=starttime,
                            endtime=endtime,
                            latitude=45.87410,
                            longitude=26.12301,
                            maxradius=5)

          mags_timeline = fh.pack_timeline(catalog, mags_timeline)
          print(f"A {catalog.count()} events happened in Romania region ever since {UTCDateTime(initial_time).date} till {UTCDateTime(leap).date}")
          timeline = pd.DataFrame(mags_timeline.items())
    else:
        timeline = loaded_timeline.drop(labels='Unnamed: 0', axis=1)
        
    timeline.columns= ['time', 'mag']

    timeline['time'] = timeline['time'].apply(lambda event: datetime.fromtimestamp(event))

    timeline_mag = timeline["mag"].to_numpy()
    timeline_time = timeline["time"].to_numpy()

    full_windows, full_labels = fh.make_windows(timeline_mag, window_size=WINDOW_SIZE, horizon=HORIZON)

    # Create train / test windows
    train_windows, test_windows, train_labels, test_labels = fh.make_train_test_splits(full_windows, full_labels)
    return train_windows, test_windows, train_labels, test_labels

def load_datasets(filename):
    timeline =pd.read_csv(filename)
    return timeline

def make_model(name):
    expand_dims_layer = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))

    input = tf.keras.layers.Input(shape=(WINDOW_SIZE,), dtype=tf.float64)
    x = expand_dims_layer(input)
    x = tf.keras.layers.LSTM(256, activation=tf.keras.activations.relu)(x) # ,  return_sequences=True)(x)
    # x  = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)(x)
    # x  = tf.keras.layers.Dense(128,  activation=tf.keras.activations.relu)(x)
    output = tf.keras.layers.Dense(HORIZON,  activation=tf.keras.activations.relu)(x)
    model = tf.keras.models.Model(input, output, name=name)

    model.compile(loss=tf.keras.losses.MeanSquaredLogarithmicError(reduction=tf.keras.losses.Reduction.SUM), 
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["msle"])
    
    return model

def fit_model(model, train_windows, test_windows, train_labels, test_labels, do_plot=False):
    checkpoints_path = f"{model.name}.ckpt"
    checkout_callback = tf.keras.callbacks.ModelCheckpoint(checkpoints_path,
        save_weights_only=True,
        save_best_only=True,
        verbose=1)
    tensorbaord_callback = tf.keras.callbacks.TensorBoard("logs")
    history = model.fit(train_windows,
                    train_labels,
                    epochs=5,
                    validation_data=(test_windows, test_labels),
                    batch_size=WINDOW_SIZE
 #                   callbacks=[checkout_callback, tensorbaord_callback]
                    )
    #model.load_weights(checkpoints_path)
    if do_plot:
        plt.figure(figsize=(10, 7))
        plt.plot(history.history["msle"])
        plt.show()
    return model

if __name__ == "__main__":
    print("Preparing datasets...")
    train_windows, test_windows, train_labels, test_labels = prepare_datasets()
    print("Building a model")
    model = make_model("EtQForecast-LSTM-week-horizon-30min")
    print("Fitting the model..")
    fit_model(model, train_windows, test_windows, train_labels, test_labels)
    model.save(f"{model.name}.h5")
    print("Evaluating...")
    model.evaluate(test_windows,  test_labels)
