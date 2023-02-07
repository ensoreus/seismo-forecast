import obspy.clients.fdsn
from obspy.core import read
from obspy.clients import seedlink
from obspy import UTCDateTime

import tensorflow as tf
import forecast_helpers as fh
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler

sec_in_hour = 60 * 60

WINDOW_SIZE=14*24*2 # half-hours in a month
HORIZON=1 # next 30 minutes

predictions = {}

def fetch_events(start_time, end_time):
    client=obspy.clients.fdsn.Client("EMSC")
    try:
        catalog = client.get_events(
                        starttime=start_time,
                        endtime=end_time,
                        latitude=45.87410,
                        longitude=26.12301,
                        maxradius=5
        )
    except:
        return obspy.core.event.catalog.Catalog()
    return catalog

def predict():
    now = UTCDateTime.now()
    sec_half_an_hour = 30 * 60
    start_time = now - sec_half_an_hour * WINDOW_SIZE
    end_time = now
    pre_history_catalog = fetch_events(start_time, end_time)
    timeline = fh.pack_timeline(pre_history_catalog, window_size=WINDOW_SIZE)
    model = tf.keras.models.load_model("EtQForecast-LSTM-week-horizon-30min.h5")
    times, mags = fh.prepare_window(timeline)
    mags= tf.reshape(mags, [1, WINDOW_SIZE])
    pred_mag = model.predict(mags)
    return pred_mag

def check_predicted(pred_mag):
    
    step_before, now, _ = time_ranges()
    # start_time_filter_str = f"{step_before.year}-{step_before.month}-{step_before.day}T{step_before.hour}:{step_before.minutes}:{step_before.seconds}.00"
    # end_time_filter_str = f"{now.year}-{now.month}-{now.day}T{now.hour}:{now.minutes}:{now.seconds}.00"
    catalog = fetch_events(step_before, now)
    # actual_event_last_step = catalog.filter(f"time >= {start_time_filter_str}", f"time <= {end_time_filter_str}")
    
    if catalog.count() >  0 and pred_mag > 0:
        last_event_mag = catalog.events[-1].magnitudes[-1].mag
        print(f"Indeed, {last_event_mag} which true:{last_event_mag}, pred:{pred_mag}")
    elif catalog.count() > 0 and pred_mag == 0:
        last_event_mag = catalog.events[-1].magnitudes[-1].mag
        print("Miss: undetected {last_event_mag}")
    elif catalog.count() ==  0 and pred_mag > 0:
        print("Miss: false detection")
    
def time_ranges():
    now = UTCDateTime.now()
    step_before = UTCDateTime(now.timestamp - 30 * 60)
    step_after = UTCDateTime(now.timestamp + 30 * 60)
    return step_before,  now,  step_after

def run_predictor():
    pred_mag =  predict()
    step_before,  now, step_after = time_ranges()
    if pred_mag > 0:
        print(f"Predicted:{pred_mag} from {now.date()} to {step_after.date()}")
        prev_pred = predictions[step_before.date]
        check_predicted(prev_pred)
        predictions[now.date] =pred_mag
    else:
        predictions[now.date] = 0
        
if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(run_predictor, 'interval', minutes=30)
    try:
        scheduler.start()
    except (KeyboardInterrupt,  SystemError):
        pass
 
