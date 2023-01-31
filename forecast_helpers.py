import obspy.clients.fdsn
from obspy.core import read
from obspy.clients import seedlink
from obspy import UTCDateTime

import pandas as pd
import numpy as np

sec_in_hour = 60 * 60
sec_in_year = 2592000 * 12
year_to_start =  2008

HORIZON = 1
WINDOW_SIZE = 30 * 24 * 2

def pick_origin_for_hour(timestamp, catalog):
  time=UTCDateTime(timestamp)
  time.minute = 0
  time.second = 0
  start_time_filter = UTCDateTime(timestamp)
  if (start_time_filter.hour - 1) > 0 :
    start_time_filter.hour = start_time_filter.hour - 1
  start_time_filter_str = f"{start_time_filter.year}-{start_time_filter.month}-{start_time_filter.day}T{start_time_filter.hour}:59:59.59"

  end_time_filter = UTCDateTime(timestamp)
  end_time_filter_str = f"{end_time_filter.year}-{end_time_filter.month}-{end_time_filter.day}T{end_time_filter.hour}:59:59.59"
  filtered = catalog.filter(f"time >= {start_time_filter_str}", f"time <= {end_time_filter_str}")
  return filtered

def pack_timeline(catalog, timeline):
      
  if timeline==None:
    timeline = {}

  first_event_time = UTCDateTime.now().timestamp - 7 * 24 * 60 * 60  #  catalog.events[-1].origins[-1].time
  last_event_time = catalog.events[0].origins[0].time
  for hour in range(int(first_event_time), int(UTCDateTime.now().timestamp), int(sec_in_hour / 2)):
    events = pick_origin_for_hour(hour, catalog)
    if events.count() > 0 and len(events[0].magnitudes) > 0:
      timeline[hour] = events[0].magnitudes[0].mag
    else:
      timeline[hour] = 0
  return timeline

def get_labelled_windows(x, horizon=HORIZON):
  """
  Creates labels for windowed dataset.

  E.g. if horizon=1
  Input: [0, 1, 2, 3, 4, 5, 6, 7] -> Output: ([0, 1, 2, 3, 4, 5, 6], [7])
  """

  return x[:, :-horizon], x[:, -horizon:]

# Create function to view numpy arrays as windows
def make_windows(x, window_size=WINDOW_SIZE, horizon=HORIZON):
  """
  Turns a 1D array into a 2D array of sequential labelled windows of window_size with horizon size labels.
  """
  # 1. Create a window of a specific window_size (add teh horizon on the end for labelling later)
  window_step = np.expand_dims(np.arange(window_size + horizon), axis=0)
  # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
  window_indexes = window_step + np.expand_dims(np.arange(len(x) - (window_size + horizon - 1)), axis=0).T
  # 3. INdex on the target array (a time series) with 2D arrat nultiple window steps
  windowed_array = x[window_indexes]
  # 4. Get the labelled windows
  windows, labels = get_labelled_windows(windowed_array, horizon=horizon)

  return windows, labels

# Make the train/test splits
def make_train_test_splits(windows, labels, test_split=0.2):
  """
  Splits matching pairs of windows and labels into train and test splits
  """
  split_size = int(len(windows) * (1 - test_split)) # this will default to 80% train/20% test data
  train_windows = windows[:split_size]
  train_labels = labels[:split_size]
  test_windows = windows[split_size:]
  test_labels = labels[split_size:]
  return train_windows, test_windows, train_labels, test_labels

def pack_timeline(catalog, timeline):
  
  if timeline==None:
    timeline = {}

  first_event_time = UTCDateTime.now().timestamp - WINDOW_SIZE * sec_in_hour / 2 # a week ago in sec
  last_event_time = catalog.events[0].origins[0].time
  for hour in range(int(first_event_time), int(UTCDateTime.now().timestamp), int(sec_in_hour / 2)):
    events = pick_origin_for_hour(hour, catalog)
    if events.count() > 0 and len(events[0].magnitudes) > 0:
      timeline[hour] = events[0].magnitudes[0].mag
    else:
      timeline[hour] = 0
  return timeline 

def prepare_window(timeline):
  mags = np.fromiter(timeline.values(), dtype=np.float64)
  times = np.fromiter(timeline.keys(), dtype=np.float64)
  return times, mags

