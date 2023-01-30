# seismo-forecast
A naïve attempt to predict earthquake magnitude in Romania region. With 30 minutes horizon. 
## Prerequisites
* Obspy (https://obspy.org)
* TensorFlow (https://tensorflow.org)
* matplotlib
* datetime
* numpy
* pandas

## Usage
### Train a model
At start, you have to create and train a model. Here we have a `train.py` script. 
You can use prepared `mags-1998-2022.csv` file which contains all the events around Vrancha mountains in a 5 degrees radius. 
Or you can use just make your own magnitudes history with latest events by removing `mags-1998-2022.csv`.
To run building and training model, run
`python train.py`
