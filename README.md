Set (and create) `user`, `password` and `port` in `config.py` in 'ml' folder for your MongoDB setup.


Requires Python <= Python3.7, personally using 3.6 and I know 3.8 doesn't work (because I tried)

Dataset: https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe

Shell script I personally use to run Docker-container of MongoDB and mount a volume on in my home directory is included (change username, password and user).
Add alias to `.bashrc`: `alias mongodb='sh mongodb.sh'`
and start with `mongodb start` and stop with `mongodb stop`



A Plotly's Dash webpage is included as wel (`web_app.py`).
This can only be used after both Dask and Spark have been run.