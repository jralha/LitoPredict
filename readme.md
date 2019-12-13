Basic XGBoost model to classify lithology based on model trained on Lapa Field wells.

Can only classify between Barra Velha and Itapema formations currently.

Performance is at 84% accuracy (Recalls for each class around 80%).

Currently a version of the repo hosted on Heroku, inputs and outputs are csv files. Used XGBoost as the prediction model.

If you try to run this repo, password and username are not secured and are set in app.py, use at your own risk, they're there just as a semblance of security.

The pickle file contains the trained model and shouldn't required anything more than XGBoost to open despite what requirements.txt says.
