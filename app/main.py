#Import Libraries
import numpy as np

import uvicorn
from typing import List
from pydantic import BaseModel

from fastapi import FastAPI

import keras
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing import sequence
from pathlib import Path
import os
import wandb

#Assign the Trained Model path
best_model = wandb.restore('model-best.h5', run_path="sivakumarrajendran/imdb_sentiment_classification/31eallt6")
#base_model="C:/Users/sivak/W&B - IMDB/model_best.h5"
print("Model name : ", best_model.name)

#Load the model
model = tf.keras.models.load_model(best_model.name)

#Instanciate fastapi
app = FastAPI()

# Declare the Request/ Input string in Pydantic way
class Reviews(BaseModel):
    review: str


@app.get('/')
def index():
    return {'message': 'IMDb Reviews Classification API!'}

@app.post('/predict')
def predict_review(data: Reviews):
    """ FastAPI 
    Args:
        data (Reviews): json file 
    Returns:
        prediction: probability of review being positive
    """
    data = data.dict()
    review = data['review']

    word_index = keras.datasets.imdb.get_word_index()
    
    test = []
    for i in review.split():
        test.append(word_index.get(i))
    while len(test) < 500:
        test = [0] + test
    sample_sent = np.stack(test)
    sample_sent = np.expand_dims(sample_sent,axis=0)
    prediction = model.predict(sample_sent)
    
    return {
        'prediction': prediction.tolist()[0][0]
    }

if __name__ == '__main__':
   uvicorn.run(app, host='127.0.0.1', port=8002)

#Open the URL http://127.0.0.1/docs and explore
