#Import libraries
import logging
import tensorflow as tf
from keras.datasets import imdb
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

#Load IMDB Dataset from tf.keras.datasets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
    path="imdb.npz",
    num_words=None,
    skip_top=0,
    maxlen=None,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3,
)

#Know the length of Train and Test dataset
print("No. of Records in Train Dataset: ", len(x_train))
print("No. of Records in Test Dataset: ", len(x_test))

#Calculate Word-Index
word2index = imdb.get_word_index()
index2word = dict([(i,w) for (w,i) in word2index.items()])

#Limit the dataset to top n words
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

#Initiate wandb
import wandb
from wandb.keras import WandbCallback
wandb.login()

#Create Model
tf.keras.backend.clear_session()
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

run = wandb.init(project = "imdb_sentiment_classification",
                 config={"model":"mlp", 
                        "max_words":500,
                        "top words": 5000,
                        "loss": "binary_crossentropy",
                        "optimizer": "adam",
                        "metrics": "accuracy",
                         "verbose": 2,
                        "epochs": 5,
                        "batch_size": 64})

config = wandb.config

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#Run the Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=config.epochs, batch_size=config.batch_size, verbose=config.verbose,callbacks=[WandbCallback()])

#Save Model
api = wandb.Api()
run = api.run("sivakumarrajendran/imdb_sentiment_classification/31eallt6") #Path hv to be parametrize
run.file("model-best.h5").download()
