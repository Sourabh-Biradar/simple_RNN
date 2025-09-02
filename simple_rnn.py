# Deep Learning

# simple RNN

# mini project 5 : Movie reviews & sentiment analysis

import pandas as pd

df = pd.read_csv('IMDb_dataset.csv')

df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
X = df['review']
y = df['sentiment']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# preprocessing

import tensorflow
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.utils import pad_sequences

vocab_size = 10000
max_len = 200

tokenizer = TextVectorization(max_tokens=vocab_size,output_sequence_length=max_len)
tokenizer.adapt(X_train)

# model builting

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense,Input

embedding_dim = 100

model = Sequential()

model.add(Input(shape=(1,), dtype=tf.string))

model.add(tokenizer)

model.add(Embedding(input_dim=vocab_size,output_dim=embedding_dim))

model.add(SimpleRNN(32,activation='tanh',return_sequences=False))

model.add(Dense(1,activation='sigmoid'))

model.summary()

# model compile
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# model fitting

val_data = (X_test,y_test)

history = model.fit(
    X_train,y_train,
    validation_data=val_data,
    epochs=100,
    batch_size=64,
    callbacks=[early_stop]
)

# save & load model
from tensorflow import keras

model.save('simple_rnn_.keras')

model = keras.models.load_model('simple_rnn_.keras')

def predict_review(review):
    
    review_tensor = tf.convert_to_tensor([review]) 
    
    prediction = model.predict(review_tensor)
    
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    
    return sentiment, float(prediction[0][0])

sentiment,pred = predict_review('fantastic movie,trilling plot')

print(f'{sentiment} , {(pred*100)}')

sentiment_,pred_ = predict_review('bad,waste of money')

print(f'{sentiment_} , {(pred_*100)}')