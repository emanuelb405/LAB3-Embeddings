
# Word2Vec


from string import punctuation
from os import listdir
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

"""# Use pretrained word embedding"""

import pandas as pd

train = pd.read_csv('files/train.csv')
test = pd.read_csv('files/test.csv')
#load vocab
file = open('files/vocab.txt', 'r')
vocab = file.read()
file.close()

#remove everything from a sentence that is not in the vocab dictionary
def preprocessing(doc,vocab):
    #load vocab
    #file = open('vocab.txt', 'r')
    #vocab = file.read()
    #file.close()
    import string
    words = doc.split()
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words]
    stripped = [word.lower() for word in stripped]
    cleaned = list()
    #check if the word is in the vocab and only keep the ones that are
    for word in stripped:
      if word in vocab:
        cleaned.append(word)
      else:
        continue
    return ' '.join(cleaned)

train['cleaned_text']=train['text'].apply(preprocessing,args=[vocab])
test['cleaned_text']=test['text'].apply(preprocessing,args=[vocab])

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


X_train = train['cleaned_text'].values
X_test = test['cleaned_text'].values
y_train = train['sentiment'].values
y_test = test['sentiment'].values


tokenizer = Tokenizer()
max_length = max([len(s.split()) for s in X_train+X_test])

#get the X_train variable pre-processed
tokenizer.fit_on_texts(X_train) 

X_train = tokenizer.texts_to_sequences(X_train)
# pad sequences

X_train = pad_sequences(X_train, maxlen=max_length, padding='post')

#get the X_test variable pre-processed
X_test = tokenizer.texts_to_sequences(X_test)

# pad sequences
X_test = pad_sequences(X_test, maxlen=max_length, padding='post')


# define vocabulary size
vocab_size = len(tokenizer.word_index) + 1

#tokenize all the input data
#X_train_tokens =  tokenizer.texts_to_sequences(X_train)
#X_test_tokens = tokenizer.texts_to_sequences(X_test)

#create paddings with the same length
#X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')
#X_test_pad = pad_sequences(X_test_tokens, maxlen=max_length, padding='post')

#!wget http://nlp.stanford.edu/data/glove.6B.zip

#!unzip glove.6B.zip

#!mv glove.6B.100d.txt models/glove.6B.100d.txt

import numpy as np

def load_embedding(filename):
    # load embedding into memory, skip first line
    file = open(filename,'r')
    lines = file.readlines()[1:]
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
    return embedding

# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
	# total vocabulary size plus 0 for unknown words
	vocab_size = len(vocab) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = zeros((vocab_size, 100))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab.items():
		vector = embedding.get(word)
		if vector is not None:
			weight_matrix[i] = vector
	return weight_matrix

# load embedding from file
raw_embedding = load_embedding('models/glove.6B.100d.txt')
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)

embedding_layer = Embedding(len(tokenizer.word_index)+1, 100, weights=[embedding_vectors], input_length=max_length, trainable=True)

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau

checkpoint = ModelCheckpoint(
    'models/best_cnn_train_glove_embedding_trainable.model',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=False,
    period=1
)
earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=30,
    verbose=1,
    mode='auto'
)

csvlogger = CSVLogger(
    filename= "results/cnn_cnn_train_glove_embedding_trainable_training_csv.log",
    separator = ",",
    append = False
)

reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    verbose=1, 
    mode='auto'
)

callbacks = [checkpoint,csvlogger,reduce]

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.layers.embeddings import Embedding

# define model
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(X_train, y_train, batch_size=128, epochs=10, 
          validation_data=(X_test, y_test), verbose=1,callbacks=callbacks)
# evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %f' % (acc*100))

