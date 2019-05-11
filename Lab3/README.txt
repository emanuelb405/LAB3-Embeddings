folders:
files - various files that are needed for different operations, also results from preprocessing
aclImdb - contains the Imdb dataset for embeddings  http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
models -contains all models used, glove embedding, my own trained embedding, and three models from transfer learning
model_creation - code for generating the models for transfer learning
results - all results
test_split / train_split - test file of cats and dogs downloaded from kaggle: https://www.kaggle.com/c/dogs-vs-cats they are generated using the preprocessing_transferlearning.py

files: -transferlearning
vgg16_pretrained_trainable_train.py - Using VGG16 CNN with Imagenet and the convolutional layers set to trainable = true
vgg16_pretrained_untrainable_train.py - Using VGG16 CNN with Imagenet and the convolutional layers set to trainable = false
vgg16_untrained_trainable_train.py - vgg16 with randomly preset weights
pre_processing_transferlearning.py -does train test split on pictures

files: -embeddings
pre_processing_embediding.py - does pre processing to create pandas df out of dataset, collect vocabulary etc
*glove_embedding_trainable.py - uses glove embedding and is trainable
*glove_embedding_trainable.py -uses glove embedding and is not trainable
*own_embedding_trainable.py - uses glove embedding and is trainable
*own_embedding_trainable.py -uses glove embedding and is not trainable
train_cnn_no_textcleaning.py -simple implementation without pre processing

Also:
You need to download the glove embedding from http://nlp.stanford.edu/data/glove.6B.zip and put it into 
the models folder for the scripts to work
