## train with vgg with imagenet weights and trainable  vgg16_pretrained_trainable"""

from keras.models import model_from_json
weights_file = "models/vgg16_pretrained_trainable.hdf5"
json_file = open('models/vgg16_pretrained_trainable.json','r')
nn_json = json_file.read()
#json_file.close()
model = model_from_json(nn_json)
model.load_weights(weights_file)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau

checkpoint = ModelCheckpoint(
    'models/vgg16_pretrained_trainable.model',
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
    filename= "results/vgg16_pretrained_trainable_training_csv.log",
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

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator()

training_set = train_datagen.flow_from_directory('train_split',
                                                  target_size=(128, 128),
                                                  batch_size=32,
                                                  class_mode='binary',
                                                  shuffle = True)

test_set = test_datagen.flow_from_directory('test_split',
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='binary',                                         
                                            shuffle = True)

history = model.fit_generator(training_set,
                    steps_per_epoch=8375/32,
                    epochs=50,
                    validation_data=test_set,
                    callbacks = callbacks,
                    validation_steps=4125/32)