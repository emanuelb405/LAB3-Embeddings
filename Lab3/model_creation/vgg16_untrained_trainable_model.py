### VGG16 without pretrained weights  vgg16_untrained_trainable


## Implementation without transfer learning VGG16 model
from keras.models import Sequential
from keras.layers import Flatten,Dense
from keras.applications.vgg16 import VGG16

imgsize = 128
base_model = VGG16(include_top=False,
                  input_shape = (imgsize,imgsize,3),
                  weights = None)

for layer in base_model.layers:
    layer.trainable = True
    
for layer in base_model.layers:
    print(layer,layer.trainable)

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(units = 128,activation='relu'))
model.add(Dense(units = 1, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.models import model_from_json
nn_json = model.to_json()
with open('models/vgg16_untrained_trainable.json', 'w') as json_file:
        json_file.write(nn_json)
weights_file = "models/vgg16_untrained_trainable.hdf5"
model.save_weights(weights_file,overwrite=True)