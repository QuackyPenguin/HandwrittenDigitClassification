import numpy as np
from keras.models import load_model
from PIL import Image

# loading
model = load_model('model.h5')

# image preprocessing
img = Image.open('digit7.png').convert('L')
img = img.resize((28, 28))
img_arr = np.array(img)
img_arr = img_arr.reshape(1, 28, 28, 1)
img_arr = img_arr.astype('float32')
img_arr /= 255.0

# prediction
y_pred = model.predict(img_arr)
digit = np.argmax(y_pred, axis=1)[0]
print('The predicted digit is: ', digit)
