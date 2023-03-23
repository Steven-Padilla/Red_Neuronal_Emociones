import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
model=tf.keras.models.load_model("modelFEC.h5")

class_names = ['angry','happy','neutral','sad']

faces = []

# Cargamos una imagen del directorio
imaget_path = "./images/validation/happy/1787.jpg"
imagen=cv2.imread(imaget_path)
# Redimensionamos la imagen y convertimos a gray
face = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
face = cv2.resize(face, (48, 48))
face2 = img_to_array(face)
face2 = np.expand_dims(face2,axis=0)

faces.append(face2)

# El modelo estima la predicción
preds = model.predict(faces)

print(f'You are {class_names[np.argmax(preds)]}')
plt.imshow(cv2.cvtColor(np.asarray(face),cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
