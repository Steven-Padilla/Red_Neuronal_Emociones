from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from mlxtend.plotting import plot_confusion_matrix 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from sklearn import metrics


#Rutas dataset
train_data_dir = './images/train'
val_data_dir = './images/validation'
datagen = ImageDataGenerator()

# Definimos algunos parámetros importantes
width_shape = 48
height_shape = 48
num_classes = 4
epochs = 50
batch_size = 32
class_names = ['angry','happy','neutral','sad']

data_gen_entrenamiento = datagen.flow_from_directory(
    train_data_dir, 
    target_size=(width_shape, height_shape),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True
)
data_gen_validacion = datagen.flow_from_directory(
    val_data_dir, 
    target_size=(width_shape, height_shape),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True
)

for imagen, etiqueta in data_gen_entrenamiento:
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(imagen[i])
    break
plt.show()

link='https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
mobilenettv2=hub.KerasLayer(link,input_shape=(224,224,3),trainable=False)
mobilenettv2.trainable=False


modelo = tf.keras.Sequential([
    mobilenettv2,
    tf.keras.layers.Dense( num_classes, activation='softmax'),
])

print(modelo.summary())

modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
tensorgraf=TensorBoard(log_dir='./logs/Denso')

historial= modelo.fit(
    data_gen_entrenamiento, 
    epochs=epochs,
    validation_data=data_gen_validacion,
    steps_per_epoch=data_gen_entrenamiento.n//batch_size,
    validation_steps=data_gen_validacion.n//batch_size,
    # callbacks=[tensorgraf]
    )
historial.save("pre_entrained_model.h5")

datagen = ImageDataGenerator()

data_gen_entrenamiento = datagen.flow_from_directory(
     val_data_dir,
    target_size=(width_shape, height_shape),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

predictions = historial.predict(data_gen_entrenamiento)
y_pred = np.argmax(predictions, axis=1)
y_real = data_gen_entrenamiento.classes

matc=confusion_matrix(y_real, y_pred)

plot_confusion_matrix(conf_mat=matc, figsize=(5,5), show_normed=False)
plt.tight_layout()

print(metrics.classification_report(y_real,y_pred, digits = 4))
# imgs=data_gen_entrenamiento[0][0]
# labels_aux=data_gen_entrenamiento[0][1]
# print(data_gen_entrenamiento[0][1])
# labels=[]
# for label in labels_aux:
#     if label[0] == 1:
#         labels.append(0)
#     else:
#         labels.append(1)

# test_loss , test_acc=modelo.evaluate(imgs,labels,verbose=1)
# print("*******************         Test accuracy : ", test_acc)
# pred = modelo.predict(imgs)
# class_names = ['angry','happy','neutral','sad']
# predic_aux=[]
# for data in pred:
#     index=class_names.index(class_names[np.argmax(data)])
#     predic_aux.append(index)
# predicted_class=np.array(predic_aux)
# matc = confusion_matrix(labels,predicted_class)
# plot_confusion_matrix(conf_mat=matc, figsize=(9,9),class_names=class_names, show_normed=False)
# plt.show()