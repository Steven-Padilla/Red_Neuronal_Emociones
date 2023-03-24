import os

# especifica el directorio en el que se encuentran tus archivos de imagen
directorio = './images/triste'

# recorre todos los archivos en el directorio
for filename in os.listdir(directorio):
    # si el archivo tiene extensión ".jpeg"
    if filename.endswith('.jpeg'):
        # construye el nuevo nombre de archivo cambiando la extensión a ".jpg"
        nuevo_nombre = filename.replace('.jpeg', '.jpg')
        # renombra el archivo
        os.rename(os.path.join(directorio, filename), os.path.join(directorio, nuevo_nombre))