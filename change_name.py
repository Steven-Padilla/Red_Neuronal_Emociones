import os

# Ruta al directorio de imágenes
ruta_imagenes = './images/triste'

# Prefijo para los nuevos nombres de las imágenes
prefijo = 'imagen_'

# Obtener la lista de archivos en el directorio
archivos = os.listdir(ruta_imagenes)

# Iterar a través de cada archivo en el directorio
for i, archivo in enumerate(archivos):
    # Obtener la extensión del archivo
    extension = os.path.splitext(archivo)[1]

    # Crear el nuevo nombre del archivo
    nuevo_nombre = prefijo + str(i+1) + extension

    # Renombrar el archivo
    os.rename(os.path.join(ruta_imagenes, archivo), os.path.join(ruta_imagenes, nuevo_nombre))
