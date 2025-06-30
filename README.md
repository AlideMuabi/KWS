# KWS
## Objetivo: A través de la función Keyword Spotting, modificar el brillo de la pantalla de una laptop utilizando comandos de voz.

## Procedimiento:
Los entrenamientos se llevaron a cabo en una computadora de escritorio con un procesador Ryzen 5500, 16 GB RAM, GPU AMD (no se utilizó) y con SO Windows 11, apoyándose de Anaconda 3.

### Base de Datos:
Se realizó una colección de audios con las siguientes características:
- 16000 muestras a 16 kHz, en mono.
- Se dividieron en carpetas. Una llamada "sube" y otra llamada "baja", correspondiente al contenido del auido.
- Se grabaron en ambientes lo más ocntrolados posibles, sin ruido externo destacable.
- Fueron un total de 245 audios en "sube" y 245 en "baja", además de una carpeta con ruido de fondo llamada "descarga" con 386 audios de las mismas características.

## Arquitectura:
Nuestro modelo (nombrándolo "Modelo_5") presenta la siguiente arquitectura
![arquitectura](https://github.com/user-attachments/assets/e7de948f-65b3-4b07-b3e9-cc5599f2a44d)

## Graficas y matrices.
![Graficas_Entrenamiento](https://github.com/user-attachments/assets/c9185caa-886d-462a-9ab8-6ba58e40ffef)

![Matriz_entrenamiento](https://github.com/user-attachments/assets/90dc3237-ad47-4c28-a0db-f651d3c2ba05)

![Matriz_despliegue](https://github.com/user-attachments/assets/b2fbb8dc-b0e7-4d68-98a0-70e7365576e0)

### Métricas de desempeño
![Metricas_desempeño](https://github.com/user-attachments/assets/273b330b-8928-414e-8510-1a6e4894821f)

## Conclusión
Se logró llegar al objetivo una vez puesto a prueba el scrpit brillo.py con las paqueterías necesarias instaladas.
