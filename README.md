# EvCam-IMU-GPS

Visor SilkyEvCam -  Lab Control

Autores: Eduardo Llantén - Pablo Garrido

Institucion: Estudiante de Licenciatura en Fisica Estado del Proyecto: v10


# Descripción

Este software constituye una plataforma avanzada de adquisicion y procesamiento de datos diseñada para experimentos de fisica experimental. El sistema permite la captura sincronizada de flujo de eventos proveniente de sensores de vision basados en eventos (Prophesee/Metavision) y telemetria inercial (IMU/GPS) mediante microcontroladores ESP32.

El programa ha sido pensado para su ejecución en entornos Linux y Windows, integrando hilos de ejecucion (QThreads) para garantizar una baja latencia en la visualizacion y una escritura de datos eficiente.


# Especificaciones Tecnicas:

1. Control de Visión por Eventos

Soporte para adquisición dual sincronizada (Master/Slave) mediante cables sync.

Gestion dinámica de biases: Ajuste en tiempo real de voltajes de comparación (diff, on, off) y periodos refractarios para el control de la sensibilidad y el ruido.

Grabación en formato RAW (EVT3): Compatibilidad total con el SDK oficial de Metavision para análisis post-procesamiento.

2. Algoritmo de Reconstruccion en Vivo

Integrador Vectorial de Euler: Reconstruccion de imagenes de intensidad a partir de eventos logaritmicos.

Parametrizacion fisica: Ajuste de la constante de tiempo de decaimiento (Tau) para la gestion de estelas y memoria visual.

Procesamiento mediante NumPy: Uso de funciones de acumulacion para procesar tasas de millones de eventos por segundo (MEv/s).

3. Telemetria IMU y GPS

Comunicacion serial de alta velocidad (hasta 2 Mbps).

Analisis de Deriva de Reloj (Clock Drift): Herramienta integrada para comparar la precision temporal del oscilador local del ESP32 frente a los pulsos PPS (Pulse Per Second) de precision atomica del GPS.

Exportacion de datos: Conversor de archivos binarios (.bin) a formato CSV para analisis numerico estadistico.


# Metodología Física

El proyecto se sustenta en principios de procesamiento de señales y teoria de campos. El uso de cámaras de eventos responde a la necesidad de capturar fenómenos de alta dinámica con una resolucion temporal de microsegundos, superando las limitaciones de las cámaras convencionales por cuadros (frames). El análisis de la deriva del PPS es critico para asegurar la integridad de las marcas de tiempo en experimentos como reconstrucción de turbulencia.


# Requisitos del Sistema

Sistema Operativo: Linux / Windows

Lenguaje: Python 3.8


  Librerias principales:

PyQt6 (Interfaz Grafica)

Metavision SDK (Drivers del Hardware)

NumPy (Calculo Numérico)

Matplotlib (Visualización de Datos)

PySerial (Comunicación de Hardware)


# Licencia

Este proyecto es de uso academico y de investigacion dentro del laboratorio de óptica. Para colaboraciones o uso de los algoritmos de reconstrucción, favor contactar al autor: eduardo.llanten.p@mail.pucv.cl o pablo.garrido@pucv.cl

