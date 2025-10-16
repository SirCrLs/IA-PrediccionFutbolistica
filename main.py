# main.py
from Fase1_preparacion import ejecutar_fase1
from Fase2_entrenamiento import ejecutar_fase2

if __name__ == "__main__":
    print("= PROYECTO PREDICCIÓN DE FUTBOL =\n")

    try:
        ejecutar_fase1()
        ejecutar_fase2()
    except Exception as e:
        print("Error")
