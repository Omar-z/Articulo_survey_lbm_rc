# Articulo_survey_lbm_rc
## Dependencias
Hay que  primero instalar las siguientes librerias en python, ya sea utilizando un manejador de entornos o directo.

**En el caso de windows** Si se quiere usar el GPU, es necesario **que todo se haga en la maquina virtual nativa de linux en windows** **(WSL)**, ya que no tiene soporte con windows los drivers de gpu con las versiones de las librerias


## Anaconda o Miniconda

1. Instalar Anaconda o Miniconda (tiene menos herramientas, más ligero) 
  - **Anaconda o Miniconda:** https://www.anaconda.com/download 

2. Una vez instalado anaconda o miniconda, abrir:

  - [Windows] **conda command prompt**
  - [Linux/MacOs] **La terminal**

3. Instalar un entorno con python 3.10+
```bash
conda create -n articulo python=3.14
```

4. Si todo sale bien saldrá un mensaje de que te puedes cambiar a tu entorno con el siguiente comando.
 ```
 conda activate articulo
 ```

Al realizar el comando anterior te debe salir un **(articulo)** en tu terminal
<img width="616" height="58" alt="image" src="https://github.com/user-attachments/assets/feb8af98-8495-4303-b22b-eaf2f4571b21" />

5. Instalar las siguientes librerias
* numpy
* pandas
* matplotlib
* seaborn
* scikit-learn
* pytorch
* ipykernel
* scipy

```
conda install numpy pandas matplotlib seaborn scipy ipykernel
```
```
conda install -c conda-forge pytorch torchvision scikit-learn
```

6. Instalar paquetes que no estan en el gestor de entorno
```
pip install fuzzylab
```

7. De aqui ya puedes abrir los archivos **guia_como_usar_...** y **Seleccionar el entorno que acabamos de configurar**
<img width="1094" height="311" alt="image" src="https://github.com/user-attachments/assets/edaaf0de-7846-4dd5-9e73-bab52d8aab9b" />

## Si se va usar GPU, hay que cambiar una variable que por ahí anda llamada device, esta con "mps" que es para MAC
- "cuda" $\rightarrow$ gpu nvidia de 2060+
- "mps" $\rightarrow$ macos de m1+ **no trabaja con floats de 64bits, solo de 32**
- "cpu" $\rightarrow$ cpu

## Commits al repositorio

Si van hacer commits al repositorio, hacerlo en un branch nuevo para que no haya tanto conflicto al hacer el merge con el main.

