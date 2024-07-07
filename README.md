# Proyecto-Rec-Sys

## Environment

El código usa python 3.9.16, cuando descargues el repo, puedes crear un environment con el comando

```python
python -m venv nombre_de_env
pip install --upgrade pip
```

y correr este comando para descargar todos los packages utilizados

```python
pip install numpy pandas torch tensorboard tensorboardX pykan tqdm scikit-learn matplotlib
```

Además se debe crear un directorio `checkpoints` dentro de `neural_cf` que almacena los checkpoints de
los modelos en cada epoch con los siguientes subdirectorios:

```python
neural_cf
|
└───checkpoint
    |
    └───gmf
    |
    └───kan
    |
    └───kan_test
    |
    └───kanmf
    |
    └───mlp
    |
    └───neumf
```

- Los archivos del directorio pueden ser ignorados si no se planea usar preentrenamiento, pero los directorios deben estar en
instanciados, ya que el codigo se cae en caso contrario.

## Run

Para entrenar un modelo en especifico, se debe editar un config en `train.py` para que tenga
los parámetros deseados, descomentar el Engine a utilizar y correr

```python
python train.py
```

Si se quiere correr los experimentos de los hiperparámetros de KAN, se corre

```python
python kan_exp.py
```
