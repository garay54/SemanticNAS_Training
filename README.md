# Guia de uso - Semantic NAS From Scratch Package

Este paquete entrena modelos de segmentacion semantica **desde cero** y ejecuta
NAS con Optuna. No usa `pretrained/best_model.pth`, no hace warm start y no esta
amarrado a grietas. Sirve para nuevos objetos/clases con mascaras semanticas.

---

## Que hay en esta carpeta

```text
semantic_nas_train_package/
├── train.py                                  # Entrenamiento NAS desde cero
├── requirements.txt                         # Dependencias
├── configs/
│   ├── search_space.yaml                    # Arquitecturas, encoders, losses y rangos
│   └── classes.example.json                 # Ejemplo de nombres/colores de clases
└── scripts/
    ├── tile_semantic_dataset.py             # Tilea imagenes y mascaras semanticas
    └── yolo_polygons_to_semantic_masks.py   # YOLO-seg polygon -> mascaras PNG
```

---

## Requisitos del sistema

- Python 3.8 o superior
- GPU NVIDIA con CUDA recomendada
- 8 GB de VRAM como punto de partida
- 20 GB o mas de espacio libre, segun tamano del dataset y numero de trials

---

## Paso 1 - Instalar dependencias

Entra a la carpeta del paquete:

```bash
cd semantic_nas_train_package
```

Instala PyTorch segun tu CUDA. Ejemplo para CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verifica GPU:

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Paso 2 - Preparar el dataset

El entrenamiento espera esta estructura:

```text
dataset/tiles/
├── train/
│   ├── images/    # .jpg, .png, .bmp, .tif
│   └── masks/     # .png con IDs de clase
├── valid/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Las mascaras deben ser PNG semanticas:

```text
0 = fondo
1 = clase/objeto 1
2 = clase/objeto 2
...
N-1 = clase/objeto N-1
```

`--num-classes` siempre incluye el fondo. Ejemplos:

```text
fondo + una clase    -> --num-classes 2
fondo + tres clases  -> --num-classes 4
```

Si tienes mascaras binarias antiguas `0/255`, puedes usar `--num-classes 2`;
el loader convierte todo valor mayor que 0 a clase `1`.

---

## Opcion A - Ya tienes imagenes con mascaras PNG

Si tus imagenes son grandes, aplica tileado preservando los IDs de clase:

```bash
python3 scripts/tile_semantic_dataset.py \
    --input tu_dataset \
    --output dataset/tiles \
    --tile-size 512 \
    --overlap 64 \
    --min-foreground 0.001
```

Si quieres filtrar solo algunas clases al decidir si guardar un tile:

```bash
python3 scripts/tile_semantic_dataset.py \
    --input tu_dataset \
    --output dataset/tiles \
    --foreground-ids 1,2 \
    --min-foreground 0.001
```

---

## Opcion B - Tienes etiquetas YOLO segmentation polygon

Convierte `labels/*.txt` a mascaras semanticas:

```bash
python3 scripts/yolo_polygons_to_semantic_masks.py \
    --input tu_dataset_yolo \
    --output dataset/semantic \
    --data-yaml tu_dataset_yolo/data.yaml
```

El script crea:

```text
dataset/semantic/{train,valid,test}/{images,masks}
dataset/semantic/classes.json
```

Luego tilea:

```bash
python3 scripts/tile_semantic_dataset.py \
    --input dataset/semantic \
    --output dataset/tiles \
    --overlap 64 \
    --min-foreground 0.001
```

Por defecto, YOLO `class_id=0` se guarda como clase semantica `1`, porque `0`
queda reservado para fondo. Si necesitas otra convencion, usa `--class-offset`.

---

## Opcion C - Mascaras RGB por color

El entrenamiento puede leer mascaras RGB solo si das un mapa de colores exactos.
Copia y edita:

```bash
cp configs/classes.example.json configs/classes.json
```

Ejemplo:

```json
{
  "classes": [
    {"id": 0, "name": "background", "color": [0, 0, 0]},
    {"id": 1, "name": "objeto_a", "color": [0, 255, 0]},
    {"id": 2, "name": "objeto_b", "color": [255, 128, 0]}
  ]
}
```

Recomendacion: si puedes, usa mascaras de un canal con IDs. Es mas robusto que
mascaras RGB porque evita errores por antialiasing o compresion.

---

## Paso 3 - Ejecutar NAS desde cero

Ejemplo binario: fondo + un objeto.

```bash
python3 train.py \
    --data dataset/tiles \
    --output-dir runs/semantic_nas \
    --num-classes 2 \
    --n-trials 20 \
    --epochs 40 \
    2>&1 | tee runs/semantic_nas.log
```

Ejemplo multiclase: fondo + 3 objetos.

```bash
python3 train.py \
    --data dataset/tiles \
    --output-dir runs/semantic_nas_multiclass \
    --num-classes 4 \
    --class-config configs/classes.json \
    --n-trials 30 \
    --epochs 50 \
    2>&1 | tee runs/semantic_nas_multiclass.log
```

Por defecto el entrenamiento es desde cero:

```text
encoder_weights = None
warm start       = no existe
checkpoint base  = no se carga
```

Si algun dia quieres probar ImageNet, puedes hacerlo explicitamente:

```bash
python3 train.py --data dataset/tiles --num-classes 4 --encoder-weights imagenet
```

Para mantenerlo estrictamente desde cero, no uses `--encoder-weights`.

---

## Que busca el NAS

El archivo `configs/search_space.yaml` define el espacio de busqueda:

```yaml
architectures:
  - Unet
  - FPN
  - DeepLabV3Plus
  - Linknet
  - UnetPlusPlus

encoders:
  - resnet18
  - resnet34
  - mobilenet_v2
  - efficientnet-b0
  - mit_b0
  - mit_b1

losses:
  - dice_ce
  - focal_dice
  - tversky_ce
```

Tambien busca:

| Parametro | Descripcion |
|---|---|
| `batch_size` | Tamano de batch por trial |
| `decoder_lr` | Learning rate principal |
| `encoder_lr_ratio` | LR del encoder relativo al decoder |
| `weight_decay` | Regularizacion AdamW |
| `dice_weight` | Peso Dice en losses combinadas |
| `augmentation` | Nivel de aumentacion: light, medium, strong |

La metrica objetivo por defecto es `mean_iou` sin fondo. Puedes cambiarla:

```bash
python3 train.py --data dataset/tiles --num-classes 4 --objective-metric mean_dice
```

---

## Artefactos generados

```text
runs/semantic_nas/
├── nas_study.db             # Historial completo de Optuna
├── best_nas_result.json     # Mejor trial global
└── trial_0/
    ├── best_model.pth       # Pesos del mejor epoch del trial
    ├── model_config.json    # Arquitectura, encoder, clases y loss
    ├── trial_info.json      # Score, mIoU, mDice, params y coste
    ├── training_log.csv     # Curva por epoca
    └── vis_samples/         # original | GT | prediccion
```

El archivo `model_config.json` debe viajar siempre con `best_model.pth` para
inferencia futura.

---

## Continuar un entrenamiento interrumpido

Optuna guarda el estudio en SQLite. Vuelve a correr el mismo comando y continua:

```bash
python3 train.py \
    --data dataset/tiles \
    --output-dir runs/semantic_nas \
    --num-classes 2 \
    --n-trials 20 \
    --epochs 40 \
    2>&1 | tee -a runs/semantic_nas.log
```

---

## Prueba rapida

Usa esto para validar instalacion y estructura sin correr un NAS largo:

```bash
python3 train.py \
    --data dataset/tiles \
    --output-dir runs/smoke_semantic_nas \
    --num-classes 2 \
    --n-trials 1 \
    --epochs 1 \
    --architectures Unet \
    --encoders resnet18 \
    --losses dice_ce \
    --max-train-batches 2 \
    --max-val-batches 2
```

---

## Ajustar el search space

Para una GPU con poca VRAM, reduce `batch_sizes` y evita encoders grandes:

```yaml
batch_sizes:
  - 2
  - 4

encoders:
  - resnet18
  - mobilenet_v2
```

Para buscar mas fuerte con mejor GPU:

```yaml
encoders:
  - resnet34
  - efficientnet-b0
  - mit_b0
  - mit_b1
  - mit_b2

batch_sizes:
  - 4
  - 8
  - 12
```

Nota: `UnetPlusPlus` se limita automaticamente a encoders compatibles cuando el
script detecta encoders MiT.

---

## Solucion de problemas frecuentes

**CUDA out of memory**

Reduce `batch_sizes` en `configs/search_space.yaml`, o usa:

```bash
--architectures Unet,FPN --encoders resnet18,mobilenet_v2
```

**Valor de clase invalido**

Revisa que tus mascaras multiclase tengan IDs `0..num_classes-1`. Si tienen
valor `255` como void, usa:

```bash
--ignore-index 255
```

No uses `--ignore-index 255` si tus mascaras binarias usan `255` como objeto.
En ese caso deja el default y usa `--num-classes 2`.

**Mascaras RGB no definidas**

Pasa:

```bash
--class-config configs/classes.json
```

con colores exactos.

**Error `_ARRAY_API not found` o `numpy.core.multiarray failed to import`**

Tu entorno tiene NumPy 2.x con un OpenCV compilado para NumPy 1.x. Reinstala las
dependencias del paquete:

```bash
pip install "numpy<2" --force-reinstall
pip install -r requirements.txt --force-reinstall
```

**Quiero entrenamiento estrictamente desde cero**

No pases `--encoder-weights`. El default es `None`.

---

## Resultado esperado

Al terminar, revisa:

```bash
cat runs/semantic_nas/best_nas_result.json
```

Y para un trial:

```bash
cat runs/semantic_nas/trial_0/trial_info.json
```

Las metricas principales son:

| Metrica | Significado |
|---|---|
| `mean_iou` | IoU medio por clase, excluyendo fondo por defecto |
| `mean_dice` | Dice medio por clase, excluyendo fondo por defecto |
| `score` | Objetivo Optuna, con penalizacion opcional por latencia/parametros |
