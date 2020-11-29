# Analiza kolejki

Program analizujący wideo w celu wykrywania ludzi i przydzielania ich do kategorii według koloru ubrań.

## Jak uruchomić program?
Żeby uruchomić program potrzebna jest instalacja właściwych pakietów. Zalecamy zrobienie tego poprzez Anacondę i Pip.

### Conda

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu
```

### Pip
(TensorFlow 2 packages wymagają wersję pip >19.0.)
```bash
# TensorFlow CPU
pip install -r requirements.txt
```

## Pobranie YOLOv4 Pre-trained Weights
Pobierz plik: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

Przenieś pobrany plik do folderu 'data' w tym repozytorium.

## Uruchomienie programu
Wykonaj poniższe komendy.
```
# save yolov4-tiny model
python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --model yolov4 --tiny

# Run yolov4-tiny object tracker
python object_tracker.py --video ./data/video/grupaB1.mpg --output ./outputs/tiny.avi
```

## Możliwe argumenty programu

```bash
save_model.py:
  --weights: path to weights file
    (default: './data/yolov4.weights')
  --output: path to output
    (default: './checkpoints/yolov4-416')
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'False')
  --input_size: define input size of export model
    (default: 416)
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)
    
 object_tracker.py:
  --video: path to input video (use 0 for webcam)
    (default: './data/video/test.mp4')
  --output: path to output video (remember to set right codec for given format. e.g. XVID for .avi)
    (default: None)
  --output_format: codec used in VideoWriter when saving video to file
    (default: 'XVID)
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'true')
  --weights: path to weights file
    (default: './checkpoints/yolov4-tiny-416')
  --size: resize images to
    (default: 416)
  --iou: iou threshold
    (default: 0.45)
  --score: confidence threshold
    (default: 0.50)
  --logs: path to the output logs
    (default: './outputs/logs.txt')
```
