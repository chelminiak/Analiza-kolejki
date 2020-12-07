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

## Zapisanie potrzebnego modelu 
Wykonaj poniższą komendę.
```
# save yolov4-tiny model
python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --model yolov4 --tiny
```

## Uruchomienie GUI
Wykonaj poniższą komendę albo uruchom program poprzez przycisk w środowisku programistycznym.
```
# run GUI
python GUI.py
```

Do poprawnego działania pliku należy podać ścieżkę do pliku wejściowego oraz dwie ścieżki do folderów, w których zapisane zostaną wynikowe pliki z logami i wideo. Jeżeli użytkownik nie poda nazwy wynikowych plików zostaną one wylosowane, zgodnie z następującą regułą: [log/video][randint(0,100000)].[txt/avi]. W celu zwiększenia intuicyjności programu program automatycznie dopisze rozszerzenie txt/avi po podanej nawie pliku wynikowego.
