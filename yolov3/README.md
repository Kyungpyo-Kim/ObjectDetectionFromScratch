# YoloV3


## Getting started
```bash
cd yolov3
python -m venv ./env
.\env\Scripts\Activate.ps1
python -m pip install pip -U
python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

cd cfg
wget https://pjreddie.com/media/files/yolov3.weights
# Invoke-WebRequest -Uri https://pjreddie.com/media/files/yolov3.weights  -OutFile yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
# Invoke-WebRequest -Uri https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg  -OutFile yolov3.cfg

mkdir data
cd data
wget https://raw.githubusercontent.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/master/data/coco.names
# Invoke-WebRequest -Uri https://raw.githubusercontent.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/master/data/coco.names -OutFile coco.names

python yolov3.py
python detector.py --images dog-cycle-car.png --det det
```


## TODO
- [ ] add pytest


## Reference
* [how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch](https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/)
