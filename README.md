# ai4eo_project

## Dataset download
Roboflow Datasets can be either imported to a jupyter notebook or downloaded as a .zip file.
I used a mixture of both. 
### SARD
Can be directly accessed from [here](https://universe.roboflow.com/animesh-shastry/sard_yolo) 
### DB Licenta
Can be directly accessed from [here](https://universe.roboflow.com/licenta-ynwvo/db_licenta) 
### VisDrone
This one was a bit tricky as they used an old-fashioned tool to do the annotations and it wasn't directly usable with any of the models here, however, and to my luck, I found out a hidden [file](https://github.com/ultralytics/yolov5/blob/master/data/VisDrone.yaml) within the yolov5 repository which was used specifically to convert this into yolo .txt format. 
### SaRNet
Data can be downloaded from here: [sarnet.zip](https://michaeltpublic.s3.amazonaws.com/sarnet.zip)
The dataset was annotated in COCO format, and this format needed to be changed.
### TinyPerson
Can be directly accessed from [here](https://universe.roboflow.com/chris-d-dbyby/tinyperson) 

### Format exchange
In order to change from one format to another I used two ways, one using FiftyOne library (in [FiftyOne](/utils/fiftyOne.py)) and the other was using the RoboFlow platform.

### Label manangment
In one of the datasets(VisDrone), we needed to ignore some of the classes and merge the human classes into one person class, for that I used a simple [script](/utils/labelUtil.py). P.S.: this works only for YOLO format.

## Models
### Yolov5
```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```
### FasterRCNN
can be directly imported from the torchvision `torchvision.models.detection.fasterrcnn_resnet50_fpn_v2`
### RetinaNet
can also be imported from torch vision `torchvision.models.detection.retinanet_resnet50_fpn_v2`


## Model Training
### Yolov5
In order to train the model I created a train script like the one we used for the DL lectures and homeworks.
However, I wanted some evaluation and metrics and some graphs and I really didn't want to reinvent the wheel, since all of this was done before. In addition, I need something that is robust to multiple many datasets, some code that I can really rely on for the training.
So after a while of training the Yolov5m on my script I decided to switch to the train script that came with the model. 
This [script](https://github.com/ultralytics/yolov5) is designed to take as input all what we want to do without caring too much about what is happening in the code, of course freezing layers and playing with different parameters inside the model won't work with it out-of-the-box, yet we can do much with it. 
All my augmentations I provided as entries in the hyp.yaml input file. Those hyper parameters are then used as transforms from the Albumentations library.
This is how I used it 
```bash
## For training
python yolov5/train.py --img 1080 --batch 4 --epochs 100 --data SARD_YOLO_1080/data.yaml --weights YOLOv5_SARD_NORMAL/weights/last.pt --hyp hyp.yaml --device 0
## For evaluation
python yolov5/val.py --data SARD_YOLO_1080/data.yaml --weights yolov5/runs/train/exp4/weights/best.pt --img 1080
## For inference/prediction
python yolov5/detect.py --weights YOLOv5_SARD_NORMAL_HFLIP/weights/best.pt --img 1080 --source SARD_YOLO_1080/test/images --device 0 --save-txt --save-conf --save-crop
```
### FasterRCNN

For FasterRCNN I used the training pipeline done by [sovit-123](https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline/tree/main)
Good thing about thing pipeline, is that it is modular. 
Data Augmentation can be added and applied in [here](/fastercnn-pytorch-training-pipeline/utils/transforms.py) in the get_train_aug function.
This is how I used it 
```bash
## For training
python fastercnn-pytorch-training-pipeline/train.py --data SARD_RCNN_640/data.yaml --use-train-aug --epochs 100 --model fasterrcnn_resnet50_fpn_v2 --name FasterRCNN_SARD_fpn_v2_with_horizontal_flip --batch 2 --disable-wandb
## For evaluation
python fastercnn-pytorch-training-pipeline/eval.py --weights outputs/training/FasterRCNN_SARD_fpn_v2/best_model.pth --data SARD_RCNN_640/data.yaml --model fasterrcnn_resnet50_fpn_v2 --verbose
## For inference/prediction
python fastercnn-pytorch-training-pipeline/inference.py --weights outputs/training/FasterRCNN_SARD_fpn_v2/best_model.pth --input SARD_RCNN_640/test/images/
```
### RetinaNet
For RetinaNet, I liked the fastercnn-pytorch-training-pipeline and for that reason I did some hacking to the script and made it compatible with RetinaNet. I used it the same way I used the FasterRCNN script.

## Extras
I have all weights of all models on my PC, and they are avaiable anytime. 
Also the inferences on all datasets.