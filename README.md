## Description
- simple object detection in specific area (polygon) for models from [TF API Model ZOO](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models)

![example]()

## Installation
```bash
python3 -m venv venv
source venv/bin/activate

pip install -U wheel pip setuptools
pip install -r requirements.txt

# TF installation
pip install tensorflow==1.15  # for CPU
# or
pip install tensorflow-gpu==1.15  # for CPU | GPU

# pull models
export GOOGLE_APPLICATION_CREDENTIALS=$PWD/credentials/gs_viewer.json
dvc pull
```

## Usage
```bash
export TF_CPP_MIN_LOG_LEVEL=5  # to avoid TF logs

python detect.py -s data/images/normal.jpg -c configs/tf_object_api_cfg.yml -p "[0,0], [0,1], [1,1], [1,0]" -a 0.0001
```
- **Note**: [model's config explained](https://github.com/jackersson/gst-plugins-tf/blob/master/docs/tf_object_detection_model_config.md)