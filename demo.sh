CONFIG=configs/tf_object_api_cfg.yml
OUT=demo/out.jpg

python detect.py -s data/images/normal.jpg -c $CONFIG -o $OUT -p "[0.06, 0.97], [0.68, 0.53], [0.99, 0.59], [0.69, 0.99]"
python detect.py -s data/images/night.jpg -c $CONFIG -o $OUT -p "[0.48, 0.53], [0.98, 0.6], [0.99, 0.98], [0.1, 0.98]"
python detect.py -s data/images/night2.jpg -c $CONFIG -o $OUT -p "[0.63, 0.44], [0.59, 0.7], [0.79, 0.75], [0.82, 0.48]"
python detect.py -s data/images/normal2.jpg -c $CONFIG -o $OUT -p "[0.0, 0.74], [0.25, 0.61], [0.75, 0.63], [0.64, 0.99], [0.0, 0.99]"
python detect.py -s data/images/rain.jpg -c $CONFIG -o $OUT -p "[0.0, 0.98], [0.0, 0.83], [0.13, 0.83], [0.99, 0.94], [0.99, 0.98]"
python detect.py -s data/images/shadow.jpg -c $CONFIG -o $OUT -p "[0.0, 0.6], [0.4, 0.45], [0.67, 0.47], [0.51, 0.79], [0.0, 0.73]" -a 0.05

