mkdir -p output/augment
mkdir -p output/infer_car
mkdir -p output/infer_road
mkdir -p output/preprocessing
scp eric@192.168.1.97:/tmp/output/augment/*.png output/augment/
scp eric@192.168.1.97:/tmp/output/infer_car/*.png output/infer_car/
scp eric@192.168.1.97:/tmp/output/infer_road/*.png output/infer_road/
scp eric@192.168.1.97:/tmp/output/preprocessing/*.png output/preprocessing/