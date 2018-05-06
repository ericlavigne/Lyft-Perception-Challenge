## Lyft Perception Challenge

This project identifies vehicles and drivable road in dashcam images for the
Lyft Perception Challenge.
Dashcam images are extracted from the [CARLA simulator](http://carla.org/).

![dashcam picture](https://github.com/ericlavigne/Lyft-Perception-Challenge/raw/master/images/923.png)

Installation
---

1. Clone the repository

```sh
git clone https://github.com/ericlavigne/Lyft-Perception-Challenge
```

2. Setup virtualenv.

```sh
cd Lyft-Perception-Challenge
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
deactivate
```

Official submission process
---

```sh
./preinstall_script.sh
cd Lyft-Perception-Challenge
grader 'python submit.py'
submit
```

Training
---

The neural network training process assumes that training data can be found
in /tmp/Train.

```sh
python train.py
```

Testing
---

For manual unit testing, the test.py script creates visual examples of each step
in the /tmp/output directory. The test.py script assumes that training data can
be found in /tmp/Train.

```sh
python test.py
```
