## Lyft Perception Challenge

This project identifies vehicles and drivable road in dashcam images for the
Lyft Perception Challenge.
Dashcam images are extracted from the [CARLA simulator](http://carla.org/).

![dashcam picture](https://raw.githubusercontent.com/ericlavigne/Lyft-Perception-Challenge/master/images/923.png)

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

Running the project
---

```sh
cd Lyft-Perception-Challenge
source env/bin/activate
python main.py
deactivate
```

Installing new library
---

```sh
cd Lyft-Perception-Challenge
source env/bin/activate
pip freeze > requirements.txt
deactivate
```
