## Lyft Perception Challenge

This project identifies vehicles and drivable road in dashcam images for the
Lyft Perception Challenge.
Dashcam images are extracted from the [CARLA simulator](http://carla.org/).

| Dashcam Image | Leaderboard |
|:-------------:|:-----------:|
| ![dashcam picture](https://github.com/ericlavigne/Lyft-Perception-Challenge/raw/master/images/923.png) | ![leaderboard](https://github.com/ericlavigne/Lyft-Perception-Challenge/raw/master/images/leaderboard.png) |

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
./setup.sh
pip install -r requirements.txt
deactivate
```

Official submission process
---

This project uses a client/server architecture for inference in order to avoid paying
a startup cost (loading libraries and models) during a real-time process. The server
and client should be started in separate consoles. The server should be allowed to
complete the warmup process before running the client.

1. Running the server

```sh
cd Lyft-Perception-Challenge
./setup.sh
python submit_server.py
submit
```

Wait for the warmup process to complete. The server will report that warmup has
completed and show speed statistics for a small video on which it performs inference
during the warmup process.

2. Running the client

```sh
grader 'python submit_client.py'
submit
```

Training
---

The neural network training process assumes that training data can be found
in /tmp/Train as a result of running setup.sh during the installation process.

```sh
python train_car.py
python train_road.py
```

Testing
---

For manual unit testing, the test.py script creates visual examples of each step
in the /tmp/output directory. The test.py script assumes that training data can
be found in /tmp/Train as a result of running setup.sh during the installation
process.

```sh
python test.py
```
