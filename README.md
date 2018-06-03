## Lyft Perception Challenge

In May 2018, Udacity and Lyft organized a competition for accurate and fast
classification of dashcam images as part of a self-driving car's perception process.
This project identifies vehicles and drivable road to meet the requirements of that
competition. All dashcam images were extracted from the
[CARLA simulator](http://carla.org/).

| Dashcam Image | Leaderboard |
|:-------------:|:-----------:|
| ![dashcam picture](https://github.com/ericlavigne/Lyft-Perception-Challenge/raw/master/images/923.png) | ![leaderboard](https://github.com/ericlavigne/Lyft-Perception-Challenge/raw/master/images/leaderboard.png) |


### Highlights

* Segmentation model combines ideas from SegNet and Inception
* Inference server process avoids loading libraries and models during a realtime process
* Multiprocessing better utilizes both CPU and GPU at the same time
* Weighted mean-squared error loss function optimizes segmentation for rare class

*Note: Find the latest version of this project on
[Github](https://github.com/ericlavigne/Lyft-Perception-Challenge).*

---

### Contents

* [Project Components](#project-components)
  * [Segmentation Model](#segmentation-model)
  * [Inference Speed Optimization](#inference-speed-optimization)
* [Usage](#usage)
  * [Installation](#installation)
  * [Running Inference for Official Grader](#running-inference-for-official-grader)
  * [Training](#training)
  * [Testing](#testing)
* [Acknowledgements](#acknowledgements)

---

### Project Components

#### Segmentation Model

#### Inference Speed Optimization

### Usage

#### Installation

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

#### Running Inference for Official Grader

This project uses a client/server architecture for inference in order to avoid paying
a startup cost (loading libraries and models) during a real-time process. The server
and client should be started in separate consoles. The server should be allowed to
complete the warmup process before running the client.

1. Running the server

```sh
cd Lyft-Perception-Challenge
./setup.sh
python submit_server.py
```

Wait for the warmup process to complete. The server will report that warmup has
completed and show speed statistics for a small video on which it performs inference
during the warmup process.

2. Running the client

```sh
cd Lyft-Perception-Challenge
grader 'python submit_client.py'
submit
```

#### Training

The neural network training process assumes that training data can be found
in /tmp/Train as a result of running setup.sh during the installation process.

```sh
python train_car.py
python train_road.py
```

#### Testing

For manual unit testing, the test.py script creates visual examples of each step
in the /tmp/output directory. The test.py script assumes that training data can
be found in /tmp/Train as a result of running setup.sh during the installation
process.

```sh
python test.py
```

### Acknowledgements


