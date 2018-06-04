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

The overall shape of the segmentation model is loosely based on SegNet. At a smaller
scale, individual convolutions in SegNet are replaced by Inception modules scaled to
similar output depth as the original SegNet convolutions. The Inception modules increase
the learning potential of the network compared to the original SegNet while also
allowing greater flexibility for increasing the layer depths without an explosion in the
number of connections.

I created two separate segmentation models tailored to the needs of the two classes:
vehicle and road. Cars were much more difficult to classify accurately, so the vehicle model
has roughly four times the output depth per Inception module compared with the road model.
Keeping the models separate makes it possible to directly allocate extra learning potential
and inference time to the more difficult classification problem.

#### Inference Speed Optimization

Self-driving cars need to interpret images at 10 frames per second to ensure that they
can react quickly to unexpected changes. This challenge allows lower inference speeds,
but deducts a steep penalty of one point per FPS below 10. The need to keep inference fast
limits the complexity of models and leads to much more difficulty in creating accurate
segmentation models.

Rather than accepting a hard limit on the complexity of segmentation models, I chose to
optimize the surrounding inference process, leaving as much room as possible for powerful
segmentation models.

1. Inference server process avoids loading libraries and models during a realtime process.

Loading libraries and models can both take substantial time when a Python program first
starts up. I solved this problem by starting a separate server process that would load
all relevant libraries, load models, and even warm up on a small practice problem. After
the server is finished with the warm up process, a client script can go through the grading
process and delegate work to the server process. The client and server communicate via
ZeroMQ.

This approach is relevant for use on a real car because the FPS requirement is intended
to measure latency in an ongoing realtime activity, for which load time is not relevant.
If a perception module needed to provide fast inference immediately when the car is turned
on, it could achieve this by by first loading a lower accuracy model that is sufficient for
driveway use and have the stronger model loaded before the car moved into a more difficult
situation.

2. Multiprocessing better utilizes both CPU and GPU at the same time.

While inference is primarily performed by the GPU, several of the surrounding processes
take substantial CPU time: loading frames from the input video, cropping and scaling to
a smaller image size for inference, scaling back up after inference, and encoding the
output in PNG format. If these processes are performed sequentially, then at any time only
the GPU or one of the CPUs will be active. Instead, I created several parallel processes
communicating via Python's multiprocessing Pipes. In a real car, this would mean that the
perception pipeline could minimize latency by accepting a second image for pre-processing
while previous images are still going through inference or post-processing.

Note: Python has weak support for multiprocessing. While this technique was effective, the
same technique in C++ would likely provide a much better speedup.

3. Batching multiple images better utilizes the GPU.

At first I avoided batching because it seemed unrealistic. If I wait for 10 images to
arrive and perform inference all at once, this could meet the 10 FPS throughput
requirement while providing a latency equivalent to only 1 FPS.

Surprisingly, it turns out that batching is relevant for a real self-driving car.
Self-driving cars have multiple cameras whose output must be processed in parallel. Tesla,
for example, uses 8 optical cameras on each car. Batch inference is an effective way to
process the output from 8 cameras at once. Also, the hosted workspace used for this
competition is about 5 times slower than my two-year old TitanX, so I suspect that the same
code running on modern hardware could handle 10 FPS inference on all 8 cameras at once.

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

Ong Chin-Kiat (chinkiat), Phu Nguyen (phmagic), and Mohamed Eltohamy all collected extra
training data from CARLA and posted that data for other students to use. I have no
measurement for how this affected my project, but suspect that it was very helpful for
improving the accuracy.

Jay Wijaya (jaycode) shared the hint that OpenCV's VideoCapture was faster than the
scikit-video operation that was used in Udacity's example script. This change improved
my speed by 0.5 FPS.

Phu Nguyen (phmagic) shared the hint that OpenCV was faster than PIL for encoding to PNG
format. This change cut the PNG encoding time in half but did not affect my overall speed
because that part of the process was already moved into its own thread and not a bottleneck.
