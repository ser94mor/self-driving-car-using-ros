# Udacity Self-Driving Car Engineer Nanodegree - Capstone project
--------

## Team 5TZones
|   Team Member   |  Name   |            Email            |
|-----------------|---------|-----------------------------|
| @Elgeweily      | Mohamed | mohamed.elgeweily@gmail.com |
| @Jerry          | Jerry   | jerrytansk@gmail.com        |
| @karthik        | Karthik | karthikeya108@gmail.com     |
| @pradeepkorivi  | Pradeep | pradeepkorivi@gmail.com     |
| @sergey.morozov | Sergey  | ser94mor@gmail.com          |

## Software Architecture
![Software architecture](https://d17h27t6h515a5.cloudfront.net/topher/2017/September/59b6d115_final-project-ros-graph-v2/final-project-ros-graph-v2.png)
Note that obstacle detection is not implemented for this project.

## Traffic Light Detection Node

A large part of the project is to implement a traffic light detector/classifier that recognizes the color of nearest upcoming traffic light and publishes it to /waypoint_updater node so it can prepare the car to speed up or slow down accordingly. Because the real world images differ substantially from simulator images, we tried out different approaches for both.

### Simulator - opencv approach
@Pradeep - pls explain your approach here

Here is a sample of the simulator dataset
(Insert image here!)

### Test Lot - SSD (Single Shot Detection)
We need to solve both object detection - where in the image is the object, and object classification - given a localized image, classify the object in the image. While there are teams who approached it as 2 separate problems to be solved, recent advancements in Deep Learning has developed models that attempt to solve both at once - SSD (Single Shot Detection) and YOLO (You Only Look Once).

We attempted transfer learning using the pre-trained SSD_inception_v2 model trained on COCO dataset, and retrain it on our own dataset for NUM_EPOCHS, achieving a final loss of FINAL_LOSS.

@Segey, pls elaborate how you collated the dataset.

Here is a sample of the dataset.
(Insert image here!)

Here are the results of our trained model.
(Insert image here!)

### Learning Points


### Future Work


### Acknowledgements
