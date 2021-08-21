# UNAL_ComputerVision2021

## About this project.
The notebook called [Main](./ComputerVisionProject/Main.ipynb) is the file that shows the initial sequence of computer vision operation to get the desired behavior, to detect lines on a hallway selected (the current apartment where I am living). Then, the image processing pipeline is created on [lane_detection_pipeline.py](./ComputerVisionProject/lane_detection_pipeline.py) as a class that receives the image and persist values to obtain better performance with each operation with the class called `LaneDetection`. The pipeline is "unit tested" with the notebook [TestComputerVisionPipeline](./ComputerVisionProject/TestComputerVisionPipeline.ipynb) where you can also see some interesting results over the image data set available on [dataSet3](./ComputerVisionProject/dataSet3) folder.

## How to run and play with the project? ##

This project is wrapped on a Docker container to run a jupyter-lab environment. Make sure that you have docker installed in your machine. Open the OS Terminal and then execute the [`build`](./docker_scripts/build) script once to create the container and [`run`](./docker_scripts/run) always:

```
$ # build to create the container (just need to be executed only once)
$ ./docker_scripts/build
$ # run to execute the project container
$ ./docker_scripts/run
```

## Car used to capture images ##
![](./ComputerVisionProject/car.jpeg)

## Environment selected ##
![](./ComputerVisionProject/environmentTesting.jpeg)
