<div>
    <h1> Video License Plate Recognition 
</div>

<div align = "center">
<table>
  <tr>
    <td><img src="https://github.com/wavelolz/Video-License-Plate-Recognition/blob/main/Picture/demo%20video%201.gif" alt="Image 1" width="500"></td>
  </tr>
    
  <tr>
    <td><img src="https://github.com/wavelolz/Video-License-Plate-Recognition/blob/main/Picture/demo%20video%202.gif" alt="Image 2" width="500"></td>
  </tr>
</table>
</div>


> The project aims to detect and recognize the license plate in the video using Pytorch and OpenCV, along with some other packages developed by distinguished developers
> The markdown file contains thorough description of the project

## The Whole Procedure of Detection
<p align = "center">
    <img src = "https://github.com/wavelolz/Video-License-Plate-Recognition/blob/main/Picture/flowchart.jpg" width = 600 height = 250>
</p>

## Locate the Cars
To find the location of the cars from video, I made use of existing packages called [cvlib](https://github.com/arunponnusamy/cvlib), which was developed using [YOLO](https://github.com/AlexeyAB/darknet)

## Find Potential License Plate from Cars
After cropping the region of cars found by cvlib, I have to find the location of license plate. To do so, I perform some image processing techniques such as Blurring, Thresholding, Morphology etc. From here, we would have several images that might be license plate. In fact, these images could be sent to number detection directly. However, it's time-consuming as most of the images do not contain license plate. So I take an additionl step to predict whether the cropped images are license plates or not.

## Decide whether the images are license plates or not
Here, to predict whether the images are license plate or not, I made use of convolutional neural network (CNN). The structure of the CNN is as following

<p align = "center">
    <img src = "https://github.com/wavelolz/Video-License-Plate-Recognition/blob/main/Picture/cnn%20model.jpg" width = 600 height = 250>
</p>


## Recognize numbers on captured license plates
Eventually, I made use of [EasyOCR](https://github.com/JaidedAI/EasyOCR) to recognize the numbers and alphapets on the license plates

## Structure of repo

```
+---Main
|       main.py
|       model.py
|
+---Model
|       datasets.py
|       patchify.py
|       train.py
|
\---Video
        003.mp4
        006.mp4
```
- Main
    - main.py: contain the main function and any other functions of the project
    - model.py: define the structure of CNN model
- Model
    - datasets.py: used to create data loader from images
    - patchify.py: used to create non-license plate training datasets
    - train.py: used to train the structure of CNN model
 ## Problem that needs to be solved
 1. Localization of license plate after detecting a car isn't accurate enough
 2. Recognition of numbers on the plate isn't accurate enough

**Though there are lots of things which need improvements in this project, it's my first approach to pytorch and computer vision. If you find this project helpful for you to catch the first glimpse into the topics, plz click on star on the top right 😄**

<div align = "center">
<img src = "https://github.com/wavelolz/Video-License-Plate-Recognition/blob/main/Picture/capo.gif" width = 300>
</div>







