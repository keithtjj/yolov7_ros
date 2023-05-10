# yolov7-ros
Forked from official YOLOv7 repo.
See [here](https://github.com/WongKinYiu/yolov7) for more information on detection variables.
## Getting Started
1) Install NVIDIA GPU drivers and CUDA Toolkit
2) Install `requirements.txt`
```
pip3 install -r requirements.txt
```
3) Build package with `catkin_make`
4) Launch camera node, output `Image` message to `/camera/image` topic.
5) Run `detector.py`
```
rosrun yolov7_ros detector.py
```
6) Stream should start and detections output to `/detections` topic as `Detections` message type.
---

## Custom Message Types
### Detections
`Header header`  
`yolov7_ros/Detection[] dets` List of detections  
### Detections
`string name` Name of object  
`int8 conf` Confidence of detection  
`int16[4] bbox` Position of bounding box top-left and bottom-right `[x1,x2, y1,y2]`.  
