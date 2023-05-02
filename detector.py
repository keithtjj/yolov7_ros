import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import rospy
from cv_bridge import CvBridge
import numpy as np
from std_msgs.msg import String, Header, Bool
from sensor_msgs.msg import Image
from yolov7_ros.msg import Detection, Detections

bridge = CvBridge()
det_list = []

det_pub = rospy.Publisher('/detections', Detections, queue_size=1)
pub_tare_toggle = rospy.Publisher('/toggle_tare', Bool, queue_size=5)

def callback(data):
    global poi_pose, det_list
    det_list = []
    raw = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
    cv2.imwrite('frame.png', raw)
    with torch.no_grad():
        detect('frame.png')

    #find doors
    yoloed = cv2.imread('frame2.png')
    lower_b = np.array([0,100,100])
    upper_b = np.array([0,130,130])
    mask = cv2.inRange(raw, lower_b, upper_b)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts[0]:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(yoloed,(x,y),(x+w,y+h),(0,255,0),2)
        area = cv2.contourArea(c)
        if area > 6000:
            obj = Detection(name='door', conf=100, bbox=[x, y, x+w, y+h])
            det_list.append(obj)
    header = Header(stamp=rospy.Time.now())
    det_pub.publish(Detections(header=header, dets=det_list))
    cv2.imshow('detector2', yoloed)
    cv2.waitKey(1)  # 1 millisecond

script_dir = Path( __file__ ).parent.absolute()
weights = script_dir.joinpath('yolov7.pt')
view_img, imgsz, trace, thresh = True, 320, True, 0.8

# Initialize
set_logging()
device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size

model = TracedModel(model, device, imgsz)

if half:
    model.half()  # to FP16

def detect(source, save_img=False):
    global weights, view_img, imgsz, trace
    # Set Dataloader
    
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        '''
        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img)[0]
        '''
        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=thresh)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    obj = Detection(name=names[int(cls)], conf=int(conf *100), bbox=bbox)
                    det_list.append(obj)  

            # Print time (inference + NMS)
            #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            #cv2.imshow('detector', im0)
            #cv2.waitKey(1)  # 1 millisecond
            cv2.imwrite('frame2.png', im0)
    #print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    rospy.init_node('yolo_detector')
    pub_tare_toggle.publish(Bool(True))
    rospy.Subscriber('/camera/image', Image, callback, queue_size=1, buff_size=2**24)
    rospy.spin()
