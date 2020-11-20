# coding=utf-8
import argparse
import os
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from Controller.MainController import dbInsOrUpd
import Controller.MainController as MainController
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized



def generateCentroid(rects):
    inputCentroids = np.zeros((len(rects), 2), dtype="int")
    for (i, (startX, startY, endX, endY)) in enumerate(rects):
        cX = int((startX + endX) / 2.0)
        cY = int((startY + endY) / 2.0)
        inputCentroids[i] = (cX, cY)
    return inputCentroids

##TestSimon
def detect(save_img):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    totalCarAmount = MainController.getLatestCarAmount()
    displayAmount = totalCarAmount
    oldCombinedAmount = 0
    combinedAmount = 0
    tempAmount = 0
    control = False


    elapsed = 0
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    start = time.time()
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)

    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    #colors = [[np.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference

    t0 = time.time()
    ct = CentroidTracker()
    listDet = ['car', 'motorcycle', 'truck']

    totalDownCar = 0
    totalDownMotor = 0
    totalDownTruck = 0

    totalUpCar = 0
    totalUpMotor = 0
    totalUpTruck = 0
    trackableObjects = {}






    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        elapsed = time.time() - start
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        rects = []
        labelObj = []
        arrCentroid = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                #cv2.resize(im0, (2560, 1440))
            else:
                p, s, im0 = path, '', im0s
                #cv2.resize(im0, (2560, 1440))

            height, width, channels = im0.shape
            cv2.line(im0, (0, int(height / 1.5)), (int(width), int(height / 1.5)), (255, 0, 0), thickness=3)

            if not control:
                cv2.putText(im0, 'Totale koeretoejer: ' + str(displayAmount), (int(width * 0.02),
                                                                                int(height * 0.4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 255), 2)
            else:
                cv2.putText(im0, 'Totale koeretoejer: ' + str(displayAmount), (int(width * 0.02),
                                                                               int(height * 0.4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (50, 255, 255), 3)

            #cv2.line(im0, (int(width / 1.8), int(height / 1.5)), (int(width), int(height / 1.5)), (255, 127, 0), thickness=3)

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % (names[int(cls)], conf)
                    # print(xyxy)
                    x = xyxy
                    tl = None or round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1  # line/font thickness
                    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                    label1 = label.split(' ')
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if label1[0] in listDet:
                        cv2.rectangle(im0, c1, c2, (0, 0, 0), thickness=tl, lineType=cv2.LINE_AA)
                        box = (int(x[0]), int(x[1]), int(x[2]), int(x[3]))
                        rects.append(box)
                        labelObj.append(label1[0])
                        tf = max(tl - 1, 1)
                        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                        cv2.rectangle(im0, c1, c2, (0, 100, 0), -1, cv2.LINE_AA)
                        cv2.putText(im0, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                                    lineType=cv2.LINE_AA)

                detCentroid = generateCentroid(rects)
                objects = ct.update(rects)

                for (objectID, centroid) in objects.items():
                    arrCentroid.append(centroid[1])
                for (objectID, centroid) in objects.items():
                    # print(idxDict)
                    to = trackableObjects.get(objectID, None)
                    if to is None:
                        to = TrackableObject(objectID, centroid)
                    else:
                        y = [c[1] for c in to.centroids]
                        direction = centroid[1] - np.mean(y)
                        to.centroids.append(centroid)
                        if not to.counted:  # arah up

                            if direction < 0 and centroid[1] < height / 1.5 and centroid[
                                1] > height / 1.7:  ##up truble when at distant car counted twice because bbox reappear
                                idx = detCentroid.tolist().index(centroid.tolist())
                                if (labelObj[idx] == 'car'):
                                    totalUpCar += 1
                                    to.counted = True
                                elif (labelObj[idx] == 'motorbike'):
                                    totalUpMotor += 1
                                    to.counted = True
                                elif (labelObj[idx] == 'truck'):
                                    totalUpTruck += 1
                                    to.counted = True

                            elif direction > 0 and centroid[1] > height / 1.5:  # arah down
                                idx = detCentroid.tolist().index(centroid.tolist())
                                if (labelObj[idx] == 'car'):
                                    totalDownCar += 1
                                    to.counted = True
                                elif (labelObj[idx] == 'motorbike'):
                                    totalDownMotor += 1
                                    to.counted = True
                                elif (labelObj[idx] == 'truck'):
                                    totalDownTruck += 1
                                    to.counted = True

                    trackableObjects[objectID] = to

                oldCarAmount = totalCarAmount

                combinedAmount = totalDownCar + totalDownTruck + totalDownMotor + \
                                     totalUpCar + totalUpMotor + totalUpTruck
                if not oldCombinedAmount == combinedAmount:
                    tempAmount = totalCarAmount + combinedAmount
                    oldCombinedAmount = combinedAmount

                if oldCarAmount < tempAmount:
                    totalCarAmount = tempAmount

                if not oldCarAmount == totalCarAmount:
                    dbInsOrUpd(totalCarAmount)
                    displayAmount += 1

                if not control:
                    cv2.putText(im0, 'Frakoerende: ',
                                (int(width * 0.6), int(height * 0.10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 255), 2)
                    cv2.putText(im0, 'Bil: ' + str(totalUpCar), (int(width * 0.6), int(height * 0.15)),
                                cv2.FONT_HERSHEY_SIMPLEX, .75, (50, 255, 255), 2)
                    cv2.putText(im0, 'Motorcykel: ' + str(totalUpMotor), (int(width * 0.6), int(height * 0.2)),
                                cv2.FONT_HERSHEY_SIMPLEX, .75, (50, 255, 255), 2)
                    cv2.putText(im0, 'Lastbil: ' + str(totalUpTruck), (int(width * 0.6), int(height * 0.25)),
                                cv2.FONT_HERSHEY_SIMPLEX, .75, (50, 255, 255), 2)

                    cv2.putText(im0, 'Modkoerende: ',
                                (int(width * 0.02), int(height * 0.10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 255), 2)
                    cv2.putText(im0, 'Bil: ' + str(totalDownCar), (int(width * 0.02), int(height * 0.15)),
                                cv2.FONT_HERSHEY_SIMPLEX, .75, (50, 255, 255), 2)
                    cv2.putText(im0, 'Motorcykel: ' + str(totalDownMotor), (int(width * 0.02), int(height * 0.2)),
                                cv2.FONT_HERSHEY_SIMPLEX, .75, (50, 255, 255), 2)
                    cv2.putText(im0, 'Lastbil: ' + str(totalDownTruck), (int(width * 0.02), int(height * 0.25)),
                                cv2.FONT_HERSHEY_SIMPLEX, .75, (50, 255, 255), 2)
                else:
                    cv2.putText(im0, 'Frakoerende: ',
                                (int(width * 0.6), int(height * 0.10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 4, (50, 255, 255), 3)
                    cv2.putText(im0, 'Bil: ' + str(totalUpCar), (int(width * 0.6), int(height * 0.15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (50, 255, 255), 3)
                    cv2.putText(im0, 'Motorcykel: ' + str(totalUpMotor), (int(width * 0.6), int(height * 0.2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (50, 255, 255), 3)
                    cv2.putText(im0, 'Lastbil: ' + str(totalUpTruck), (int(width * 0.6), int(height * 0.25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (50, 255, 255), 3)

                    cv2.putText(im0, 'Modkoerende: ',
                                (int(width * 0.02), int(height * 0.10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 4, (50, 255, 255), 3)
                    cv2.putText(im0, 'Bil: ' + str(totalDownCar), (int(width * 0.02), int(height * 0.15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (50, 255, 255), 3)
                    cv2.putText(im0, 'Motorcykel: ' + str(totalDownMotor), (int(width * 0.02), int(height * 0.2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (50, 255, 255), 3)
                    cv2.putText(im0, 'Lastbil: ' + str(totalDownTruck), (int(width * 0.02), int(height * 0.25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (50, 255, 255), 3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.namedWindow('Main', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Main', 1920, 1080)
                cv2.imshow("Main", im0)

                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)

                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    #parser.add_argument('--source', type=str, default='inference/videos/test.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect(False)
