import argparse
import os
import platform
import shutil
import time
import numpy as np
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch.nn.functional as F
from numpy import random

from PIL import Image
import PIL.ImageOps

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.datasets import letterbox

def exif_transpose(img):
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img

def load_image_file(file, mode='RGB'):
    # Load the image with PIL
    img = PIL.Image.open(file)

    if hasattr(PIL.ImageOps, 'exif_transpose'):
        # Very recent versions of PIL can do exit transpose internally
        img = PIL.ImageOps.exif_transpose(img)
    else:
        # Otherwise, do the exif transpose ourselves
        img = exif_transpose(img)

    img = img.convert(mode)

    return img


@torch.no_grad()
def predict_img_se(img0, model, device):
    resize = 250
    crop_size = 240
    transform = transforms.Compose([
        transforms.Resize(size=(resize, resize), interpolation=2),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    img = transform(img0)
    img = img.unsqueeze(0)
    model.to(device)
    img = img.to(device)
    aux1, aux2, aux3, out1, out2 = model(img)

    return aux1, aux2, aux3, out1, out2

def detect_person(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    source = r"C:\Users\jasne\Desktop\smoking_final\my_data\train\smoking_calling"

    # save_path = r'C:\Users\jasne\Desktop\train_1\calling'
    save_path = r'C:\Users\jasne\Desktop\train_1\smoking_calling'

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
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
        save_img = False
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    

    start_time = time.time()
    for path, img, im0s, vid_cap in dataset:
        # for file in images:
        start = time.time()
        # img = cv2.imread(path)
        # img0 = load_image_file(path)
        height, width = im0s.shape[:2]
        # print(width, height)
        file = os.path.split(path)[-1]
        file_name = os.path.join(save_path, file)
        print(file_name)
        if max(width, height) <= 500:          
            # img = Image.open(path).convert('RGB')
            cv2.imwrite(file_name, im0s)
            
        else:
            # To tensor
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        
            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                # save_path = str(Path(out) / Path(p).name)
                # txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                # s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # print(det)
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label = (names[int(cls)])
                        if label == "person":
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            # print(xyxy)
                            # c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3])) 
                            pad = 50
                            x, y, w, h =int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                            # if (w - x) < width / 4:
                            #     continue
                            # else:
                            x, y, w, h = max(0, x-pad), max(0, y-pad), min(width, w+pad), min(height, h+pad)
                            image = im0s[y:h, x:w]
                else:
                    image = im0s.copy()

            cv2.imwrite(file_name, image)

            # print(type(image))

            # 测试得到的新图像
            # cv2.imshow('img', image)
            # if cv2.waitKey(0) == ord('q'):  # q to quit
            #     raise StopIteration
                    
def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # source = opt.source
    # images = os.listdir(source)
    # images.sort(key=lambda x: int(x.split('.')[0]))

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
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
        save_img = False
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Load model
    # newnet = torch.load('./checkpoints/newnet_se_expand_2/4_023_0.7146.pt')
    newnet = torch.load("./checkpoints/4_085_0.7906.pt")
    print('Load NewNet Done!!!')

    class_2_index = {0: 'calling', 1: 'normal', 2: 'smoking', 3: 'smoking_calling'}

    result_list = []
    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    start_time = time.time()
    for path, img, im0s, vid_cap in dataset:
        # for file in images:
        start = time.time()
        # img = cv2.imread(path)
        img0 = load_image_file(path)
        height, width = im0s.shape[:2]
        # print(width, height)
        file = os.path.split(path)[-1]

        if max(width, height) <= 372:          
            # img = Image.open(path).convert('RGB')
            aux1, aux2, aux3, out1, out2 = predict_img_se(img0, newnet, device)             
            # 1 output = out1 + out2
            output = (out1 + out2) / 2
            preds = F.softmax(output, dim=1) # compute softmax 
            torch_prob, index = torch.max(preds, 1)
            torch_predict = class_2_index[int(index)]
            torch_prob = torch_prob.item()
            print(' -> {}: {}, prob: {}, elapse: {}s'.format(file, torch_predict, torch_prob, time.time() - start))

            result_data = {'image_name': str(file), 'category': torch_predict, 'score': float(torch_prob)}
            result_list.append(result_data)
            
        else:
            # To tensor
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        
            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                # save_path = str(Path(out) / Path(p).name)
                # txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                # s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label = (names[int(cls)])
                        if label == "person":
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            # print(xyxy)
                            # c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3])) 
                            pad = 50
                            x, y, w, h =int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            x, y, w, h = max(0, x-pad), max(0, y-pad), min(width, w+pad), min(height, h+pad)
                            image = im0s[y:h, x:w]
                else:
                    image = im0s.copy()

            # print(type(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image.astype(np.uint8))

            aux1, aux2, aux3, out1, out2 = predict_img_se(image, newnet, device)             
            # 1 output = out1 + out2
            output = (out1 + out2) / 2
            preds = F.softmax(output, dim=1) # compute softmax 
            torch_prob, index = torch.max(preds, 1)
            torch_predict = class_2_index[int(index)]
            torch_prob = torch_prob.item()
            print(' -> {}: {}, prob: {}, elapse: {}s'.format(file, torch_predict, torch_prob, time.time() - start))

            result_data = {'image_name': str(file), 'category': torch_predict, 'score': float(torch_prob)}
            result_list.append(result_data)
            # 测试得到的新图像
            # cv2.imshow('img', image)
            # if cv2.waitKey(0) == ord('q'):  # q to quit
            #     raise StopIteration
    
    # 把结果排序
    result_list.sort(key=lambda x: int(x['image_name'].split('.')[0]))
    elapse = time.time() - start_time
    print(f'Elapse: {elapse}s')

    # 把结果写入json
    import json
    save_dir = 'result'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = save_dir + os.sep + 'result_yolo_newnet_eca_cbma_expand.json'
    with open(filename, 'w') as file_obj:
        json.dump(result_list, file_obj)
    print('Saved Result!!! {}'.format(filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=r"C:\Users\jasne\Desktop\testA", help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
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
            detect()
