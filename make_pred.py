import dlib
import cv2
import os
from model import VGG19
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
from skimage import io
from skimage.transform import resize
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='PyTorch Fer2013 Prediction')
parser.add_argument('--input_img', type=str, default='test_images/1.jpg', help='the directory of your input image')
parser.add_argument('--output_img', type=str, default='test_images/results/1.jpg', help='the directory of your output image')
pred_parser = parser.parse_args()

#detect face and crop image
dnnFaceDetector = dlib.cnn_face_detection_model_v1('face_detection/mmod_human_face_detector.dat')
img = cv2.imread(pred_parser.input_img)
out_img = img.copy()
h, w, _ = img.shape
extract_results = dnnFaceDetector(img, 1)

cnt = 0
for faceRect in extract_results:
    x1 = faceRect.rect.left()
    y1 = faceRect.rect.top()
    x2 = faceRect.rect.right()
    y2 = faceRect.rect.bottom()
    cropped_image = img[y1:y2, x1:x2]
    cnt += 1
    cv2.imwrite('test_images/cropped_images/cropped_image_' + str(cnt) + '.jpg', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#make prediction on cropped images
transform_test = transforms.Compose([
    transforms.TenCrop(44),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

net = VGG19()
checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

class_list, score_list = [], []
for i in range(cnt):
  raw_img = io.imread('test_images/cropped_images/cropped_image_' + str(i+1) + '.jpg')
  gray = rgb2gray(raw_img)
  gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)
  img = gray[:, :, np.newaxis]
  img = np.concatenate((img, img, img), axis=2)
  img = Image.fromarray(img)
  inputs = transform_test(img)
  ncrops, c, h, w = np.shape(inputs)
  inputs = inputs.view(-1, c, h, w).cuda()
  inputs = Variable(inputs, volatile=True)
  outputs = net(inputs)
  outputs_avg = outputs.view(ncrops, -1).mean(0)  
  score = F.softmax(outputs_avg)
  _, predicted = torch.max(outputs_avg.data, 0)
  class_pred = class_names[int(predicted.cpu().numpy())]
  class_conf = 0
  for j in range(len(class_names)):
      if class_conf < score.data.cpu().numpy()[j]:
        class_conf = score.data.cpu().numpy()[j]
  class_list.append(class_pred)
  score_list.append(class_conf)
#output predicted images and bounding box
cnt1 = 0
for faceRect in extract_results:
    cnt1 += 1
    x1 = faceRect.rect.left()
    y1 = faceRect.rect.top()
    x2 = faceRect.rect.right()
    y2 = faceRect.rect.bottom()
    out_img = cv2.rectangle(out_img, (x1, y1), (x2, y2), [0, 0, 255], 2)
    out_img = cv2.putText(out_img, class_list[cnt1-1] + ": {:.2}".format(score_list[cnt1-1]), (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              1, (0, 0, 255), 2, cv2.LINE_AA)
    #cv2.imwrite('test_images/results/result_' + str(i) + '.jpg', out_img)
    cv2.imwrite(pred_parser.output_img, out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()