import os
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from numpy.lib.npyio import save
from PIL import Image
from skimage.filters import gaussian
from torch import nn
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
#import tensorflow.compat.v2 as tf

from network.hardware.device import get_device
#from network.inference.post_process import post_process_output
from network.utils.data.camera_data import CameraData
from network.utils.dataset_processing.grasp import detect_grasps
from network.utils.visualisation.plot import plot_results


class ObjectMatching:
    IMG_WIDTH = 224
    IMG_SHAPE = (IMG_WIDTH, IMG_WIDTH, 3)
    IMG_ROTATION = -np.pi * 0.5
    CAM_ROTATION = 0
    PIX_CONVERSION = 277
    DIST_BACKGROUND = 1.115
    MAX_GRASP = 0.085

    def __init__(self, net_path, network='GR_ConvNet', device='cpu'):
        if (device == 'cpu'):
            self.net = torch.load(net_path, map_location=device)
            self.device = get_device(force_cpu=True)
        else:
            #self.net = torch.load(net_path, map_location=lambda storage, loc: storage.cuda(1))
            #self.device = get_device()
            print(
                "GPU is not supported yet! :( -- continuing experiment on CPU!"
            )
            self.net = torch.load(net_path, map_location='cpu')
            self.device = get_device(force_cpu=True)

        if network == 'GR_ConvNet':
            self.internal_representation = nn.Sequential(
                *list(self.net.children())[:-13])
        elif network == 'CGR_ConvNet':
            # print(*list(self.net.children()))
            self.internal_representation = nn.Sequential(
                *list(self.net.children())[:6],
                nn.AvgPool2d((3, 3), stride=(1, 1), padding=(1, 1)),
                nn.Flatten())
        else:
            print('Other networks not supported yet!')
        self.network = network
        self.base_model = tf.keras.applications.MobileNetV2(input_shape=self.IMG_SHAPE, include_top=False, weights='imagenet')
        self.base_model.trainable = False

    def post_process_output(self, q_img, cos_img, sin_img, width_img, pixels_max_grasp):
        """
        Post-process the raw output of the network, convert to numpy arrays, apply filtering.
        :param q_img: Q output of network (as torch Tensors)
        :param cos_img: cos output of network
        :param sin_img: sin output of network
        :param width_img: Width output of network
        :return: Filtered Q output, Filtered Angle output, Filtered Width output
        """
        q_img = q_img.cpu().numpy().squeeze()
        ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
        width_img = width_img.cpu().numpy().squeeze() * pixels_max_grasp

        q_img = gaussian(q_img, 1.0, preserve_range=True)
        ang_img = gaussian(ang_img, 1.0, preserve_range=True)
        width_img = gaussian(width_img, 1.0, preserve_range=True)

        return q_img, ang_img, width_img
    
    def crop_image(self, img, xy, scale_factor):
        '''Crop the image around the tuple xy

        Inputs:
        -------
        img: Image opened with PIL.Image
        xy: tuple with relative (x,y) position of the center of the cropped image
            x and y shall be between 0 and 1
        scale_factor: the ratio between the original image's size and the cropped image's size
        '''
        center = (img.size[0] * xy[0], img.size[1] * xy[1])
        new_size = (img.size[0] / scale_factor, img.size[1] / scale_factor)
        left = max (0, (int) (center[0] - new_size[0] / 2))
        right = min (img.size[0], (int) (center[0] + new_size[0] / 2))
        upper = max (0, (int) (center[1] - new_size[1] / 2))
        lower = min (img.size[1], (int) (center[1] + new_size[1] / 2))
        cropped_img = img.crop((left, upper, right, lower))
        return cropped_img

    # def createHeatMap(self, rgb_image, depth_image, exampleRepresentation):
    #     cutoff = np.median(depth_image)
    #     heatmap = np.zeros([224,224],dtype=np.uint8)
    #     lastImage = img = np.zeros([224,224,3],dtype=np.uint8)
    #     lastDistance = 0
    #     interval = 10
    #     intervalStep=0

    #     for i in range(len(depth_image)):
    #         row = depth_image[i]
    #         for j in range(len(row)):
    #             if (depth_image[i][j] >= cutoff):
    #                 rgb_image[i][j] = [255,255,255]
    #     plt.imshow(rgb_image, interpolation='nearest')
    #     plt.show()
    

    #     for i in range(224):
    #         if i == 50: print("i is 50")
    #         if i == 100: print("i is 100")
    #         if i == 150: print("i is 150")
    #         if i == 200: print("i is 200")
    #         for j in range(224):
    #             if depth_image[i,j]<cutoff:
    #                 intervalStep +=1
    #                 if intervalStep > interval:
    #                     intervalStep = 0
    #                     img = np.zeros([224,224,3],dtype=np.uint8)
    #                     img.fill(255)
    #                     for k in range(224):
    #                         for l in range(224):
    #                             relativeX = int((k-111)/4)
    #                             relativeY = int((l-111)/4)
    #                             pixelX = i+relativeX
    #                             pixelY = j+relativeY
    #                             if pixelX >= 0 and pixelX < 224 and pixelY >= 0 and pixelY < 224:
    #                                 #color = [50,100,50]
    #                                 color = rgb_image[pixelX, pixelY]
    #                                 img[k,l] = color
    #                             # 
    #                             # if pixelX < 0 or pixelX >= 224 or pixelY < 0 or pixelY >= 224:
    #                             #     img[k,l,0], img[k,l,1], img[k,l,2] = 0,0,0
    #                             # else:
    #                             #     img[k,l,0], img[k,l,1], img[k,l,2] = rgb_image[i,j, 0], rgb_image[i,j, 1], rgb_image[i,j, 2]
    #                     if np.array_equal(lastImage,img):
    #                         lastImage = img
    #                         distance = lastDistance
    #                     else:
    #                         representation = self.calculateRepresentation(img)
    #                         distance = np.linalg.norm(exampleRepresentation-representation)
    #                         lastDistance = distance
    #                         lastImage = img
    #                     heatmap[i,j] = distance
    #     heatmap = ((heatmap + 0.1) * (1/0.3) * 255).astype('uint8')
    #     plt.imshow(heatmap, interpolation='nearest')
    #     plt.show()

    #                 #xy = [111,111]
    #                 #newImage = self.crop_image(img, xy, 2)
    #                 #plt.imshow(newImage, interpolation='nearest')
    #                 #plt.show()


    def calculateMobileNetRepresentation(self, rgb):
        input = rgb.reshape(1,224,224,3)
        #input = np.array(rgb)
        featureVector = self.base_model.predict(input)
        return featureVector

        # bese_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        #     input_shape=None, alpha=1.0, include_top=True, weights='imagenet',
        #     input_tensor=None, pooling=None, classes=1000,
        #     classifier_activation='softmax', **kwargs
        # )

    def calculateConvNetRepresentation(self, rgb, depth, n_grasps=1, show_output=False):
        max_val = np.max(depth)
        depth = depth * (255 / max_val)
        depth = np.clip((depth - depth.mean()) / 175, -1, 1)

        if (self.network == 'GR_ConvNet'):
            ##### GR-ConvNet #####
            depth = np.expand_dims(np.array(depth), axis=2)
            img_data = CameraData(width=self.IMG_WIDTH, height=self.IMG_WIDTH)
            x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)
        elif (self.network == 'CGR_ConvNet'):
            ##### GR-ConvNet #####
            depth = np.expand_dims(np.array(depth), axis=2)
            img_data = CameraData(width=self.IMG_WIDTH, height=self.IMG_WIDTH)
            x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)
        else:
            print("The selected network has not been implemented yet -- please choose another network!")
            exit()

        with torch.no_grad():
            xc = x.to(self.device)
            if (self.network == 'GR_ConvNet'):
                ##### GR-ConvNet #####
                pred = self.net.predict(xc)
                internal = self.internal_representation(xc)
                pixels_max_grasp = int(self.MAX_GRASP * self.PIX_CONVERSION)
                q_img, ang_img, width_img = self.post_process_output(
                    pred['pos'], pred['cos'], pred['sin'], pred['width'],
                    pixels_max_grasp)
            elif (self.network == 'CGR_ConvNet'):
                ##### CGR-ConvNet #####
                pred = self.net(xc)
                internal = self.internal_representation(xc)
                pixels_max_grasp = int(self.MAX_GRASP * self.PIX_CONVERSION)
                q_img, ang_img, width_img = self.post_process_output(
                    pred[0], pred[1], pred[2], pred[3], pixels_max_grasp)
            elif (self.network == 'GGCNN'):
                pred = self.net(xc)
                pixels_max_grasp = int(self.MAX_GRASP * self.PIX_CONVERSION)
                q_img, ang_img, width_img = self.post_process_output(
                    pred[0], pred[1], pred[2], pred[3], pixels_max_grasp)
            else:
                print("you need to add your function here!")

        return internal

    def matchExampleWithObjectRepresentation(self, exampleRepresentation, objectRepresentations, realSegmentID=0):
        """
            function should return the index of objectRepresentations

        """
        distances = []
        for _, representation in enumerate(objectRepresentations):
            difference = np.linalg.norm(exampleRepresentation-representation)
            #difference = torch.dist(exampleRepresentation, representation)
            #sumDifference = torch.sum(difference)
            #distance = math.sqrt(sumDifference)
            distances.append(difference)

        distances = np.array(distances)
        bestPredictedID = int(np.argmin(distances))
        worstPredictedID = int(np.argmax(distances))

        differences = np.abs(distances-np.median(distances))
        medianDifferences = np.median(differences)

        if differences[bestPredictedID]/medianDifferences < 2:
            uncertain = True
        else:
            uncertain = False
        
        return bestPredictedID, worstPredictedID, uncertain