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

from network.hardware.device import get_device
#from network.inference.post_process import post_process_output
from network.utils.data.camera_data import CameraData
from network.utils.dataset_processing.grasp import detect_grasps
from network.utils.visualisation.plot import plot_results


class ObjectMatching:
    IMG_WIDTH = 224
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

    def calculateRepresentation(self, rgb, depth, n_grasps=1, show_output=False):
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
                # print(
                #     '\n--------------------------------------------------------------------'
                # )
                # print(internal)
                # print(
                #     '--------------------------------------------------------------------\n'
                # )
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

    def matchExampleWithObjectRepresentation(self, exampleRepresentation, objectRepresentations, realSegmentID):
        """
            function should return the index of objectRepresentations

        """
        distances = []
        for _, representation in enumerate(objectRepresentations):
            difference = torch.dist(exampleRepresentation, representation)
            #sumDifference = torch.sum(difference)
            #distance = math.sqrt(sumDifference)
            distances.append(difference)

        predictedSegmentID = int(np.argmin(np.array(distances)))
        if predictedSegmentID != realSegmentID:
            print()
        return predictedSegmentID