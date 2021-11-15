import math
import numpy as np
import torch.utils.data
from torch import nn
import tensorflow as tf
from network.hardware.device import get_device
from network.utils.data.camera_data import CameraData
import subprocess
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances


class ObjectMatching:
    IMG_WIDTH = 224
    IMG_SHAPE = (IMG_WIDTH, IMG_WIDTH, 3)
    IMG_ROTATION = -np.pi * 0.5
    CAM_ROTATION = 0
    PIX_CONVERSION = 277
    DIST_BACKGROUND = 1.115
    MAX_GRASP = 0.085

    def __init__(self, net_path = '', network='GR_ConvNet', device='cpu'):
        if network == 'GR_ConvNet' or network == 'CGR_ConvNet':
            if (device == 'cpu'):
                self.net = torch.load(net_path, map_location=device)
                self.device = get_device(force_cpu=True)
            else:
                #self.net = torch.load(net_path, map_location=lambda storage, loc: storage.cuda(1))
                #self.device = get_device()
                print("GPU is not supported yet! :( -- continuing experiment on CPU!")
                self.net = torch.load(net_path, map_location='cpu')
                self.device = get_device(force_cpu=True)
            if network == 'GR_ConvNet':
                ##### GR-ConvNet #####
                self.internal_representation = nn.Sequential(*list(self.net.children())[:-13])
            elif network == 'CGR_ConvNet':
                ##### CGR-ConvNet #####
                self.internal_representation = nn.Sequential(*list(self.net.children())[:6],
                    nn.AvgPool2d((3, 3), stride=(1, 1), padding=(1, 1)), nn.Flatten())
        elif network == 'mobileNetV2':
            ##### mobileNetV2 #####
            self.base_model = tf.keras.applications.MobileNetV2(input_shape=self.IMG_SHAPE, include_top=False, weights='imagenet')
            self.base_model.trainable = False
        elif network == 'GOOD':
            ##### GOOD #####
            self.base_model = 0 # implement GOOD
        else:
            print('Other networks not supported yet!')
        self.representationNetwork = network

    def calculateRepresentation(self, rgb, depth, name=''):
        depth_raw = depth
        max_val = np.max(depth)
        depth = depth * (255 / max_val)
        depth = np.clip((depth - depth.mean()) / 175, -1, 1)

        if (self.representationNetwork == 'GR_ConvNet'):
            ##### GR-ConvNet #####
            depth = np.expand_dims(np.array(depth), axis=2)
            img_data = CameraData(width=self.IMG_WIDTH, height=self.IMG_WIDTH)
            x, _, _ = img_data.get_data(rgb=rgb, depth=depth)
        elif (self.representationNetwork == 'CGR_ConvNet'):
            ##### GR-ConvNet #####
            depth = np.expand_dims(np.array(depth), axis=2)
            img_data = CameraData(width=self.IMG_WIDTH, height=self.IMG_WIDTH)
            x, _, _ = img_data.get_data(rgb=rgb, depth=depth)
        elif (self.representationNetwork == 'mobileNetV2'):
            ##### mobileNetV2 #####
            input = rgb.reshape(1,224,224,3)
        elif (self.representationNetwork == 'GOOD'):
            ##### GOOD #####
            rgb_raw = o3d.geometry.Image(rgb)
            depth_raw = o3d.geometry.Image(depth_raw)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_raw,depth_raw,convert_rgb_to_intensity=False,depth_scale=1.0
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(
                    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
            # Flip it, otherwise the pointcloud will be upside down
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            filepath = './pointclouds/image.pcd'
            o3d.io.write_point_cloud(filepath, pcd, write_ascii=True)
            input = filepath # implement GOOD
        else:
            print("The selected network has not been implemented yet -- please choose another network!")
            exit()

        if self.representationNetwork == 'mobileNetV2':
            ##### mobileNetV2 #####
            internal = self.base_model.predict(input)
        elif self.representationNetwork == 'GOOD':
            ##### GOOD #####
            internal = self.get_good(input) # implement GOOD
            print(internal)
        else:
            with torch.no_grad():
                xc = x.to(self.device)
                if (self.representationNetwork == 'GR_ConvNet'):
                    ##### GR-ConvNet #####
                    internal = self.internal_representation(xc)
                elif (self.representationNetwork == 'CGR_ConvNet'):
                    ##### CGR-ConvNet #####
                    internal = self.internal_representation(xc)
                else:
                    print("The selected network has not been implemented yet -- please choose another network!")
                    exit()
        return internal

    def matchExampleWithObjectRepresentation(self, exampleRepresentation, objectRepresentations, realSegmentID=0, exampleName='', targetNames=['']):
        """
            function should return the index of objectRepresentations

        """
        distances = []
        for _, representation in enumerate(objectRepresentations):
            if self.representationNetwork == 'GR_ConvNet' or  self.representationNetwork == 'CGR_ConvNet':
                difference = torch.dist(exampleRepresentation, representation)
                sumDifference = torch.sum(difference)
                distance = math.sqrt(sumDifference)
            elif self.representationNetwork == 'mobileNetV2':
                ##### mobileNetV2 #####
                distance = np.linalg.norm(exampleRepresentation-representation)
            elif self.representationNetwork == 'GOOD':
                ##### GOOD #####
                distance = euclidean_distances(np.array(exampleRepresentation).reshape(1, -1), np.array(representation).reshape(1, -1))[0][0]
                #print(f'Euclidean distance between {targetNames[i]} and example {exampleName} is {distance}')
            else:
                print('distance function for this network is not implemented')
                exit()
            distances.append(distance)
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
    
    def get_good(self, path):
        '''
        Gets GOOD descriptor of a pointcloud by calling a C++ executable.
        :param path: Location of the .pcd pointcloud
        :return: GOOD descriptor histogram of the pointcloud
        '''
        subproc = subprocess.Popen(('./GOOD/build/goodexe', path, '5', '0.015'), stdout=subprocess.PIPE)
        subproc.wait()
        return eval(subproc.stdout.read().decode(encoding='utf-8'))