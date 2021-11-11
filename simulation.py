import argparse
import math
import os
import sys
import time
from random import randrange

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import torch
from tqdm import tqdm

from environment.env import Environment
from environment.utilities import Camera
from grasp_generator import GraspGenerator
from object_matching import ObjectMatching
from segmenter import Segmenter
from utils import IsolatedObjData, PackPileData, YcbObjects, summarize


class GrasppingScenarios():
    def __init__(self, network_model="GGCNN"):

        self.network_model = network_model
        self.failedSegmentMatchCounter = 0

        if (network_model == "GR_ConvNet"):
            ##### GR-ConvNet #####
            self.IMG_SIZE = 224
            self.network_path = 'trained_models/GR_ConvNet/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
            sys.path.append('trained_models/GR_ConvNet')
        elif (network_model == "CGR_ConvNet"):
            ##### CGR-ConvNet #####
            self.IMG_SIZE = 224
            self.network_path = 'trained_models/3D_GDM-RSON/CNN/pretrained_model/model1'
            sys.path.append('trained_models/3D_GDM-RSON')
        else:
            print("The selected network has not been implemented yet!")
            exit()

        self.CAM_Z = 1.9
        self.depth_radius = 1
        self.ATTEMPTS = 3
        self.fig = plt.figure(figsize=(10, 10))
        self.data = None
        self.env = None

    def draw_predicted_grasp(self, grasps, color=[0, 0, 1], lineIDs=[]):
        x, y, z, yaw, opening_len, obj_height = grasps

        gripper_size = opening_len + 0.02
        finger_size = 0.075
        # lineIDs = []
        lineIDs.append(
            p.addUserDebugLine([x, y, z], [x, y, z + 0.15], color,
                               lineWidth=6))

        lineIDs.append(
            p.addUserDebugLine([
                x - gripper_size * math.sin(yaw),
                y - gripper_size * math.cos(yaw), z
            ], [
                x + gripper_size * math.sin(yaw),
                y + gripper_size * math.cos(yaw), z
            ],
                               color,
                               lineWidth=6))

        lineIDs.append(
            p.addUserDebugLine([
                x - gripper_size * math.sin(yaw),
                y - gripper_size * math.cos(yaw), z
            ], [
                x - gripper_size * math.sin(yaw),
                y - gripper_size * math.cos(yaw), z - finger_size
            ],
                               color,
                               lineWidth=6))
        lineIDs.append(
            p.addUserDebugLine([
                x + gripper_size * math.sin(yaw),
                y + gripper_size * math.cos(yaw), z
            ], [
                x + gripper_size * math.sin(yaw),
                y + gripper_size * math.cos(yaw), z - finger_size
            ],
                               color,
                               lineWidth=6))

        return lineIDs

    def remove_drawing(self, lineIDs):
        for line in lineIDs:
            p.removeUserDebugItem(line)

    def dummy_simulation_steps(self, n):
        for _ in range(n):
            p.stepSimulation()

    def is_there_any_object(self, camera):
        self.dummy_simulation_steps(10)
        _, depth, _ = camera.get_cam_img()
        if (depth.max() - depth.min() < 0.0025):
            return False
        else:
            return True

    def graspExampleFromObjectsScenario(self, runs, device, vis, debug):
        objects = YcbObjects('objects/ycb_objects',
            mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
            mod_stiffness=['Strawberry'])

        ## reporting the results at the end of experiments in the results folder
        self.data = IsolatedObjData(objects.obj_names, runs, 'pickSameProductScenario')

        ## camera settings: cam_pos, cam_target, near, far, size, fov
        center_x, center_y, center_z = 0.05, -0.52, self.CAM_Z
        camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (self.IMG_SIZE, self.IMG_SIZE), 40)

        center_x, center_y, center_z = 0.05, 0.6, self.CAM_Z
        exampleCamera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (self.IMG_SIZE, self.IMG_SIZE), 40)

        self.env = Environment(camera, exampleCamera, vis=vis, debug=debug, finger_length=0.06)

        graspGenerator = GraspGenerator(self.network_path, camera, self.depth_radius, self.fig, self.IMG_SIZE, self.network_model, device)
        objectMatchingModel = ObjectMatching(self.network_path, self.network_model, device)

        for i in tqdm(range(runs)):
            objects.shuffle_objects()

            ## Run the grasp experiment
            self.env.reset_robot()
            self.env.remove_all_obj()

            # Init objects
            # Example object - RIGHT
            # Randomly select example object out of the 5 objects in the pack
            exampleObjectNumber = randrange(0, 5)
            path, mod_orn, mod_stiffness = objects.get_obj_info(objects.obj_names[exampleObjectNumber])
            exampleOrn = self.env.load_example_obj(path, mod_orn, mod_stiffness)

            exampleID = -1 #exampleObjectNumber
            
            # Pile of objects - LEFT
            number_of_objects = 5
            info = objects.get_n_first_obj_info(number_of_objects)
            self.env.create_packed(info, exampleID, exampleOrn)

            matchingObjectID = self.env.obj_ids[exampleObjectNumber + 1]
            self.graspExampleFromObjectsExperiment(objects.obj_names[0], self.ATTEMPTS, exampleCamera, camera, graspGenerator, objectMatchingModel, vis, matchingObjectID)

            ## Write results to disk
            #self.data.write_json(self.network_model)
            #summarize(self.data.save_dir, runs, self.network_model)
        print(f"failed segmented matches: {self.failedSegmentMatchCounter}")

    def debugTruthObject(self, matchingObjectID, pileSegments, depth_img, generator):
        truthPos, _ = p.getBasePositionAndOrientation(matchingObjectID)
        distances = []
        for _, segment in enumerate(pileSegments):
            pos, _, _ = segment     
            xObject, yObject, _ = generator.pixelsToRobotFrame(int(pos[0]), int(pos[1]), depth_img)
            xTruth, yTruth = truthPos[0], truthPos[1]
            difX = xObject - xTruth
            difY = yObject - yTruth
            distances.append(math.hypot(difX, difY))
        return int(np.argmin(np.array(distances)))

    def graspExampleFromObjectsExperiment(self, obj_name, number_of_attempts, exampleCamera : Camera, camera : Camera, graspGenerator : GraspGenerator, objectMatchingModel : ObjectMatching, vis, matchingObjectID):
        number_of_failures = 0
        idx = 0  ## select the best grasp configuration
        failed_grasp_counter = 0
        finished = False

        while self.is_there_any_object(camera) and self.is_there_any_object(exampleCamera) and number_of_failures < number_of_attempts and not finished:
            segmenter = Segmenter()
            # First, capture an image with the example camera, and calculate its representation
            fullExampleBgr, fullExampleDepth, _ = exampleCamera.get_cam_img()
            fullExampleRgb = cv2.cvtColor(fullExampleBgr, cv2.COLOR_BGR2RGB)

            # change segmentation to 1 segment
            exampleSegments = segmenter.get_segmentations(fullExampleRgb, fullExampleDepth, 1)
            if len(exampleSegments) != 1:
                number_of_failures += 1
                break
            _, exampleRgb, exampleDepth = exampleSegments[0]
            exampleRepresentation = objectMatchingModel.calculateRepresentation(exampleRgb, exampleDepth, n_grasps=3)

            # Next, capture an image with the objects camera, segment image and calculate several representations
            pileBgr, pileDepth, _ = camera.get_cam_img()
            pileRgb = cv2.cvtColor(pileBgr, cv2.COLOR_BGR2RGB)

            pileSegments = segmenter.get_segmentations(pileRgb, pileDepth, 5)
            if len(pileSegments) == 0:
                number_of_failures += 1
                break

            objectRepresentations = []
            for _, segment in enumerate(pileSegments):
                _, segmentRGB, segmentDepth = segment
                objectRepresentations.append(objectMatchingModel.calculateRepresentation(segmentRGB, segmentDepth, n_grasps=3))

            # For each object representation, match it with the sample object representation
            realSegmentID = self.debugTruthObject(matchingObjectID, pileSegments, pileDepth, graspGenerator)
            predictedSegmentID = objectMatchingModel.matchExampleWithObjectRepresentation(exampleRepresentation, objectRepresentations, realSegmentID)
            
            if realSegmentID != predictedSegmentID:
                wrongPrediction = True
                self.failedSegmentMatchCounter += 1
            else:
                wrongPrediction = False

            pos, segmentRGB, segmentDepth = pileSegments[predictedSegmentID]
            posX = int(pos[0]) - 111
            posY = int(pos[1]) - 111

            predictions, _ = graspGenerator.predict(segmentRGB, segmentDepth, n_grasps=3)

            grasps = []
            for grasp in predictions:
                x, y, z, roll, opening_len, obj_height = graspGenerator.grasp_to_robot_frame(grasp, pileDepth, posX, posY)
                #x, y, z, roll, opening_len, obj_height = graspGenerator.grasp_to_robot_frame(grasp, segmentDepth, posX, posY)
                grasps.append((x, y, z, roll, opening_len, obj_height))

            if (grasps == []):
                self.dummy_simulation_steps(1)
                print("could not find a grasp point!")
                if failed_grasp_counter > 3:
                    print("Failed to find a grasp points > 3 times. Skipping.")
                    break
                failed_grasp_counter += 1
                continue

            if (idx > len(grasps) - 1):
                print("idx = ", idx)
                if len(grasps) > 0:
                    idx = len(grasps) - 1
                else:
                    number_of_failures += 1
                    continue

            if vis:
                LID = []
                for g in grasps:
                    LID = self.draw_predicted_grasp(g, color=[1, 0, 1], lineIDs=LID)
                time.sleep(3.5)
                self.remove_drawing(LID)
                self.dummy_simulation_steps(10)
                #return

            lineIDs = self.draw_predicted_grasp(grasps[idx])

            x, y, z, yaw, opening_len, obj_height = grasps[idx]
            succes_grasp, succes_target = self.env.grasp((x, y, z), yaw, opening_len, obj_height, wrongPrediction = wrongPrediction)

            self.data.add_try(obj_name)

            if succes_grasp:
                self.data.add_succes_grasp(obj_name)
            if succes_target:
                self.data.add_succes_target(obj_name)

            ## remove visualized grasp configuration
            if vis:
                self.remove_drawing(lineIDs)

            self.env.reset_robot()

            if succes_target:
                number_of_failures = 0
                if vis:
                    debugID = p.addUserDebugText("success", [-0.0, -0.9, 0.8], [0, 0.50, 0], textSize=2)
                    time.sleep(0.25)
                    p.removeUserDebugItem(debugID)
                    break
            else:
                number_of_failures += 1
                idx += 1
                if vis:
                    debugID = p.addUserDebugText("failed", [-0.0, -0.9, 0.8], [0.5, 0, 0], textSize=2)
                    time.sleep(0.25)
                    p.removeUserDebugItem(debugID)

def parse_args():
    parser = argparse.ArgumentParser(description='Grasping demo')

    parser.add_argument('--scenario',
                        type=str,
                        default='twotable',
                        help='Grasping scenario (isolated/packed/pile)')
    parser.add_argument('--network',
                        type=str,
                        default='GR_ConvNet',
                        help='Network model (GR_ConvNet/CGR_ConvNet)')

    parser.add_argument('--runs',
                        type=int,
                        default=10,
                        help='Number of runs the scenario is executed')
    parser.add_argument('--attempts',
                        type=int,
                        default=3,
                        help='Number of attempts in case grasping failed')

    parser.add_argument('--save-network-output',
                        dest='output',
                        type=bool,
                        default=False,
                        help='Save network output (True/False)')

    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='device (cpu/gpu)')
    parser.add_argument('--vis',
                        type=bool,
                        default=True,
                        help='vis (True/False)')
    parser.add_argument('--report',
                        type=bool,
                        default=True,
                        help='report (True/False)')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    output = args.output
    runs = args.runs
    ATTEMPTS = args.attempts
    device = args.device
    vis = args.vis
    report = args.report

    grasp = GrasppingScenarios(args.network)

    if args.scenario == 'twotable':
        grasp.graspExampleFromObjectsScenario(runs, device, vis, debug=False)