from fileinput import filename
import argparse
import math
import sys
import time
from random import randrange
import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from tqdm import tqdm

from environment.env import Environment
from environment.utilities import Camera
from grasp_generator import GraspGenerator
from object_matching import ObjectMatching
from segmenter_kmeans import Segmenter as Segmenter_kmeans
from segmenter_watershed import Segmenter as Segmenter_watershed
from utils import IsolatedObjData, YcbObjects


class GrasppingScenarios():
    def __init__(self, grasp_network_model="GR_ConvNet", matching_network_model = 'mobileNetV2', segmentation_method='kmeans', scenario = 'packed'):
        self.scenario = scenario
        self.grasp_network_model = grasp_network_model
        self.matching_network_model = matching_network_model
        self.failedSegmentMatchCounter = 0

        if (grasp_network_model == "GR_ConvNet"):
            ##### GR-ConvNet #####
            self.IMG_SIZE = 224
            self.grasp_network_path = 'trained_models/GR_ConvNet/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
            sys.path.append('trained_models/GR_ConvNet')
        elif (grasp_network_model == "CGR_ConvNet"):
            ##### CGR-ConvNet #####
            self.IMG_SIZE = 224
            self.grasp_network_path = 'trained_models/3D_GDM-RSON/CNN/pretrained_model/model1'
            sys.path.append('trained_models/3D_GDM-RSON')
        else:
            print("The selected grasping network has not been implemented yet!")
            exit()

        if (matching_network_model == "GR_ConvNet"):
            ##### GR-ConvNet #####
            self.IMG_SIZE = 224
            self.matching_network_path = 'trained_models/GR_ConvNet/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
            sys.path.append('trained_models/GR_ConvNet')
        elif (matching_network_model == "CGR_ConvNet"):
            ##### CGR-ConvNet #####
            self.IMG_SIZE = 224
            self.matching_network_path = 'trained_models/3D_GDM-RSON/CNN/pretrained_model/model1'
            sys.path.append('trained_models/3D_GDM-RSON')
        elif (matching_network_model == 'mobileNetV2'):
            self.IMG_SIZE = 224
            self.matching_network_path = ''
        elif (matching_network_model == 'GOOD'):
            self.IMG_SIZE = 224
            self.matching_network_path = ''
        else:
            print("The selected matching network has not been implemented yet!")
            exit()
        
        self.segmentation_method = segmentation_method
        self.synthetic = False
        if segmentation_method == 'kmeans':
            self.segmenter = Segmenter_kmeans()
        elif segmentation_method == 'watershed':
            self.segmenter = Segmenter_watershed()
        elif segmentation_method == 'synthetic':
            self.segmenter = Segmenter_kmeans()
            self.synthetic = True
        else:
            print('The selected segmentation method has not been implemented yet!')
            exit()

        self.CAM_Z = 1.9
        self.depth_radius = 1
        self.ATTEMPTS = 3
        self.fig = plt.figure(figsize=(10, 10))
        self.data = None
        self.env = None
        self.experimentResults = []
        self.match_results = []

    def draw_predicted_grasp(self, grasps, color=[0, 0, 1], lineIDs=[]):
        x, y, z, yaw, opening_len, obj_height = grasps

        gripper_size = opening_len + 0.02
        finger_size = 0.075

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

        graspGenerator = GraspGenerator(self.grasp_network_path, camera, self.depth_radius, self.fig, self.IMG_SIZE, self.grasp_network_model, device)
        objectMatchingModel = ObjectMatching(self.matching_network_path, self.matching_network_model, device)

        for i in tqdm(range(runs)):
            objects.shuffle_objects()

            ## Run the grasp experiment
            self.env.reset_robot()
            self.env.remove_all_obj()

            number_of_objects = 5
            # Init objects
            # Example object - RIGHT
            # Randomly select example object out of the 5 objects in the pack
            exampleObjectNumber = randrange(0, number_of_objects)
            path, mod_orn, mod_stiffness = objects.get_obj_info(objects.obj_names[exampleObjectNumber])
            exampleOrn = self.env.load_example_obj(path, mod_orn, mod_stiffness)
            exampleID = -1 #exampleObjectNumber
            self.example_object_name = objects.obj_names[exampleObjectNumber]

            # Pile of objects - LEFT
            info = objects.get_n_first_obj_info(number_of_objects)
            
            self.target_object_names = []
            for i in range(number_of_objects):
                self.target_object_names.append(objects.obj_names[i])

            if self.scenario == 'packed':
                self.env.create_packed(info, exampleID, exampleOrn, names=self.target_object_names)
            elif self.scenario == 'pile':
                self.env.create_pile(info, names=self.target_object_names)

            matchingObjectID = self.env.obj_ids[exampleObjectNumber + 1]
            self.graspExampleFromObjectsExperiment(objects.obj_names[0], self.ATTEMPTS, exampleCamera, camera, graspGenerator, objectMatchingModel, vis, matchingObjectID)
        
        self.writeExperimentResults()

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

    def getExampleRepresentation(self, exampleCamera : Camera,objectMatchingModel : ObjectMatching):
        # First, capture an image with the example camera, and calculate its representation
        fullExampleBgr, fullExampleDepth, _ = exampleCamera.get_cam_img()
        fullExampleRgb = cv2.cvtColor(fullExampleBgr, cv2.COLOR_BGR2RGB)
        failed = False
        # change segmentation to 1 segment
        exampleSegments = Segmenter_kmeans().get_segmentations(fullExampleRgb, fullExampleDepth, 1)
        if len(exampleSegments) != 1:
            failed = True
            exampleRepresentation = 0
        else:
            _, exampleRgb, exampleDepth = exampleSegments[0]
            exampleRepresentation = objectMatchingModel.calculateRepresentation(exampleRgb, exampleDepth)
        return exampleRepresentation, failed
    
    def createGrasp(self,graspGenerator : GraspGenerator, segmentRGB, segmentDepth, pileDepth, idx, posX = 0, posY = 0):
        predictions, _ = graspGenerator.predict(segmentRGB, segmentDepth, n_grasps=3)
        failed = False
        grasps = []
        for grasp in predictions:
            success, x, y, z, roll, opening_len, obj_height = graspGenerator.grasp_to_robot_frame(grasp, pileDepth, posX, posY)
            if not success:
                failed = True
                return 0,0,failed 
            grasps.append((x, y, z, roll, opening_len, obj_height))

        if (grasps == []):
            self.dummy_simulation_steps(1)
            print("could not find a grasp point!")
            failed = True
            return 0,0,failed
  
        if (idx > len(grasps) - 1):
            print("idx = ", idx)
            if len(grasps) > 0:
                idx = len(grasps) - 1
            else:
                failed = True
                return 0,0,failed

        if vis:
            LID = []
            for g in grasps:
                LID = self.draw_predicted_grasp(g, color=[1, 0, 1], lineIDs=LID)
            time.sleep(3.5)
            self.remove_drawing(LID)
            self.dummy_simulation_steps(10)

        lineIDs = self.draw_predicted_grasp(grasps[idx])
        return grasps[idx], lineIDs, failed

    def manipulatePile(self, graspGenerator, segmentRgb, segmentDepth, pileDepth, idx, posX = 0, posY = 0):
        removeOneObject = True
        if removeOneObject: 
            predictedGrasp,lineIDs, failed = self.createGrasp(graspGenerator, segmentRgb, segmentDepth, pileDepth, idx, posX, posY)
            if failed: return True

            x, y, z, yaw, opening_len, obj_height = predictedGrasp
            _, succes_grasp = self.env.grasp((x, y, z), yaw, opening_len, obj_height, wrongPrediction = True)
            self.env.reset_robot()
            if vis: self.remove_drawing(lineIDs)
            if succes_grasp: return False
        else:
            self.env.changePile(violence = 10000)
            self.env.reset_robot()
            return False
        return True
        
    def drawResults(self, lineIDs, correctObjectManipulated):
        self.remove_drawing(lineIDs)
        if correctObjectManipulated:
                debugID = p.addUserDebugText("success", [-0.0, -0.9, 0.8], [0, 0.50, 0], textSize=2)
                time.sleep(0.25)
                p.removeUserDebugItem(debugID)
        else:
                debugID = p.addUserDebugText("failed", [-0.0, -0.9, 0.8], [0.5, 0, 0], textSize=2)
                time.sleep(0.25)
                p.removeUserDebugItem(debugID)

    def writeExperimentResults(self):
        if len(self.experimentResults) == 0:
            print("no experiment results to write")
            exit()
        
        header = ['run', 'numberOfFaultPredictedSegments', 'pileManipulations', 'wrongWorstPrediction', 'pileManipulationGraspFaults', 'correctObjectMatch', 'correctManipulations', 'correctGrasps', 'correctObjectManipulations', 'totalPredictions']
        fileName = 'results scenario ' + self.scenario + ' segmenter ' + self.segmentation_method + ', matching ' + self.matching_network_model + ', grasping network ' + self.grasp_network_model + '.csv'
        with open(fileName, 'w', encoding='UTF8', newline='') as f:
            csvFile = csv.writer(f)   
            csvFile.writerow(header)      
            for index, result in enumerate(self.experimentResults):
                result.insert(0, str(index))
                csvFile.writerow(result)

        if self.synthetic:
            if len(self.match_results) == 0:
                print('no matching results to write')
                exit()
            header = ['run', 'attempt', 'target', 'best', 'worst']
            fileName = 'matches scenario ' + self.scenario + ' .csv'
            with open(fileName, 'w', encoding='UTF8', newline='') as f:
                csvFile = csv.writer(f)
                csvFile.writerow(header)
                run = -1
                for result in self.match_results:
                    if result[0] == 0:
                        run += 1
                    line = [run] + result
                    csvFile.writerow(line)

    def graspExampleFromObjectsExperiment(self, obj_name, number_of_attempts, exampleCamera : Camera, camera : Camera, graspGenerator : GraspGenerator, objectMatchingModel : ObjectMatching, vis, matchingObjectID):
        number_of_failures = 0

        # for loging
        numberOfFaultPredictedSegments = 0
        pileManipulations = 0
        wrongWorstPrediction = 0
        pileManipulationGraspFaults = 0
        correctObjectMatch = 0
        correctManipulations = 0
        correctGrasps = 0
        correctObjectManipulations = 0
        totalPredictions = 0

        idx = 0  ## select the best grasp configuration

        manipulationAttempts = 0
        matchno = 0
        while number_of_failures < number_of_attempts:
            if not (self.is_there_any_object(camera) and self.is_there_any_object(exampleCamera)):
                number_of_failures +=1
                break
            
            # First, get the example representation
            exampleRepresentation, failed = self.getExampleRepresentation(exampleCamera, objectMatchingModel)
            if failed:
                number_of_failures += 1
                break
            
            # Next, capture an image with the objects camera, segment image and calculate several representations
            pileBgr, pileDepth, pileSegmentation = camera.get_cam_img()
            pileRgb = cv2.cvtColor(pileBgr, cv2.COLOR_BGR2RGB)
            unique = np.unique(pileSegmentation)
            trueNumberOfSegments = len(unique)-2
            syntheticSegmenter = self.synthetic
            if syntheticSegmenter:
                pileSegments = []
                pileSegmentNames = []
                (unique, counts) = np.unique(pileSegmentation, return_counts=True)
                for segmentIndex, count in enumerate(counts):
                    if count < 3000:
                        segmentRgbImage = np.zeros([224,224,3],dtype=np.uint8)
                        segmentRgbImage.fill(255)
                        segmentDepthImage = np.zeros([224,224],dtype=np.float32)
                        segmentDepthImage.fill(0.91182417)
                        for i in range(224):
                            for j in range(224):
                                pixel = pileSegmentation[i,j]
                                if pixel == unique[segmentIndex]:
                                    rgbValue = pileRgb[i,j][:]
                                    segmentRgbImage[i,j] = rgbValue
                                    depthValue = pileDepth[i,j]
                                    segmentDepthImage[i,j] = depthValue
                        name = self.env.obj_names[unique[segmentIndex]]
                        segment = self.segmenter.get_segmentations(segmentRgbImage, segmentDepthImage, 1)[0]
                        pileSegments.append(segment)
                        pileSegmentNames.append(name)
                predictedNumberOfSegments = len(pileSegments)
            else:
                pileSegments = self.segmenter.get_segmentations(pileRgb, pileDepth, "guess")
                predictedNumberOfSegments = len(pileSegments)
            if len(pileSegments) == 0:
                number_of_failures += 1
                break
            if int(trueNumberOfSegments) != int(predictedNumberOfSegments):
                numberOfFaultPredictedSegments +=1

            objectRepresentations = []
            for _, segment in enumerate(pileSegments):
                _, segmentRGB, segmentDepth = segment
                objectRepresentations.append(objectMatchingModel.calculateRepresentation(segmentRGB, segmentDepth))#, segmentDepth, n_grasps=3))

            # For each object representation, match it with the sample object representation
            realSegmentID = self.debugTruthObject(matchingObjectID, pileSegments, pileDepth, graspGenerator)
            bestPredictedID, worstPredictedID, uncertain = objectMatchingModel.matchExampleWithObjectRepresentation(exampleRepresentation, objectRepresentations, realSegmentID, targetNames=self.target_object_names, exampleName=self.example_object_name)
            if syntheticSegmenter:
                self.match_results.append([matchno, self.example_object_name, pileSegmentNames[bestPredictedID], pileSegmentNames[worstPredictedID]])
            matchno += 1
                #print('Target: ' + self.example_object_name + ', Best match: ' + pileSegmentNames[bestPredictedID] + ', Worst match: ' + pileSegmentNames[worstPredictedID])

            # if the match is uncertain, try to manipulate the pile by removing the worst object
            manipulationAttempts +=1
            if uncertain and manipulationAttempts <= 3:        
                pileManipulations +=1
                if worstPredictedID == realSegmentID: wrongWorstPrediction +=1
                pos, segmentRGB, segmentDepth = pileSegments[worstPredictedID]
                faultyGrasp = self.manipulatePile(graspGenerator, segmentRGB, segmentDepth, pileDepth, idx, int(pos[0]-111), int(pos[1]-111)) 
                if faultyGrasp: pileManipulationGraspFaults +=1
                continue

            totalPredictions += 1
            if realSegmentID == bestPredictedID: correctObjectMatch +=1

            pos, segmentRGB, segmentDepth = pileSegments[bestPredictedID]
            predictedGrasp, lineIDs, failed = self.createGrasp(graspGenerator, segmentRGB, segmentDepth, pileDepth, idx,  int(pos[0]) - 111, int(pos[1]) - 111)
            if failed:
                number_of_failures += 1
                continue

            x, y, z, yaw, opening_len, obj_height = predictedGrasp
            succes_grasp, succes_target = self.env.grasp((x, y, z), yaw, opening_len, obj_height)

            self.data.add_try(obj_name)
            if succes_grasp: correctGrasps +=1

            correctObjectManipulated = self.env.check_target_reached(matchingObjectID)
            if correctObjectManipulated: 
                correctObjectManipulations +=1
            
            self.env.reset_robot()
            ## remove visualized grasp configuration
            if vis: self.drawResults(lineIDs, correctObjectManipulated)

            if succes_target: 
                correctManipulations +=1
                break
            else:
                number_of_failures += 1

        self.experimentResults.append([numberOfFaultPredictedSegments, pileManipulations, wrongWorstPrediction, pileManipulationGraspFaults, correctObjectMatch, correctManipulations, correctGrasps, correctObjectManipulations, totalPredictions])
        

def parse_args():
    parser = argparse.ArgumentParser(description='Grasping demo')

    parser.add_argument('--scenario',
                        type=str,
                        default='pile',
                        help='Scenario (packed/pile)')
    parser.add_argument('--graspingNetwork',
                        type=str,
                        default='CGR_ConvNet',
                        help='Network model (GR_ConvNet/CGR_ConvNet)')
    parser.add_argument('--matchingNetwork',
                        type=str,
                        default='mobileNetV2',
                        help='Network model (GR_ConvNet/CGR_ConvNet/mobileNetV2/GOOD)')
    parser.add_argument('--segmentationMethod',
                        type=str,
                        default='kmeans',
                        help='Segmentation method (kmeans/watershed/synthetic)')                        
    parser.add_argument('--runs',
                        type=int,
                        default=10,
                        help='Number of runs the scenario is executed')
    parser.add_argument('--attempts',
                        type=int,
                        default=3,
                        help='Number of attempts in case grasping failed')
    parser.add_argument('--vis',
                        type=bool,
                        default=True,
                        help='vis (True/False)')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    runs = args.runs
    ATTEMPTS = args.attempts
    vis = args.vis

    grasp = GrasppingScenarios(args.graspingNetwork, args.matchingNetwork, args.segmentationMethod, args.scenario)
    grasp.graspExampleFromObjectsScenario(runs, device = 'cpu', vis = vis, debug=False)