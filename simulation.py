from grasp_generator import GraspGenerator
from environment.utilities import Camera
from environment.env import Environment
from utils import YcbObjects, PackPileData, IsolatedObjData, summarize
import numpy as np
import pybullet as p
import argparse
import os
import sys
import cv2
import math
import matplotlib.pyplot as plt
import time


class GrasppingScenarios():

    def __init__(self,network_model="GGCNN"):
        
        self.network_model = network_model

        if (network_model == "GR_ConvNet"):
            ##### GR-ConvNet #####
            self.IMG_SIZE = 224
            self.network_path = 'trained_models/GR_ConvNet/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
            sys.path.append('trained_models/GR_ConvNet')
        elif (network_model == "GGCNN"):
            self.IMG_SIZE = 300
            self.network_path = 'trained_models/GG_CNN/epoch_50_cornell'
            sys.path.append('trained_models/GG_CNN')
        else:
            # you need to add your network here!
            print("The selected network has not been implemented yet!")
            exit() 
        
        
        self.CAM_Z = 1.9
        self.depth_radius = 1
        self.ATTEMPTS = 3
        self.fig = plt.figure(figsize=(10, 10))
        self.data = None
       
                
    def draw_predicted_grasp(self,grasps,color = [0,0,1],lineIDs = []):
        x, y, z, yaw, opening_len, obj_height = grasps

        gripper_size = opening_len + 0.02 
        finger_size = 0.075
        # lineIDs = []
        lineIDs.append(p.addUserDebugLine([x, y, z], [x, y, z+0.15],color, lineWidth=6))

        lineIDs.append(p.addUserDebugLine([x - gripper_size*math.sin(yaw), y - gripper_size*math.cos(yaw), z], 
                                    [x + gripper_size*math.sin(yaw), y + gripper_size*math.cos(yaw), z], 
                                    color, lineWidth=6))

        lineIDs.append(p.addUserDebugLine([x - gripper_size*math.sin(yaw), y - gripper_size*math.cos(yaw), z], 
                                    [x - gripper_size*math.sin(yaw), y - gripper_size*math.cos(yaw), z-finger_size], 
                                    color, lineWidth=6))
        lineIDs.append(p.addUserDebugLine([x + gripper_size*math.sin(yaw), y + gripper_size*math.cos(yaw), z], 
                                    [x + gripper_size*math.sin(yaw), y + gripper_size*math.cos(yaw), z-finger_size], 
                                    color, lineWidth=6))
        
        return lineIDs
    
    def remove_drawing(self,lineIDs):
        for line in lineIDs:
            p.removeUserDebugItem(line)
    
    def dummy_simulation_steps(self,n):
        for _ in range(n):
            p.stepSimulation()

    def is_there_any_object(self,camera):
        self.dummy_simulation_steps(10)
        rgb, depth, _ = camera.get_cam_img()
        #print ("min RGB = ", rgb.min(), "max RGB = ", rgb.max(), "rgb.avg() = ", np.average(rgb))
        #print ("min depth = ", depth.min(), "max depth = ", depth.max())
        if (depth.max()- depth.min() < 0.0025):
            return False
        else:
            return True                  
                    
    def isolated_obj_scenario(self,runs, device, vis, output, debug):

        objects = YcbObjects('objects/ycb_objects',
                            mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                            mod_stiffness=['Strawberry'])

        
        ## reporting the results at the end of experiments in the results folder
        self.data = IsolatedObjData(objects.obj_names, runs, 'twotable_results')

        ## camera settings: cam_pos, cam_target, near, far, size, fov
        center_x, center_y, center_z = 0.05, -0.52, self.CAM_Z
        camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (self.IMG_SIZE, self.IMG_SIZE), 40)

        center_x, center_y, center_z = 0.05, 0.6, self.CAM_Z
        exampleCamera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (self.IMG_SIZE, self.IMG_SIZE), 40)
        
        env = Environment(camera, exampleCamera, vis=vis, debug=debug, finger_length=0.06)
        
        generator = GraspGenerator(self.network_path, camera, self.depth_radius, self.fig, self.IMG_SIZE, self.network_model, device)
        exampleGenerator = GraspGenerator(self.network_path, exampleCamera, self.depth_radius, self.fig, self.IMG_SIZE, self.network_model, device)
        
        objects.shuffle_objects()
        for i in range(runs):
            print("----------- run ", i+1, " -----------")
            print ("network model = ", self.network_model)
            print ("size of input image (W, H) = (", self.IMG_SIZE," ," ,self.IMG_SIZE, ")")
            print(f" all objects {objects.obj_names}")

            ## Run the grasp experiment
            env.reset_robot()          
            env.remove_all_obj()  

            ## Init objects
            ### Isolated object - LEFT
            path, mod_orn, mod_stiffness = objects.get_obj_info(objects.obj_names[0])
            env.load_isolated_obj(path, mod_orn, mod_stiffness)

            ### Packed object - RIGHT
            number_of_objects = 5
            info = objects.get_n_first_obj_info(number_of_objects)
            if True:
                env.create_example_packed(info, 1.08)
            else:
                env.create_example_pile(info, 1.08)

            # Make sure the items have come to rest   
            self.dummy_simulation_steps(30)

            ### Isolated Grasp
            self.run_grasp_experiment(objects.obj_names[0], self.ATTEMPTS, camera, generator, env, i, vis)
            #env.robotToExamplePos()
            ### Pile/Packed Grasp
            self.run_grasp_experiment(objects.obj_names[0], self.ATTEMPTS, exampleCamera, exampleGenerator, env, i, vis)

            ## Write results to disk
            self.data.write_json(self.network_model)
            summarize(self.data.save_dir, runs, self.network_model)

    def run_grasp_experiment(self, obj_name, number_of_attempts, camera, generator, env, i, vis):
        number_of_failures = 0
        idx = 0 ## select the best grasp configuration
        failed_grasp_counter = 0
        while self.is_there_any_object(camera) and number_of_failures < number_of_attempts:     
            
            bgr, depth, _ = camera.get_cam_img()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        
            grasps, save_name = generator.predict_grasp(rgb, depth, n_grasps=number_of_attempts, show_output=output)
            if (grasps == []):
                self.dummy_simulation_steps(50)
                #print ("could not find a grasp point!")
                if failed_grasp_counter > 3:
                    print("Failed to find a grasp points > 3 times. Skipping.")
                    break
                    
                failed_grasp_counter += 1                 
                continue

            #print ("grasps.length = ", len(grasps))
            if (idx > len(grasps)-1):  
                print ("idx = ", idx)
                if len(grasps) > 0 :
                    idx = len(grasps)-1
                else:
                    number_of_failures += 1
                    continue    

            if vis:
                LID =[]
                for g in grasps:
                    LID = self.draw_predicted_grasp(g,color=[1,0,1],lineIDs=LID)
                time.sleep(0.5)
                self.remove_drawing(LID)
                self.dummy_simulation_steps(10)
                

            lineIDs = self.draw_predicted_grasp(grasps[idx])

            x, y, z, yaw, opening_len, obj_height = grasps[idx]
            succes_grasp, succes_target = env.grasp((x, y, z), yaw, opening_len, obj_height)

            self.data.add_try(obj_name)
            
            if succes_grasp:
                self.data.add_succes_grasp(obj_name)
            if succes_target:
                self.data.add_succes_target(obj_name)

            ## remove visualized grasp configuration 
            if vis:
                self.remove_drawing(lineIDs)

            env.reset_robot()
            
            if succes_target:
                number_of_failures = 0
                if vis:
                    debugID = p.addUserDebugText("success", [-0.0, -0.9, 0.8], [0,0.50,0], textSize=2)
                    time.sleep(0.25)
                    p.removeUserDebugItem(debugID)
                
                if save_name is not None:
                    os.rename(save_name + '.png', save_name + f'_SUCCESS_grasp{i}.png')
                

            else:
                number_of_failures += 1
                idx +=1        
                #env.reset_robot() 
                # env.remove_all_obj()                        
        
                if vis:
                    debugID = p.addUserDebugText("failed", [-0.0, -0.9, 0.8], [0.5,0,0], textSize=2)
                    time.sleep(0.25)
                    p.removeUserDebugItem(debugID)



def parse_args():
    parser = argparse.ArgumentParser(description='Grasping demo')

    parser.add_argument('--scenario', type=str, default='twotable', help='Grasping scenario (isolated/packed/pile)')
    parser.add_argument('--network', type=str, default='GR_ConvNet', help='Network model (GR_ConvNet/GGCNN)')

    parser.add_argument('--runs', type=int, default=1, help='Number of runs the scenario is executed')
    parser.add_argument('--attempts', type=int, default=3, help='Number of attempts in case grasping failed')

    parser.add_argument('--save-network-output', dest='output', type=bool, default=False,
                        help='Save network output (True/False)')

    parser.add_argument('--device', type=str, default='cpu', help='device (cpu/gpu)')
    parser.add_argument('--vis', type=bool, default=True, help='vis (True/False)')
    parser.add_argument('--report', type=bool, default=True, help='report (True/False)')

                        
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    output = args.output
    runs = args.runs
    ATTEMPTS = args.attempts
    device=args.device
    vis=args.vis
    report=args.report
    
    grasp = GrasppingScenarios(args.network)

    if args.scenario == 'isolated':
        grasp.isolated_obj_scenario(runs, device, vis, output=output, debug=False)
    # elif args.scenario == 'packed':
    #     grasp.packed_or_pile_scenario(runs, args.scenario, device, vis, output=output, debug=False)
    # elif args.scenario == 'pile':
    #     grasp.packed_or_pile_scenario(runs, args.scenario, device, vis, output=output, debug=False)
    # elif args.scenario == 'twotable':
    #     grasp.packed_or_pile_scenario(runs, args.scenario, device, vis, output=output, debug=False)
