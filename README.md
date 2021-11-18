## Finding and Grasping Objects Based on Visual Examples
For requirements
    import requirements.txt
    compile the c++-files of the folder "GOOD" with CMakeList.txt

Start experiment
    An experiment can be started by executing the file "simulation.py". With Several arguments can be given 

        --scenario = pile
                    the scenarios of the objects for which the experiment will be executed. 
                    Implementen are 'packed' and 'pile'.
        --graspingNetwork = CGR_ConvNet
                    the grasping network which is used to generate the grasps. 
                    Implemented are 'GR_ConvNet' and 'CGR_ConvNet'.
        --matchingNetwork = CGR_ConvNet
                    the model which is used to calculate the object representations. 
                    Implemented are 'GR_ConvNet', 'CGR_ConvNet', 'mobileNetV2', and 'GOOD'.
        --segmentationMethod = CGR_ConvNet
                    the segmentation which is used to segment the objects on the table. 
                    Implemented are 'kmeans', 'watershed', and 'synthetic'. 
                    In case of 'synthetic', the ground truth segments from the PyBullet segmenter will be used.
        --runs = 10
                    The number of runs that the experiment is repeated.
        --attempts = 3
                    The number of attempts to manipulate the object into the target position until the experiment is terminated. 
        --vis = True
                    Enables visualization. Set False to speed up simulation.