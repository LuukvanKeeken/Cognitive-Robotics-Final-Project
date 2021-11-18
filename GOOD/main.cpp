// Pointcloud related includes
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>

// GOOD
#include "good.cpp"

// Other includes
#include <iostream>
#include <sstream>

// Namespaces
using namespace std;
using namespace pcl;

typedef pcl::PointXYZRGBA PointT;

int getGood(boost::shared_ptr<PointCloud<PointT>> pointcloud, int bins, float thresh){
    GOODEstimation<PointT> GOOD_descriptor(bins, thresh); // Initialize GOOD
    GOOD_descriptor.setInputCloud(pointcloud); // Add pointcloud
    std::vector<float> object_description; // Store output
    GOOD_descriptor.compute(object_description); // Compute output

    // Print as python array
    cout << '[';
    for(size_t i = 0; i < object_description.size() - 1; ++i){
        std::cout << object_description.at(i) << ',';
    }
    std::cout << object_description.back() << "]" << std::endl;
}

int main(int argc, const char *argv[]){
    std::string path = argv[1];
    int bins = 0;
    float thresh = 0.0;
    if(argc>2){
        istringstream convertStream(argv[2]);
        convertStream >> bins;
    }
    if(argc>3){
        istringstream convertStream(argv[3]);
        convertStream >> thresh;
    }

    boost::shared_ptr<PointCloud<PointT>> target_pc (new PointCloud<PointT>);

    if(io::loadPCDFile<PointT>(path.c_str(), *target_pc) == -1){
        cout << "No PCD at path %s :" << path.c_str();
        return(0);
    }
    getGood(target_pc, bins, thresh);

    return 0;
}