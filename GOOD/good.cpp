/*
 * Software License Agreement (BSD License)
 *
 *  	Hamidreza Kasaei - http://wiki.ieeta.pt/wiki/index.php/Hamidreza_Kasaei
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *
 */

#include "good.h"

/** \brief GOOD: a Global Orthographic Object Descriptor for 3D object recognition and manipulation.
  * GOOD descriptor has been designed to be robust, descriptive and efficient to compute and use. 
  * It has two outstanding characteristics: 
  * 
  * (1) Providing a good trade-off among :
  *	- descriptiveness,
  *	- robustness,
  *	- computation time,
  *	- memory usage.
  * 
  * (2) Allowing concurrent object recognition and pose estimation for manipulation.
  * 
  * \note This is an implementation of the GOOD descriptor which has been presented in the following papers:
  * 
  *	[1] Kasaei, S. Hamidreza,  Ana Maria Tom??, Lu??s Seabra Lopes, Miguel Oliveira 
  *	"GOOD: A global orthographic object descriptor for 3D object recognition and manipulation." 
  *	Pattern Recognition Letters 83 (2016): 312-320.http://dx.doi.org/10.1016/j.patrec.2016.07.006
  *
  *	[2] Kasaei, S. Hamidreza, Lu??s Seabra Lopes, Ana Maria Tom??, Miguel Oliveira 
  * 	"An orthographic descriptor for 3D object learning and recognition." 
  *	2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Daejeon, 2016, 
  *	pp. 4158-4163. doi: 10.1109/IROS.2016.7759612
  * 
  * Please adequately refer to this work any time this code is being used by citing above papers.
  * If you do publish a paper where GOOD descriptor helped your research, we encourage you to cite the above papers in your publications.
  * 
  * \author Hamidreza Kasaei (Seyed.Hamidreza[at]ua[dot]pt)
  */

template <typename PointInT>
GOODEstimation<PointInT>::GOODEstimation()
{
  number_of_bins_ = 5;
  threshold_ = 0.0015;
}

template <typename PointInT>
GOODEstimation<PointInT>::GOODEstimation(unsigned int number_of_bins, float threshold)
{
  number_of_bins_ = number_of_bins;
  threshold_ = threshold;
};
    
/*________________________________
|                               |
|           Methods             |
|_______________________________| */

template <typename PointInT>
void 
GOODEstimation<PointInT>::setInputCloud (boost::shared_ptr<pcl::PointCloud<PointInT> > cloud)
{ 
  input_ = cloud; 
};


template <typename PointInT>
void
GOODEstimation<PointInT>::setNumberOfBins (unsigned int number_of_bins)
{
  number_of_bins_ = number_of_bins;
}

template <typename PointInT>
void
GOODEstimation<PointInT>::setThreshold (float threshold)
{
  threshold_ = threshold;
}

template <typename PointInT>
void
GOODEstimation<PointInT>::getOrthographicProjections (std::vector < boost::shared_ptr<pcl::PointCloud<PointInT> > > &vector_of_projected_views)
{
  vector_of_projected_views = vector_of_projected_views_;  
}

template <typename PointInT>
void
GOODEstimation<PointInT>::getTransformedObject (boost::shared_ptr<pcl::PointCloud<PointInT> > &transformed_point_cloud)
{
  transformed_point_cloud = transformed_point_cloud_;
} 

template <typename PointInT>
void
GOODEstimation<PointInT>::getCenterOfObjectBoundingBox (pcl::PointXYZ &center_of_bbox)
{
  center_of_bbox = center_of_bbox_;
}

template <typename PointInT>     
void
GOODEstimation<PointInT>::getObjectBoundingBoxDimensions(pcl::PointXYZ &bbox_dimensions)
{
  bbox_dimensions = bbox_dimensions_;
}

template <typename PointInT>  
void 
GOODEstimation<PointInT>::getOrderOfProjectedPlanes(std::string &order_of_projected_plane)
{
  order_of_projected_plane = order_of_projected_plane_;
}

template <typename PointInT>
void 
GOODEstimation<PointInT>::getTransformationMatrix( Eigen::Matrix4f &transformation)
{
  transformation = transformation_;
} 

template <typename PointInT>
void 
GOODEstimation<PointInT>::computeBoundingBoxDimensions (boost::shared_ptr<pcl::PointCloud<PointInT> > pc, pcl::PointXYZ& dimensions)
{
  PointInT minimum_pt;
  PointInT maximum_pt;
  pcl::getMinMax3D(*pc, minimum_pt, maximum_pt); // min max for bounding box
  dimensions.x = (maximum_pt.x - minimum_pt.x); 
  dimensions.y = (maximum_pt.y - minimum_pt.y); 
  dimensions.z = (maximum_pt.z - minimum_pt.z); 	
}

template <typename PointInT>
void
GOODEstimation<PointInT>::projectPointCloudToPlane (boost::shared_ptr<pcl::PointCloud<PointInT> > pc_in, boost::shared_ptr<pcl::ModelCoefficients> coefficients, boost::shared_ptr<pcl::PointCloud<PointInT> > pc_out)
{
  //Create the projection object
  pcl::ProjectInliers<PointInT> projection;
  projection.setModelType(pcl::SACMODEL_NORMAL_PLANE); //set model type
  projection.setInputCloud(pc_in);
  projection.setModelCoefficients(coefficients);
  projection.filter(*pc_out);
}

template <typename PointInT>
void 
GOODEstimation<PointInT>::convert2DHistogramTo1DHistogram (std::vector <std::vector <unsigned int> >  histogram_2D, std::vector <unsigned int>  &histogram)
{
  for (size_t i=0 ; i < histogram_2D.size(); i++)
  {   
    for (size_t j=0; j < histogram_2D.at(i).size(); j++)
    {
      histogram.push_back(histogram_2D.at(i).at(j));
    }
  }
}

template <typename PointInT>
void
GOODEstimation<PointInT>::signDisambiguationXAxis (boost::shared_ptr<pcl::PointCloud<PointInT> >  XoZ_projected_view, float threshold, int &sign )
{
  int Xpositive =0; 
  int Xnegative =0; 
  unsigned int thereshold = trunc (XoZ_projected_view -> points.size()/10);

  //XoZ page
  for (size_t i=0; i< XoZ_projected_view -> points.size(); i++)
  {	
    if (XoZ_projected_view->points.at(i).x > threshold)
    {
      Xpositive ++;
    }
    else if (XoZ_projected_view->points.at(i).x < - threshold)
    {
      Xnegative ++;
    }	  
  }  
  if ((Xpositive < Xnegative) and (Xnegative - Xpositive >= thereshold)) 
  {
    sign = -1;
  }
  else 
  {
    sign = 1;
  }		  
}

template <typename PointInT>
void 
GOODEstimation<PointInT>::signDisambiguationYAxis (boost::shared_ptr<pcl::PointCloud<PointInT> >  YoZ_projected_view, float threshold, int &sign )
{
  int Ypositive =0; 
  int Ynegative =0; 
  unsigned int thereshold = trunc (YoZ_projected_view -> points.size()/10);
  //YoZ page
  for (size_t i=0; i < YoZ_projected_view -> points.size(); i++)
  {	
    if (YoZ_projected_view->points.at(i).y > threshold)
    {
      Ypositive ++;
    }
    else if (YoZ_projected_view->points.at(i).y < - threshold)
    {
      Ynegative ++;
    }
  }  
  if ((Ypositive < Ynegative) and (Ynegative - Ypositive >= thereshold))
  {
    sign = -1;
  }
  else 
  {
    sign = 1;
  }		  
}
 
 
template <typename PointInT>
void
GOODEstimation<PointInT>::create2DHistogramFromYOZProjection (boost::shared_ptr<pcl::PointCloud<PointInT> >  YOZ_projected_view,
			double largest_side, unsigned int number_of_bins, int sign, std::vector < std::vector<unsigned int> > &YOZ_histogram)
{
  
  double x = largest_side/2; 
  double y = largest_side/2; 
  double z = largest_side/2; 
  double interval_x = largest_side/number_of_bins; 
  double interval_y = largest_side/number_of_bins; 
  double interval_z = largest_side/number_of_bins; 
  for (size_t i=0; i < YOZ_projected_view-> points.size(); i++)
  {
    pcl::PointXYZ p;
    p.y = sign * YOZ_projected_view->points.at(i).y +  y;
    p.z = YOZ_projected_view ->points.at(i).z + z;
    
    //if adaptive_support_lenght parameter == false, some points might be projected outside of the plane, we must discard them.
    if ((trunc(p.y / interval_y) < YOZ_histogram.size()) and (trunc(p.z / interval_z) < YOZ_histogram.at(0).size())
	  and (trunc(p.y / interval_y) >= 0) and (trunc(p.z / interval_z) >= 0))
    {
      YOZ_histogram.at(trunc(p.y / interval_y)).at(trunc(p.z / interval_z))++;
    }
  }    
}

template <typename PointInT>
void 
GOODEstimation<PointInT>::create2DHistogramFromXOZProjection ( boost::shared_ptr<pcl::PointCloud<PointInT> >  XOZ_projected_view, double largest_side, 
				      unsigned int number_of_bins, int sign, std::vector < std::vector<unsigned int> > &XOZ_histogram)
{  
  double x = largest_side/2; 
  double y = largest_side/2; 
  double z = largest_side/2; 
  double interval_x = largest_side/number_of_bins; 
  double interval_y = largest_side/number_of_bins; 
  double interval_z = largest_side/number_of_bins; 
  for (size_t i=0; i < XOZ_projected_view-> points.size(); i++)
  {
    pcl::PointXYZ p;
    p.x =sign *  XOZ_projected_view->points.at(i).x +  x;    
    p.z = XOZ_projected_view ->points.at(i).z + z;		
    
    if ((trunc(p.x / interval_x) < XOZ_histogram.size()) and (trunc(p.z / interval_z) < XOZ_histogram.at(0).size())
	  and (trunc(p.x / interval_x) >=0) and (trunc(p.z / interval_z) >=0))
    {
      XOZ_histogram.at(trunc(p.x / interval_x)).at(trunc(p.z / interval_z))++;
    }
  }  
}

template <typename PointInT>
void 
GOODEstimation<PointInT>::create2DHistogramFromXOYProjection (boost::shared_ptr<pcl::PointCloud<PointInT> >  XOY_projected_view,	double largest_side,
				      unsigned int number_of_bins, int sign, std::vector < std::vector<unsigned int> > &XOY_histogram)
{    
  double x = largest_side/2; 
  double y = largest_side/2; 
  double z = largest_side/2; 
  double interval_x = largest_side/number_of_bins; 
  double interval_y = largest_side/number_of_bins; 
  double interval_z = largest_side/number_of_bins; 
  for (size_t i=0; i < XOY_projected_view-> points.size(); i++)
  {
    pcl::PointXYZ p;
    p.x = sign * XOY_projected_view->points.at(i).x + x;
    p.y = sign * XOY_projected_view->points.at(i).y + y;        
    if ((trunc(p.x / interval_x) < XOY_histogram.size()) and (trunc(p.y / interval_y) < XOY_histogram.at(0).size())and 
	  (trunc(p.x / interval_x) >= 0) and (trunc(p.y / interval_y) >=0))
    {
	XOY_histogram.at(trunc(p.x / interval_x)).at(trunc(p.y / interval_y))++;
    }
  }
}

template <typename PointInT>
void
GOODEstimation<PointInT>::normalizingHistogram (std::vector <unsigned int> histogram, std::vector <float> &normalized_histogram)
{
  int sum_all_bins = 0;    
  float normalizing_bin = 0;
  //compute sumation of all histogram's bins.
  for(std::vector<unsigned int>::iterator it = histogram.begin(); it != histogram.end(); ++it)
  {
    sum_all_bins += *it;
  }
  if (sum_all_bins != 0)
  {  
    for(std::vector<unsigned int>::iterator it = histogram.begin(); it != histogram.end(); ++it)
    {
      normalized_histogram.push_back(*it/ float (sum_all_bins));
    }  
  }
}

template <typename PointInT>
void
GOODEstimation<PointInT>::viewpointEntropy (std::vector <float> normalized_histogram, float &entropy)
{
  //http://stats.stackexchange.com/questions/66108/why-is-entropy-maximised-when-the-probability-distribution-is-uniform
  entropy =0;  
  for(std::vector<float>::iterator it = normalized_histogram.begin(); it != normalized_histogram.end(); ++it)
  {
    if (*it != 0)
    {
      float entropy_tmp = *it * log2(*it);
      entropy += entropy_tmp;
    }
  }
  entropy = -entropy;  
}

template <typename PointInT>
void
GOODEstimation<PointInT>::findMaxViewPointEntropy (std::vector <float> view_point_entropy, int &index)
{
  index = 0;
  std::vector<float>::iterator it;
  it=std::max_element(view_point_entropy.begin(),view_point_entropy.end());
  index = it - view_point_entropy.begin();
  //std::cout << "\nindex ="<< it - view_point_entropy.begin() <<"\t , content = " << *it <<"\n";  
}

template <typename PointInT>  
void
GOODEstimation<PointInT>::averageHistograms (std::vector< float> histogram1, std::vector< float> historam2, std::vector< float> historam3, std::vector< float> &average)
{
  for (size_t i=0; i <histogram1.size(); i++ )
  {
    average.push_back(float(histogram1.at(i)+historam2.at(i)+historam3.at(i))/float(3.00));
  }
}

template <typename PointInT>
void
GOODEstimation<PointInT>::meanOfHistogram (std::vector< float> histogram, float &mean)
{ 	    
  // http://www.stat.yale.edu/Courses/1997-98/101/rvmnvar.htm
  float mu = 0;
  for (size_t i = 0; i < histogram.size(); i++)
  {
    mu += (i+1)*histogram.at(i);
  }
  mean = mu;  
}

template <typename PointInT>
void
GOODEstimation<PointInT>::varianceOfHistogram (std::vector< float> histogram, float mean, float &variance)
{
  //https://people.richland.edu/james/lecture/m170/ch06-prb.html
  //http://www.stat.yale.edu/Courses/1997-98/101/rvmnvar.htm  
  float variance_tmp = 0;
  for (size_t i = 0; i < histogram.size(); i++)
  {
    variance_tmp += pow((i+1)-mean,2)*histogram.at(i);
  }
  variance = variance_tmp;
}

template <typename PointInT>
void
GOODEstimation<PointInT>::objectViewHistogram (int maximum_entropy_index, std::vector< std::vector<float> >normalized_projected_views,
		    std::vector< float> &sorted_normalized_projected_views,
		    std::string &name_of_sorted_projected_plane /*debug*/)
{
  float variance1 = 0;
  float variance2 = 0;
  float mean =0;

  sorted_normalized_projected_views.insert ( sorted_normalized_projected_views.end(), normalized_projected_views.at(maximum_entropy_index).begin(), 
			      normalized_projected_views.at(maximum_entropy_index).end());    
   
  switch (maximum_entropy_index)
  {
    case 0 :
      
      name_of_sorted_projected_plane += "YoZ - ";      
      meanOfHistogram(normalized_projected_views.at(1), mean);
      varianceOfHistogram(normalized_projected_views.at(1), mean, variance1);
      meanOfHistogram(normalized_projected_views.at(2), mean);
      varianceOfHistogram(normalized_projected_views.at(2), mean, variance2);
      
      if (variance1 <= variance2)
      {
	name_of_sorted_projected_plane += "XoZ - XoY ";
	sorted_normalized_projected_views.insert ( sorted_normalized_projected_views.end(), normalized_projected_views.at(1).begin(),normalized_projected_views.at(1).end());	
	sorted_normalized_projected_views.insert ( sorted_normalized_projected_views.end(), normalized_projected_views.at(2).begin(),normalized_projected_views.at(2).end());
      }
      else
      {		     
	name_of_sorted_projected_plane += "XoY - XoZ ";
	sorted_normalized_projected_views.insert ( sorted_normalized_projected_views.end(), normalized_projected_views.at(2).begin(),normalized_projected_views.at(2).end());
	sorted_normalized_projected_views.insert ( sorted_normalized_projected_views.end(), normalized_projected_views.at(1).begin(),normalized_projected_views.at(1).end());
      }
      break;
      
    case 1 :
      name_of_sorted_projected_plane += "XoZ - ";      
      meanOfHistogram(normalized_projected_views.at(0), mean);
      varianceOfHistogram(normalized_projected_views.at(0), mean, variance1);      
      meanOfHistogram(normalized_projected_views.at(2), mean);
      varianceOfHistogram(normalized_projected_views.at(2), mean, variance2);
            
      if (variance1 <= variance2)
      {
	name_of_sorted_projected_plane += "YoZ - XoY ";
	sorted_normalized_projected_views.insert ( sorted_normalized_projected_views.end(),normalized_projected_views.at(0).begin(),normalized_projected_views.at(0).end());
	sorted_normalized_projected_views.insert ( sorted_normalized_projected_views.end(),normalized_projected_views.at(2).begin(),normalized_projected_views.at(2).end());
      }
      else
      {
        name_of_sorted_projected_plane += "XoY - YoZ ";
	sorted_normalized_projected_views.insert ( sorted_normalized_projected_views.end(), normalized_projected_views.at(2).begin(),normalized_projected_views.at(2).end());
	sorted_normalized_projected_views.insert ( sorted_normalized_projected_views.end(), normalized_projected_views.at(0).begin(),normalized_projected_views.at(0).end());
      }
      break;
	
    case 2 :
      name_of_sorted_projected_plane += "XoY - ";		      
      meanOfHistogram(normalized_projected_views.at(0), mean);
      varianceOfHistogram(normalized_projected_views.at(0), mean, variance1);
      meanOfHistogram(normalized_projected_views.at(1), mean);
      varianceOfHistogram(normalized_projected_views.at(1), mean, variance2);

      if (variance1 <= variance2)
      {
	name_of_sorted_projected_plane += "YoZ - XoZ ";
	sorted_normalized_projected_views.insert ( sorted_normalized_projected_views.end(),normalized_projected_views.at(0).begin(),normalized_projected_views.at(0).end());
	sorted_normalized_projected_views.insert ( sorted_normalized_projected_views.end(),normalized_projected_views.at(1).begin(),normalized_projected_views.at(1).end());
      }
      else
      {
	name_of_sorted_projected_plane += "XoZ - YoZ ";
	sorted_normalized_projected_views.insert ( sorted_normalized_projected_views.end(), normalized_projected_views.at(1).begin(), normalized_projected_views.at(1).end());
	sorted_normalized_projected_views.insert ( sorted_normalized_projected_views.end(), normalized_projected_views.at(0).begin(), normalized_projected_views.at(0).end());
      }
      break;  
      
    default:
      break;
  }
}

template <typename PointInT>
void
GOODEstimation<PointInT>::computeLargestSideOfBoundingBox (pcl::PointXYZ dimensions, double &largest_side )
{    
  std::vector<double> tmp;
  tmp.push_back (dimensions.x);  tmp.push_back (dimensions.y);  tmp.push_back (dimensions.z);  
  std::vector<double>::iterator it;
  it=std::max_element(tmp.begin(),tmp.end());
  largest_side=*it;  
  largest_side += 0.02;
}

template <typename PointInT>
void
GOODEstimation<PointInT>::computeDistanceBetweenProjections (std::vector <std::vector <float> > projection1, std::vector <std::vector <float> > projection2, float &distance)
{
  float sum  = 0 ;
  for (size_t i =0; i < projection1.size(); i++)
  { for (size_t j =0; j < projection1.size(); j++)
    {
      float d = projection1.at(i).at(j) - projection2.at(i).at(j);
      sum += pow(d,2);
    }
  } 
  distance = sqrt(sum);  
}

template <typename PointInT>
inline std::vector < std::vector<unsigned int> > GOODEstimation<PointInT>::initializing2DHistogram (unsigned int number_of_bins)
{
  std::vector<unsigned int> row( number_of_bins, 0);
  std::vector < std::vector<unsigned int> > histogram2D(number_of_bins, row);
  return histogram2D;
}

template <typename PointInT>
void
GOODEstimation<PointInT>::compute (std::vector< float > &object_description )	
{
  double largest_side = 0;
  int  sign = 1;
  std::vector <float> view_point_entropy;
  std::string name_of_sorted_projected_plane;
  boost::shared_ptr<pcl::PointCloud<PointInT> > initial_cloud_projection_along_x_axis (new pcl::PointCloud<PointInT>);//Declare a boost share ptr to the pointCloud
  boost::shared_ptr<pcl::PointCloud<PointInT> > initial_cloud_projection_along_y_axis(new pcl::PointCloud<PointInT>);//Declare a boost share ptr to the pointCloud
  boost::shared_ptr<pcl::PointCloud<PointInT> > initial_cloud_projection_along_z_axis (new pcl::PointCloud<PointInT>);//Declare a boost share ptr to the pointCloud
  bool visualize = false;
  pcl::PointXYZ pt; 
  
  /* __________________________
  |                            |
  | construct ORF based on PCA |
  |____________________________| */

  ///////// the theory of new shape descriptor ///////////////////////////
  // //NOTE  the PCA base reference frame construction basically does:
  // 1) compute the centroid (c0, c1, c2) and the normalized covariance
  // 2) compute the eigenvectors e0, e1, e2. The reference system will be (e0, e1, e0 X e1) --- note: e0 X e1 = +/- e2
  // 3) move the points in that RF --- note: the transformation given by the rotation matrix (e0, e1, e0 X e1) & (c0, c1, c2) must be inverted
  // 4) compute the max, the min and the center of the diagonal (mean_diag)
  // 5) given a box centered at the origin with size (max_pt.x - min_pt.x, max_pt.y - min_pt.y, max_pt.z - min_pt.z) 
  //    the transformation you have to apply is Rotation = (e0, e1, e0 X e1) & Translation = Rotation * mean_diag + (c0, c1, c2)
  
  // compute principal directions	  
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*input_, centroid);
  Eigen::Matrix3f covariance;
  computeCovarianceMatrixNormalized(*input_, centroid, covariance);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
  Eigen::Matrix3f eigen_vectors = eigen_solver.eigenvectors();
  eigen_vectors.col(2) = eigen_vectors.col(0).cross(eigen_vectors.col(1));
  //std::cout << "eigen vectores :  \n " << eigen_vectors << std::endl;
  
  Eigen::Vector3f eigen_values =  eigen_solver.eigenvalues();
  //std::cout << "The eigenvalues of the covariance matrix before sorting are \n:" << eigen_values << endl;
  
  //sorting eigen vectors based on eigen values
  eigen_vectors.col(0)= eigen_vectors.col(2);
  eigen_vectors.col(2) = eigen_vectors.col(0).cross(eigen_vectors.col(1));
  //std::cout << "eigen vectores cross product :  \n " << eigen_vectors << std::endl;

  // move the points to the PCA based reference frame
  Eigen::Matrix4f p2w(Eigen::Matrix4f::Identity());
  p2w.block<3,3>(0,0) = eigen_vectors.transpose();
  p2w.block<3,1>(0,3) = -1.f * (p2w.block<3,3>(0,0) * centroid.head<3>());
  transformation_ = p2w;
  
  boost::shared_ptr<pcl::PointCloud<PointInT> > transformed_point_cloud (new pcl::PointCloud<PointInT>);
  pcl::transformPointCloud(*input_, *transformed_point_cloud, p2w);
  transformed_point_cloud_ = transformed_point_cloud;
  //compute the max, the min and the center of the diagonal (mean_diag)
  PointInT min_pt, max_pt;
  pcl::getMinMax3D(*transformed_point_cloud_, min_pt, max_pt);
  const Eigen::Vector3f mean_diag = 0.5f*(max_pt.getVector3fMap() + min_pt.getVector3fMap());
  // centroid transform
  Eigen::Quaternionf qfinal(eigen_vectors);//rotation matrix
  Eigen::Vector3f center_of_bbox = eigen_vectors*mean_diag + centroid.head<3>(); // Translation = Rotation * center_diag + (c0, c1, c2)
  center_of_bbox_.x = center_of_bbox(0,0);
  center_of_bbox_.y = center_of_bbox(1,0);
  center_of_bbox_.z = center_of_bbox(2,0);
/* _________________________________
  |                         	   |
  | construct three projection view |
  |_________________________________| */
		  
  // ax+by+cz+d=0, where b=c=d=0, and a=1, or said differently, the YoZ plane.
  pcl::ModelCoefficients::Ptr coefficients_x (new pcl::ModelCoefficients ());
  coefficients_x->values.resize (4);
  coefficients_x->values[0] = 1.0; coefficients_x->values[1] = 0; coefficients_x->values[2] = 0; coefficients_x->values[3] = 0;
  projectPointCloudToPlane(transformed_point_cloud_, coefficients_x, initial_cloud_projection_along_x_axis);
  for (size_t i=0; i< initial_cloud_projection_along_x_axis->points.size(); i++)
  {
      initial_cloud_projection_along_x_axis->points.at(i).x = 0.25;
  }
  vector_of_projected_views_.push_back(initial_cloud_projection_along_x_axis);
 
  //ax+by+cz+d=0, where a=c=d=0, and b=1, or said differently, the XoZ plane.
  pcl::ModelCoefficients::Ptr coefficients_y (new pcl::ModelCoefficients ());
  coefficients_y->values.resize (4);
  coefficients_y->values[0] = 0.0; coefficients_y->values[1] = 1.0; coefficients_y->values[2] = 0; coefficients_y->values[3] = 0;

  projectPointCloudToPlane(transformed_point_cloud_, coefficients_y, initial_cloud_projection_along_y_axis);
  for (size_t i=0; i< initial_cloud_projection_along_y_axis->points.size(); i++)
  {
      initial_cloud_projection_along_y_axis->points.at(i).y = 0.3;
  }
  vector_of_projected_views_.push_back(initial_cloud_projection_along_y_axis);

	    
  // ax+by+cz+d=0, where a=b=d=0, and c=1, or said differently, the XoY plane.
  pcl::ModelCoefficients::Ptr coefficients_z (new pcl::ModelCoefficients ());
  coefficients_z->values.resize (4); 
  coefficients_z->values[0] = 0; coefficients_z->values[1] = 0; coefficients_z->values[2] = 1.0;   coefficients_z->values[3] = 0;
  projectPointCloudToPlane(transformed_point_cloud_, coefficients_z, initial_cloud_projection_along_z_axis);
  for (size_t i=0; i< initial_cloud_projection_along_z_axis->points.size(); i++)
  {
      initial_cloud_projection_along_z_axis->points.at(i).z = 0.3;
  }		
  vector_of_projected_views_.push_back(initial_cloud_projection_along_z_axis);

  /* _________________________
  |                           |
  |  Axes sign disambiguation |
  |___________________________| */
		  
  computeBoundingBoxDimensions(transformed_point_cloud_, bbox_dimensions_);	
  computeLargestSideOfBoundingBox(bbox_dimensions_ ,largest_side);		

  /* _________________________
  |                           |
  |  Axes sign disambiguation |
  |___________________________| */

  int Sx=1, Sy=1;
  signDisambiguationXAxis(initial_cloud_projection_along_y_axis, threshold_, Sx );//XoZ Plane
  signDisambiguationYAxis(initial_cloud_projection_along_x_axis, threshold_, Sy );//YoZ Plane
  sign = Sx * Sy;
  sign_=sign;
  /* _______________________________________________________
  |                       				  |
  |  compute histograms of projection of the given object   |
  |_________________________________________________________| */
    
  std::vector <unsigned int> complete_object_histogram;
  std::vector <unsigned int> complete_object_histogram_normalized;//each projection view is normalized sepreatly
  std::vector < std::vector<unsigned int> > XOY_histogram = initializing2DHistogram(number_of_bins_);		
  std::vector < std::vector<unsigned int> > XOZ_histogram = initializing2DHistogram(number_of_bins_);      
  std::vector < std::vector<unsigned int> > YOZ_histogram = initializing2DHistogram(number_of_bins_);  
  
  std::vector <std::vector <float> > normalized_projected_views;
  
  //projection along X axis
  create2DHistogramFromYOZProjection( initial_cloud_projection_along_x_axis, largest_side, number_of_bins_, sign,YOZ_histogram);

  std::vector <unsigned int> histogramYOZ1D;
  convert2DHistogramTo1DHistogram(YOZ_histogram, histogramYOZ1D);
  complete_object_histogram.insert(complete_object_histogram.end(), histogramYOZ1D.begin(), histogramYOZ1D.end());
  std::vector <float> normalized_histogramYoZ;
  normalizingHistogram( histogramYOZ1D, normalized_histogramYoZ);
  normalized_projected_views.push_back(normalized_histogramYoZ);
  
  complete_object_histogram_normalized.insert(complete_object_histogram_normalized.end(), normalized_histogramYoZ.begin(), normalized_histogramYoZ.end());
  float YoZ_entropy = 0;
  viewpointEntropy(normalized_histogramYoZ, YoZ_entropy);
  //viewpointEntropyNotNormalized(histogramYOZ1D, YoZ_entropy);
  view_point_entropy.push_back(YoZ_entropy);

  //projection along Y axis
  create2DHistogramFromXOZProjection( initial_cloud_projection_along_y_axis,largest_side, number_of_bins_, sign,XOZ_histogram);

  std::vector <unsigned int> histogramXOZ1D;
  convert2DHistogramTo1DHistogram(XOZ_histogram, histogramXOZ1D);
  complete_object_histogram.insert(complete_object_histogram.end(), histogramXOZ1D.begin(), histogramXOZ1D.end());

  std::vector <float> normalized_histogramXoZ;
  normalizingHistogram( histogramXOZ1D, normalized_histogramXoZ);
  normalized_projected_views.push_back(normalized_histogramXoZ);

  complete_object_histogram_normalized.insert(complete_object_histogram_normalized.end(), normalized_histogramXoZ.begin(), normalized_histogramXoZ.end());
  float XoZ_entropy = 0;
  viewpointEntropy(normalized_histogramXoZ, XoZ_entropy);
  view_point_entropy.push_back(XoZ_entropy);

  //projection along Z axis	
  create2DHistogramFromXOYProjection( initial_cloud_projection_along_z_axis, largest_side, number_of_bins_, sign, XOY_histogram);

  std::vector <unsigned int> histogramXOY1D;
  convert2DHistogramTo1DHistogram(XOY_histogram, histogramXOY1D);
  complete_object_histogram.insert(complete_object_histogram.end(), histogramXOY1D.begin(), histogramXOY1D.end());
  
  std::vector <float> normalized_histogramXoY;
  normalizingHistogram( histogramXOY1D, normalized_histogramXoY);
  normalized_projected_views.push_back(normalized_histogramXoY);

  complete_object_histogram_normalized.insert(complete_object_histogram_normalized.end(), normalized_histogramXoY.begin(), normalized_histogramXoY.end());
  float XoY_entropy = 0;
  viewpointEntropy(normalized_histogramXoY, XoY_entropy);
  view_point_entropy.push_back(XoY_entropy);
 
  std::vector <float> normalized_histogram;
  normalizingHistogram( complete_object_histogram, normalized_histogram);
  //printHistogram ( normalized_histogram, "normalized_complete_object_histogram");
   
  int maximum_entropy_index = 0;
  findMaxViewPointEntropy(view_point_entropy, maximum_entropy_index);

  objectViewHistogram( maximum_entropy_index, normalized_projected_views, object_description, name_of_sorted_projected_plane);
  order_of_projected_plane_ =  name_of_sorted_projected_plane;
  
}
