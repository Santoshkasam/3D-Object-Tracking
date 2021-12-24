
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);


    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

double median(std::vector<double> &v){
    
    /*
        Input: Vector of double elements, which are distance ratios 

        Output: Median of the elements in the input vector
    */
    
    size_t size = v.size();
    
    if( !(size%2) ) // If size is even
    {
        int medianElem1 = (size-2)/2;
        int medianElem2 = size/2;

        std::nth_element(v.begin(), v.begin() + medianElem1, v.end());
        std::nth_element(v.begin(), v.begin() + medianElem2, v.end());

        return ((v[medianElem1] + v[medianElem2])/2);
 
    } else  // If size is odd
    {
        int medianElem = (size-1)/2 ;
        std::nth_element(v.begin(), v.begin() + medianElem, v.end()); // the nth_element function partially sorts the vector to obtain the nth value in ascending order

        return v[medianElem];
    }
 
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    /* Funtion inputs
        kptMatches: It contains all the keypoint matches between current frame
                 and previous frame.
        boundingBox: It is a boundingBox in the current frame.
    */

    /* Algorithm: > For every match, take the current keypoint and check if it is within the boundingBox. 
                > If yes, calculate the euclidean distance between the keypoints of the match.
                > Insert the match and the corresponding distance in a multimap
                > Sum all the distances
                > Calculate the mean of all the distances
                > Calculate the Standard deviation (sd)
                > Identify the matches whose distance falls within the range of mean-(1*SD) to mean+(1*SD) of the distribution
                  and add them to the kptMatches of the boundingBox.
    */

    /* Note:
       The above filteration is performed in order to remove the outliers.
       We consider that the keypoints move rigidly over the frames. Hence, when keypoints of two unrelated 
       regions, or keypoints that are far from each other ,in reality, are matched, this process helps elimenate
       them. 
    */

    std::vector<cv::DMatch> allBBmatches;
    double totalDist = 0;
    multimap<cv::DMatch, double> matchDistPairs;

    // Identify the keypoints within the roi of the boundingBox
    for(cv::DMatch match : kptMatches){
        
        cv::KeyPoint CurrKpt = kptsCurr[match.trainIdx];
        cv::KeyPoint PrevKpt = kptsPrev[match.queryIdx];
         
        if(boundingBox.roi.contains(CurrKpt.pt)){
        
        double distBtwKpts = cv::norm(CurrKpt.pt - PrevKpt.pt);
        
        matchDistPairs.insert(pair<cv::DMatch, double> (match, distBtwKpts));
        totalDist += distBtwKpts;   
        }
    } 

    // Calculate mean and Standard deviation
    double mean = totalDist/allBBmatches.size();
    double var = 0;
    for(auto matchDistpair : matchDistPairs){
        var += pow((matchDistpair.second - mean), 2);
    }
    var /= matchDistPairs.size();
    double stdDeviation = sqrt(var);
    
    // Add the inliers into the kptMatches vector of boundingBox
    for(auto matchDistpair : matchDistPairs){
        
        if( (mean - stdDeviation) < matchDistpair.second < (mean + stdDeviation)){ 
        
            boundingBox.kptMatches.push_back(matchDistpair.first);

        }
    }
    
}



// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    std::vector<double> distRatios;
    
    // Outer keypoint matches loop
    for(auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1){
        cv::KeyPoint kptOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kptOuterPrev = kptsPrev.at(it1->queryIdx);

        // Inner keypoint matches loop
        for(auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2 ){
            double minDist = 100.0;
            cv::KeyPoint kptInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kptInnerPrev = kptsPrev.at(it2->queryIdx);

            double distCurr = cv::norm(kptOuterCurr.pt - kptInnerCurr.pt);
            double distPrev = cv::norm(kptOuterPrev.pt - kptInnerPrev.pt);

            if(distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist){
                double distRatio = (distCurr/distPrev);
                distRatios.push_back(distRatio);
            }
        }
    }

    if(distRatios.size() == 0){
        TTC = NAN;
        return; 
    }

    double medianDistRatio = median(distRatios);
    double dT = 1.0/frameRate;

    TTC = -dT/(1-medianDistRatio);
  
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    
    // Step 1: Arrange all the lidarPointsPrev in ascending order of x distance and take the 30th percentile x value.
    // Note: Least percentile x value is the closest one.
    vector<double> allPrevXcoordinates;
    double percentile = 0.30;
    for(auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); it++){
        allPrevXcoordinates.push_back(it->x);
    }

    sort(allPrevXcoordinates.begin(), allPrevXcoordinates.end());

    int xPrevBestIndex = 0;
    if(allPrevXcoordinates.size() > 1){
        xPrevBestIndex = (int)((allPrevXcoordinates.size() * percentile) - 1);
        cout << "prev index = " << xPrevBestIndex << endl;
    } else {
        xPrevBestIndex = 0;
    }
    
    // Step 2: Arrange all the lidarPointsCurr in ascending order of x distance and take the 30th percentile x value
    // Note: The percentile index is calculated assuming that every distance value is unique.
    vector<double> allCurrXcoordinates;
    
    for(auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); it++){
        allCurrXcoordinates.push_back(it->x);
    }

    sort(allCurrXcoordinates.begin(), allCurrXcoordinates.end());

    int xCurrBestIndex;
    if(allCurrXcoordinates.size() > 1){
        xCurrBestIndex = (int)((allCurrXcoordinates.size() * percentile) - 1);
        cout << "curr index = " << xCurrBestIndex << endl;
    } else {
        xCurrBestIndex = 0;
    }

    double minXCurr = allCurrXcoordinates[xCurrBestIndex];
    double minXPrev = allPrevXcoordinates[xPrevBestIndex];

    // Step 3: calculate the TTC based on these two x values;
    TTC = (minXCurr / (minXPrev-minXCurr)) * (1/frameRate);
    cout << "TTC Lidar: " << TTC << "X distance: "<< minXCurr <<endl;
    
    
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{   
    // Step 1: Initialize the counter
    // A two dimensional vector is initiated, with zeros, where the indices of dim1 = IDs of prev bounding 
    // boxes and indices of dim2 = IDs of curr bounding boxes. This counter is populated with the number of 
    // matches between each pair of prev and curr bounding box.   
    vector<vector<int>> bbMatchCounts(prevFrame.boundingBoxes.size(), vector<int>(currFrame.boundingBoxes.size(),0));
    
    // Step 2: Count the number of occurrences of each combination of bounding boxes over all the matches.
    for(auto it1 = matches.begin(); it1 != matches.end(); ++it1 ){

        // Extract the current and previous keypoints
        //  info: matches is a vecotor of datatype "Dmatches". 
        //        Dmatches contains trainIdx (Id of current keypoint), queryIdx (Id of previous keypoint) 
        cv::KeyPoint currKeypoint = currFrame.keypoints.at(it1->trainIdx);
        cv::KeyPoint prevKeypoint = prevFrame.keypoints.at(it1->queryIdx);

        // Find the bounding boxes of the corresponding keypoints and populate the counter
        for(auto it2 = prevFrame.boundingBoxes.begin(); it2 != prevFrame.boundingBoxes.end(); it2++){
            
            if (it2->roi.contains(prevKeypoint.pt)){

                for(auto it3 = currFrame.boundingBoxes.begin(); it3 != currFrame.boundingBoxes.end(); it3++){
                    
                    if(it3->roi.contains(currKeypoint.pt)){

                        bbMatchCounts[it2->boxID][it3->boxID] += 1;
                    } 
                }
            }
        }
    }
  
    // Step 3: For every bounding box in the previous frame, find a bounding box in the current frame that
    //         has highest number of matches  
    for(int i = 0; i < bbMatchCounts.size(); i++){

        // Find the best match
        int bestCurrBBid = max_element(bbMatchCounts[i].begin(), bbMatchCounts[i].end()) - bbMatchCounts[i].begin();
        
        // Insert the pair box IDs of the match pair (prevBoxID, currBoxID)
        bbBestMatches.insert(make_pair(i, bestCurrBBid));

    } 
    
    // Optionl: print the BB combinations 
    for(auto it = bbBestMatches.begin(); it != bbBestMatches.end(); ++it){
        cout << "Previous frame BB: "<< it->first << " matches with Current frame BB: " << it->second << endl;
    }
}
