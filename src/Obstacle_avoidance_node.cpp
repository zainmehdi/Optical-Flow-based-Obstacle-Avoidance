//
// Created by zain on 18. 4. 3.
//


/*
 *
 * This node runs in parellel to main node and helps in
 * avoiding obstacles. Used optical flow balancing strategy to figure out which way to go.
 * Can also be used as a stand alone node to avoid obstacle for a real or simulated robot.
 * For simulated robot however some changes need to be made in  the part where velocities
 * are sent.
 *
 * http://journals.sagepub.com/doi/pdf/10.5772/5715
 *
*/

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "ctime"
#include "chrono"
#include "ros/ros.h"
#include <eigen3/Eigen/Dense>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <ctype.h>
#include "ros/ros.h"
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"
#include <geometry_msgs/Twist.h>


#include <iostream>
#include <ctype.h>


using namespace cv;
using namespace std;
using namespace Eigen;
namespace enc = sensor_msgs::image_encodings;



Point2f point;
bool addRemovePt = false;
Mat img,src;
Mat gray, prevGray, image, frame;
vector<Point2f> points[2],v;
TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
Size subPixWinSize(10,10), winSize(31,31);
float left_sum,right_sum,up_sum,down_sum;
vector<uchar> status;
geometry_msgs::Twist velocity;
ros::Publisher vel_pub;


const int MAX_COUNT = 100;
bool needToInit = false;
bool nightMode = false;

static void onMouse( int event, int x, int y, int /*flags*/, void* /*param*/ )
{
    if( event == EVENT_LBUTTONDOWN )
    {
        point = Point2f((float)x, (float)y);
        addRemovePt = true;
    }
}


// Function to calculate Focus of Expansion.
// Helpful resources
// https://www.dgp.toronto.edu/~donovan/stabilization/opticalflow.pdf
// https://stackoverflow.com/questions/18245076/optical-flow-and-focus-of-expansion-in-opencv?rq=1

void calculate_FOE(vector<Point2f> prev_pts, vector<Point2f> next_pts)
{

    MatrixXf A(next_pts.size(),2);
    MatrixXf b(next_pts.size(),1);
    Point2f tmp;

    for(int i=0;i<next_pts.size();i++)
    {

        if(!status[i])
            continue;
        tmp= next_pts[i]-prev_pts[i];
        A.row(i)<<next_pts[i].x-prev_pts[i].x,next_pts[i].y-prev_pts[i].y;
        b.row(i)<<(prev_pts[i].x*tmp.x)-(prev_pts[i].y*tmp.y);

    }


    Matrix<float,2,1> FOE;
    FOE=((A.transpose()*A).inverse())*A.transpose()*b;

    Point2f c;
    c.x=FOE(0,0);
    c.y=FOE(1,0);
    circle( image, c, 8, Scalar(255,255,0), -1, 8);


}

// Function to draw flow vector and calculate magnitude

void Draw_flowVectors(vector<Point2f> prev_pts, vector<Point2f> next_pts)
{

    right_sum=0;
    left_sum=0;
    up_sum=0;
    down_sum=0;

    for(int i=0;i<next_pts.size();i++)
    {
        CvPoint p,q;

        if(!status[i])
            continue;

        p.x = (int) prev_pts[i].x;
        p.y = (int) prev_pts[i].y;
        q.x = (int) next_pts[i].x;
        q.y = (int) next_pts[i].y;
        double angle;
        angle = atan2( (double) p.y - q.y, (double) p.x - q.x );
        double hypotenuse;  hypotenuse = sqrt( pow(p.y - q.y,2) + pow(p.x - q.x,2 ));

        // Scaling of arrow vectors
        q.x = (int) (p.x - 3 * hypotenuse * cos(angle));
        q.y = (int) (p.y - 3 * hypotenuse * sin(angle));
        arrowedLine( image, p, q, Scalar(255,255,255), 1, CV_AA, 0 );



        // Calculating magnitude

        int mag= sqrt(pow(next_pts[i].x-prev_pts[i].x,2)+pow(next_pts[i].y-prev_pts[i].y,2));

       // Flow vectors are categorized according to there position in image
       // We can choose horizontal and vertical boundary as per our choice

        if(next_pts[i].x > 320)
            right_sum+=mag;
        else
            left_sum+=mag;

        if(next_pts[i].y <329)
            up_sum+=mag;
        else
            down_sum+=mag;



    }

    cout<<"Left Flow "<<left_sum<<" Right Flow "<<right_sum
        <<" Up Flow "<<up_sum<<" Down Flow "<<down_sum<<endl;
    cout<<".........................................."<<endl;


}






void imageCb(const sensor_msgs::ImageConstPtr& msg)
{

    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, enc::BGR8);
    }
    catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    frame=cv_ptr->image;



    if( frame.empty() )
        return;

    frame.copyTo(image);
    cvtColor(image, gray, COLOR_BGR2GRAY);

    Rect ROI=Rect(200,1,250,300);
    Mat mask;
    Mat mask_image;
    mask = Mat::zeros(gray.size(), CV_8U);
    mask_image=Mat(mask,ROI);
    mask_image=Scalar(255,255,255);
    if( nightMode )
        image = Scalar::all(0);

    if( needToInit )
    {
        // automatic initialization
        goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, mask, 3, 3, 0, 0.04);
        cornerSubPix(gray, points[1], subPixWinSize, Size(-1,-1), termcrit);
        addRemovePt = false;
    }
    else if( !points[0].empty() )
    {

        vector<float> err;
        if(prevGray.empty())
            gray.copyTo(prevGray);
        calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                             3, termcrit, 0, 0.001);


        if(!points[0].empty() && !points[1].empty())
        {
            Draw_flowVectors(points[0],points[1]);
            // calculate_FOE(points[0],points[1]);       // wasnt working properly so omitted it
        }

        size_t i, k;
        for( i = k = 0; i < points[1].size(); i++ )
        {
            if( addRemovePt )
            {
                if( norm(point - points[1][i]) <= 10 )
                {
                    addRemovePt = false;
                    continue;
                }
            }

            if( !status[i] )
                continue;

            points[1][k++] = points[1][i];
            circle( image, points[1][i], 3, Scalar(0,255,0), -1, 8);
        }
        points[1].resize(k);
    }

    if( addRemovePt && points[1].size() < (size_t)MAX_COUNT )
    {
        vector<Point2f> tmp;
        tmp.push_back(point);
        cornerSubPix( gray, tmp, winSize, Size(-1,-1), termcrit);
        points[1].push_back(tmp[0]);
        addRemovePt = false;
    }

    needToInit = false;
    imshow("Obstacle_Avoidance", image);

    char c = (char)waitKey(10);
    if( c == 27 )
        return;
    switch( c )
    {
        case 'r':
            needToInit = true;
            break;
        case 'c':
            points[0].clear();
            points[1].clear();
            break;
        case 'n':
            nightMode = !nightMode;
            break;
    }



    std::swap(points[1], points[0]);
    cv::swap(prevGray, gray);


    // Here we decide when and in which direction to avoid obstacle
    // I use a specific magnitude threshold which when exceeded
    // triggers velocity commands to be published to Unity or actual robot

    if(right_sum>left_sum && (abs(right_sum-left_sum)>15))
    {

        velocity.angular.z=25;
        vel_pub.publish(velocity);
    }

   else if(right_sum<left_sum && (abs(right_sum-left_sum)>15))
    {

        velocity.angular.z=-25;
        vel_pub.publish(velocity);
    }	
    else
    {
        velocity.linear.y=0;
        velocity.angular.z=0;
        vel_pub.publish(velocity);
        
    }




}

//void imageCb(const sensor_msgs::ImageConstPtr& msg);
//
int main( int argc, char** argv ) {

    ros::init(argc, argv, "Obstacle_Avoidance");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber image_sub;
    image_transport::Publisher image_pub;
    image_sub = it.subscribe("image_raw", 1, imageCb);
    vel_pub=nh.advertise<geometry_msgs::Twist>("cmd_vel",1000);

    ros::spin();

    return 0;
}
