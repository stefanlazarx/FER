//Proiect destinat recunoasterii expresiei faciale - Stefan Lazar & Bianca Popescu
//Etape: - Detectia fetei dintr-o imagine din cadrul bazei de date
//       - Scoaterea in evidenta a zonelor de interes si a punctelor de interes din cadrul acestor zone
//       - Clasificarea intr-una din cele 7 emotii, in functie de pozitionarea punctelor de interes fata de un anumit 'standard' de normalitate
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::face;

void main()
{
	string path = "Tools/JAFFEDBASE/YM.NE1.49.tiff";
	Mat img = imread(path);

	CascadeClassifier face;
	face.load("Tools/haarcascade_frontalface.xml");

	Ptr<Facemark> facemark = FacemarkLBF::create();
	facemark->loadModel("Tools/lbfmodel.yaml.txt");

	vector<Rect> faces;
	face.detectMultiScale(img, faces, 1.1, 10);
	Mat imgFace;

	for (int i = 0; i < faces.size(); i++)
	{
		rectangle(img, faces[i].tl(), faces[i].br(), Scalar(0, 128, 255), 3); //drawing the rectangle surrounding the face
		circle(img, Point((faces[i].tl() + faces[i].br()) * 0.5), 3, Scalar(255, 255, 0), FILLED, LINE_8); //drawing the center of the face 
		imgFace = img(faces[i]);
		resize(imgFace, imgFace, Size(imgFace.cols * 2, imgFace.rows * 2));
	}
	cout << faces.size();
	imshow("Subject", img);
	imshow("Subject's resizes face", imgFace);
	waitKey(0);
}