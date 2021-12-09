//Proiect destinat recunoasterii expresiei faciale - Stefan Lazar & Bianca Popescu
//Etape: - Detectia fetei dintr-o imagine din cadrul bazei de date
//       - Scoaterea in evidenta a zonelor de interes si a punctelor de interes din cadrul acestor zone
//       - Clasificarea intr-una din cele 7 emotii, in functie de pozitionarea punctelor de interes fata de un anumit 'standard' de normalitate

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include<iostream>

using namespace cv;
using namespace std;

void main()
{
	string path = "Resources/KA.AN1.39.tiff";
	Mat img = imread(path);

	CascadeClassifier face;
	face.load("Resources/haarcascade_frontalface.xml");

	vector<Rect> faces;
	face.detectMultiScale(img, faces, 1.1, 10);

	for (int i = 0; i < faces.size(); i++)
	{
		rectangle(img, faces[i].tl(), faces[i].br(), Scalar(0, 255, 0), 2);
	}
	cout << faces.size();
	imshow("Subject", img);
	waitKey(0);
}