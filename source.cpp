//~ Proiect destinat recunoasterii expresiei faciale - Stefan Lazar & Bianca Popescu
//~ Etape: 
// - Detectia fetei dintr-o imagine din cadrul bazei de date    
// - Scoaterea in evidenta a zonelor de interes si a punctelor de interes din cadrul acestor zone
// - Clasificarea intr-una din cele 7 emotii, in functie de pozitionarea punctelor de interes fata de un anumit 'standard' de normalitate
#include <opencv2/highgui.hpp> // -> Pentru afisarea imaginilor
#include <opencv2/imgcodecs.hpp> // -> Pentru citirea imaginilor
#include <opencv2/imgproc.hpp> // -> Pentru redimensionarea imaginilor (in cazul nostru, a fetei detectate)
#include <opencv2/objdetect.hpp> // -> Pentru folosirea clasificatorului HAAR in vederea detectiei fetei
#include <opencv2/face.hpp> // -> *Pentru detectia punctelor din ROI-uri (Regions of interest) din cadrul fetei detectate
#include <iostream>

/* Definirea namespace-urilor */
using namespace cv;
using namespace std;
using namespace cv::face;

void main()
{
	string path = "Tools/JAFFEDBASE/MK.NE3.115.tiff"; // -> Initializarea string-ului ce reprezinta path-ul relativ la imaginea de input
	/* -> Toate imaginile din JAFFED sunt in format TIFF
	   -> Primele doua litere sunt initialele candidatei, urmatoarele doua sunt abrevierea emotiei (NE = Neutral), urmat de
	   un numar de ordine al pozei in cadrul bazei de date */
	Mat img = imread(path); // Citirea imaginii intr-un obiect de tip Mat numit "img"
	CascadeClassifier face; /* Declaram un obiect de tip CascadeClassifier, pe care il vom folosi in vederea detectiei fetei/fetelor
						din imaginea sursa */
	face.load("Tools/haarcascade_frontalface.xml"); //Incarcam clasificatorul HAAR din fisierul xml

	vector<Rect> faces; /* Declaram un vector de dreptunghiuri, vector ce va fi populat cu dreptunghiurile
						in care vor fi incadrate fetele detectate de clasificator */
	face.detectMultiScale(img, faces, 1.1, 5); /* Aceasta functie detecteaza obiecte de diferite dimensiuni din imaginea de input (img)
												si le stocheaza sub forma unor dreptunghiuri, in vectorul de dreptunghiuri (faces)
												-> Al 3-lea parametru face referire la factorul de scalare - cat de mult este micsorata
												imaginea la fiecare scalare a imaginii pentru a facilita detectia fetei
												ex: un factor de scalare a imaginii mai apropiat de 1 inseamna o posibila detectie mai
												buna, dar mai inceata, pe cand un factor mai apropiat de 1.5 inseamna o posibilitate de
												ratare a unor fete din imagine, in favoarea vitezei
												-> Al 4-lea parametru reprezinta cati vecini trebuie fiecare dreptunghi stabilit
												ca si canditat sa aiba pentru a fi luat in considerare - afecteaza calitatea fetelor
												detectate
												ex: un numar mai mare de vecini necesari rezulta in mai putine detectii, dar cu
												calitati superioare (arbitrar, il alegem intre 3 si 6) */

	Mat imgFace; // In acest obiect de tip Mat vom stoca imaginea fetei detectate, pe care o vom si mari ulterior

	for (int i = 0; i < faces.size(); i++) /*Parcurgem vectorul de dreptunghiuri (de fete detectate, in cazul nostru)
										   Se va realiza o singura iteratie, deoarece dimensiunea vectorului "faces" este 1. In fiecare
										   imagine vom avea o singura fata, nu mai multe */
	{
		//rectangle(img, faces[i].tl(), faces[i].br(), Scalar(0, 128, 255), 3); /*Desenam in imaginea originala dreptunghiul corespunzator fetei
																			  //oferind ca parametri coordonatele coltului din stanga sus,
																			  //din dreapta jos, culoarea (in format BGR) si grosimea liniilor */
		//circle(img, Point((faces[i].tl() + faces[i].br()) * 0.5), 2, Scalar(255, 255, 0), FILLED, LINE_8); /* Desenam un punct in mijlocul fetei,
																										   //pentru utilizari viitoare */
		imgFace = img(faces[i]); // In obiectul imgFace, punem fata detectata
		resize(imgFace, imgFace, Size(imgFace.cols * 2, imgFace.rows * 2));
		cout << "\n\n\n" << img.rows << " " << img.cols;//Pe care o facem de 2 ori mai mare pe inaltime si latime
	}

	//Ptr<Facemark> facemark = FacemarkLBF::create();
	/*Aici am incercat sa folosesc facemark-ul pentru detectia ROI-urilor, insa am intampinat dificultati in a integra header-ele necesare,
	inca nu reusesc sa imi dau seama ce imi produce eroare de linkeditare, deoarece sunt convins ca toate sunt la locul lor, dar o sa il fac sa
	mearga pana la urma. Nu stiu sigur daca voi ramane la algoritmul acesta pentru detectia ROI-urilor pana la urma, dar am vrut sa il fac
	cu un clasificator pentru a incepe apoi algoritmul la care m-am gandit pentru clasificarea emotiilor, anume folosind mai multe triunghiuri
	si arii de triunghiuri create intre diferite puncte din cadrul mai multor regiuni de interes */
	//facemark->loadModel("Tools/lbfmodel.yaml.txt");


	Mat croppedFace = imgFace(Range(75, 285), Range(50, 270));
	Mat blurredCropped, blurredWhole;

	GaussianBlur(img, blurredWhole, Size(3, 3), 0, 0);
	GaussianBlur(croppedFace, blurredCropped, Size(3, 3), 0, 0);

	Mat faceEdges;
	Mat eyebrows, eyes, nose, lips;

	Canny(blurredWhole, faceEdges, 120, 200, 3, false);
	//eyebrows = croppedFace(Range(0, 30), Range(0, 220));
	eyebrows = faceEdges(Range(95, 120), Range(70, 190)); //eyebrows -> 25 x 120 (95 <-> 120) x (70 <-> 190)

	//eyes = croppedFace(Range(35, 65), Range(0, 220));
	GaussianBlur(blurredWhole, blurredWhole, Size(5, 5), 0, 0);
	Canny(blurredWhole, faceEdges, 110, 230, 3, false);
	eyes = faceEdges(Range(120, 140), Range(70, 190));


	//nose = croppedFace(Range(70, 140), Range(50, 170));
	Mat blurfornose;
	GaussianBlur(img, blurfornose, Size(3, 3), 0, 0);
	Canny(blurfornose, faceEdges, 125, 225, 3, false);
	nose = faceEdges(Range(135, 175), Range(80, 180));

	//lips = croppedFace(Range(165, 200), Range(0, 220));
	Canny(blurredWhole, faceEdges, 100, 200, 3, false);
	lips = faceEdges(Range(180, 210), Range(75, 170));

	imshow("lips", lips);
	cout << "\n\n\n" << croppedFace.cols << " " << croppedFace.rows;
	//imshow("Subject", img); /* Afisari */
	//imshow("Cropped face", croppedFace);
	//imshow("Edges", faceEdges);
	//imshow("Eyebrows", eyebrows);
	waitKey(0);
}