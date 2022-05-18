//Proiect destinat recunoasterii expresiei faciale - Stefan Lazar & Bianca Popescu
//Etape: - Detectia fetei dintr-o imagine din cadrul bazei de date
//       - Scoaterea in evidenta a zonelor de interes si a punctelor de interes din cadrul acestor zone
//       - Clasificarea intr-una din cele 7 emotii, in functie de pozitionarea punctelor de interes fata de un anumit 'standard' de normalitate
//~ Proiect destinat recunoasterii expresiei faciale - Stefan Lazar & Bianca Popescu
//~ Etape: 
// - Detectia fetei dintr-o imagine din cadrul bazei de date    
// - Scoaterea in evidenta a zonelor de interes si a punctelor de interes din cadrul acestor zone
// - Clasificarea intr-una din cele 7 emotii, in functie de pozitionarea punctelor de interes fata de un anumit 'standard' de normalitate (expresia neutra)
#include <opencv2/highgui.hpp> // -> Pentru afisarea imaginilor
#include <opencv2/imgcodecs.hpp> // -> Pentru citirea imaginilor
#include <opencv2/imgproc.hpp> // -> Pentru redimensionarea imaginilor (in cazul nostru, a fetei detectate)
#include <opencv2/objdetect.hpp> // -> Pentru folosirea clasificatorului HAAR in vederea detectiei fetei
#include <opencv2/face.hpp> // -> *Pentru detectia punctelor din ROI-uri (Regions of interest) din cadrul fetei detectate
#include <iostream>
#include <math.h>


/* Definirea namespace-urilor */
using namespace cv;
using namespace std;
using namespace cv::face;

Ptr<Facemark> facemark; 
CascadeClassifier faceDetector; /* Declaram un obiect de tip CascadeClassifier, pe care il vom folosi in vederea detectiei fetei/fetelordin imaginea sursa */






float dist(float x1, float y1, float x2, float y2) /* Functie ce returneaza distanta intre doua puncte date ca parametri*/
{
	return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

void detectiePuncte(Mat img, vector<vector<Point2f>> puncte, float &lungimeGura, float &latimeGura, float &bazaNas) /* Functie ce primeste ca parametri o 
																													imagine intr-un obiect de tip Mat, un vector de vector de puncte si 3 marimi ce reprezinta
																													lungimea gurii, latimea/inaltimea gurii, si lungimea bazei nasului*/
{
	vector<Rect> faces; /* Declaram un vector de dreptunghiuri, vector ce va fi populat cu dreptunghiurile
						in care vor fi incadrate fetele detectate de clasificator */
	faceDetector.detectMultiScale(img, faces, 1.1, 5); /* Aceasta functie detecteaza obiecte de diferite dimensiuni din imaginea de input (img)
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

	

	Mat imgFace; /* In acest obiect de tip Mat vom stoca imaginea fetei detectate, pe care o vom si mari ulterior */

	if (faces.size() != 0) /* Daca s-a detectat o fata, parcurgem vectorul de fete (in cazul nostru, dimensiunea vectorului este 1, deoarece in imaginile noastre exista doar cate o persoana */
	{
		for (int i = 0; i < faces.size(); i++)
		{
			imgFace = img(faces[i]); /* Decupam dreptunghiul in care este incadrata fata din imagine (img) si il punem in obiectul Mat numit imgFace */
			resize(imgFace, imgFace, Size(imgFace.cols * 2, imgFace.rows * 2)); /* Marim imaginea fetei de 2 ori, pentru o vizualizare mai buna */
			faces[i] = Rect(faces[i].x = 0, faces[i].y = 0, faces[i].width * 2, faces[i].height * 2); /* Setam coordonatele originii dreptunghiului in care este incadrata fata
																									  de la cele relative imaginii originale la (0,0), avand ulterior aceeasi lungime si latime ca imaginea fetei
																									  (adica ii dublam marimile, cum am procedat cu imaginea fetei la linia anterioara) */
		}

		if (facemark->fit(imgFace, faces, puncte)) /* Functie de tip boolean ce returneaza 'true' in caz de succes si 'false' in caz de esec, se intra in if daca s-a putut popula "matricea" de puncte de interes din
												   cadrul regiunilor de interes.
												   -> 'imgFace' este imaginea in care se realizeaza cautarea punctelor, 'faces' reprezinta vectorul de fete detectate, iar 'puncte' este vectorul de vectori de variabile de tip Point2f,
													unde fiecare linie contine 68 de coloane/puncte de interes in cadrul fetei ce corespunde liniei respective. Daca ar fi existat mai multe fete in imagine,
													"matricea" puncte ar fi avut mai multe linii */
		{

			/* Din cele 68 de puncte ce au fost detectate de catre clasificator, ne alegem exact punctele de care avem nevoie pentru un raport dintre rata de succes a detectarii emotiei si dificultate cat mai bun */
			Point2f EL_L = puncte[0][18], EL_R = puncte[0][20], ER_L = puncte[0][23], ER_R = puncte[0][25], EL_I = puncte[0][21], ER_I = puncte[0][22],
				N_C = puncte[0][29], N_L = puncte[0][31], N_R = puncte[0][35], N_M = puncte[0][28], N_B = puncte[0][33],
				M_L = puncte[0][60], M_R = puncte[0][64], M_T = puncte[0][51], M_B = puncte[0][57],
				EYEL_TL = puncte[0][37], EYEL_BR = puncte[0][40], EYER_TL = puncte[0][43], EYER_BR = puncte[0][46];

			

			lungimeGura = dist(M_L.x, M_L.y, M_R.x, M_R.y), /* Se calculeaza distanta dintre coltul din stanga a gurii si cel din dreapta, pentru a afla "lungimea" gurii */
			latimeGura = dist(M_T.x, M_T.y, M_B.x, M_B.y); /* Se procedeaza asemanator pentru a afla "latimea/inaltimea" gurii, folosing punctele de sus si de jos ale gurii */

			bazaNas = dist(N_L.x, N_L.y, N_R.x, N_R.y); /* Pentru 'bazaNas' am folosit punctele din stanga si din dreapta de la baza nasului */

			/* lungimeNas = dist(N_M.x, N_M.y, N_B.x, N_B.y);
			   diagOchiL = dist(EYEL_TL.x, EYEL_TL.y, EYEL_BR.x, EYEL_BR.y);
			   diagOchiR = dist(EYER_TL.x, EYER_TL.y, EYER_BR.x, EYER_BR.y); */




			cout << "Segment lungime gura = " << lungimeGura << endl;
			cout << "Segment inaltime gura = " << latimeGura << endl;
			cout << "Baza nasului = " << bazaNas << endl;

			/* Parcurgem vectorul de fete si pentru fiecare fata, parcurgem vectorul de puncte de interes -> Daca unul din puncte este unul din interesul nostru personal, desenam un cerc pe imaginea fetei 
			la coordonatele punctului respectiv */
			for (int i = 0; i < faces.size(); i++)
			{
				for (int k = 0; k < puncte[i].size(); k++)
				{
					if (k == 18 || k == 20 || k == 21 || k == 22 || k == 23 || k == 25 || k == 28 || k == 29 || k == 31 || k == 33 || k == 35 || k == 37 || k == 40 || k == 43 || k == 46 || k == 51 || k == 57 || k == 60 || k == 64) {

						/*
						18 - leftside left eyebrow
						20 - rightside left eyebrow
						21 - interior left eyebrow
						22 - interior right eyebrow
						23 - leftside right eyebrow
						25 - rightside right eyebrow
						28 - mid-upper nose
						29 - center (of the nose)
						31 - bottom left nose
						33 - bottom center nose
						35 - bottom right nose
						37 - left eye top left
						40 - left eye bottom right
						43 - right eye top left
						46 - right eye bottom right
						51 - top mouth
						57 - bottom mouth
						60 - left mouth
						64 - right mouth
						*/
						cv::circle(imgFace, puncte[i][k], 2, cv::Scalar(0, 0, 255), FILLED); /*Desenam un cerc rosu umplut de grosime 2, in imaginea fetei la coordonatele punctului de la pozitia [i][k] (fata i, punctul k)
						din cadrul "matricii" de puncte */
					}
				}
			}
		}

		imshow("Poza cu punctele de interes", imgFace); /* Afisarea imaginii fetei cu punctele de interes desenate */
		waitKey(5);
	}
	else cout << endl << "Eroare la detectia fetei!" << endl; /* Daca (cumva) nu s-a reusit detectia fetei, se afiseaza un mesaj de eroare */
}
int main()
{
	string path = "Tools/JAFFEDBASE"; 
	/* -> Initializarea string-ului ce reprezinta path-ul relativ la imaginea de input
	   -> Toate imaginile din JAFFED sunt in format TIFF
	   -> Primele doua litere sunt initialele candidatei, urmatoarele doua sunt abrevierea emotiei (NE = Neutral, HA = Happy, SA = Sad, DI = Disgust, FE = Fear, SU = Surprised, AN = Angry */


	Mat img = imread(path + "/YM.NE.tiff"); /* Citirea imaginii cu expresia neutra intr-un obiect de tip Mat numit 'img' */
	Mat imgX = imread(path + "/YM.AN.tiff"); /* Citirea imaginii cu emotia care se doreste a fi clasificata intr-un obiect de tip Mat numit 'imgX' */

	/*
	resize(img, img, Size(256, 256), INTER_LINEAR);
	resize(imgX, imgX, Size(256, 256), INTER_LINEAR);
	*/ // Daca exista poze in baza de date de o alta rezolutie inafara de 256 px x 256 px, se da resize la dimensiunea dorita (256 x 256)


	facemark = FacemarkLBF::create(); /* Instantiem un obiect de tip FacemarkLBF, folosind parametrii standard ai constructorului */
	facemark->loadModel("Tools/lbfmodel.yaml.txt"); /* Incarcam un model antrenat deja numit 'lbfmodel', pentru a folosi cu succes functia 'fit' asupra obiectului 'facemark', descrisa mai sus */
	faceDetector.load("Tools/haarcascade_frontalface.xml"); /* Incarcam clasificatorul HAAR din fisierul .xml intr-un obiect de tip CascadeClassifier, numit faceDetector. Acesta a fost initializat global, 
															alaturi de declararea facemark-ului */

	//float lungimeNasN, diagOchiRN, diagOchiLN;
	float lungimeGuraN, latimeGuraN, bazaNasN; /* Declaram variabilele in care se vor memora distantele specifice imaginii cu expresia neutra (N) */
	vector<vector<Point2f>> puncteNE; /* Declaram vectorul de vectori de puncte de interes specifice fetei din imaginea cu expresia neutra */
	detectiePuncte(img, puncteNE, lungimeGuraN, latimeGuraN, bazaNasN); /* Apelam functia de detectie a punctelor de interes, care ne va actualiza variabilele specifice marimilor de interes */


	//float lungimeNasX, diagOchiRX, diagOchiLX;
	/* Urmatoarele 3 linii de cod sunt asemanatoare celor 3 de mai sus, cu mentiunea ca sunt specifice celei de a doua imagini, ce contine expresia pe care dorim s-o clasificam */
	float lungimeGuraX, latimeGuraX, bazaNasX; 
	vector<vector<Point2f>> puncteX;
	detectiePuncte(imgX, puncteX, lungimeGuraX, latimeGuraX, bazaNasX);

	
	/* Dupa rularea algoritmului asupra imaginilor din baza noastra de date si incercarea stabilirii unui numitor comun intre rezultatele pentru aceeasi emotie, am ajuns la urmatoarele concluzii:
		Fericit: 
			1) Lungimea gurii e mai mare cu cel putin 15
			2) Baza nasului e mai mare

		Trist:
			1) Diferenta in lungimea gurii e de maxim 8
			2) Latimea gurii e mai mica cu maxim 6
			3) Baza e mai mare cu maxim 3.5

		Dezgust:
			1) Latimea gurii e mai mare cu cel putin 8
			2) Baza e mai mare

		Frica:
			1) Lungimea gurii e mai mare
			2) Latimea gurii e mai mare cu cel putin 2

		Surprins:
			1) Latimea gurii este mai mare cu cel putin 13
			2) Lungimea gurii este mai mica cu cel putin 4

		Nervos:
			1) Daca niciunul din celelalte cazuri nu e valabil */
	
	cout << endl << "Emotii posibile: Fericit, Trist, Infricosat, Uimit, Dezgustat, Nervos" << endl;

	if ((lungimeGuraX - lungimeGuraN) > 15 && (bazaNasX - bazaNasN) > 0)
	{
		cout << "Emotii posibile: Fericit" << endl;
	}
	else if ((abs(lungimeGuraX - lungimeGuraN) < 8) && ((latimeGuraN - latimeGuraX) < 6) && ((bazaNasX - bazaNasN) < 3.5))
	{
		cout << "Emotii posibile: Trist" << endl;;
	}
	else if ((lungimeGuraX - lungimeGuraN) > 0 && (latimeGuraX - latimeGuraN) > 2)
		cout << "Emotii posibile: Frica" << endl;
	else if ((latimeGuraX - latimeGuraN) > 8 && (bazaNasX - bazaNasN) > 0)
		cout << "Emotii posibile: Dezgust" << endl;
	else if ((latimeGuraX - latimeGuraN) > 13 && (lungimeGuraN - lungimeGuraX) > 4)
		cout << "Emotii posibile: Surprins" << endl;
	else cout << "Emotii posibile: Nervos" << endl;
	
	
	/* Cazuri gresite:

	KA: fear -> sad, angry -> sad
	KL: angry -> sad
	KM: disgust <-> surprised
	YM: disgust -> sad, angry -> sad

	17/24 -> 70% success rate

	Pentru imbunatatirea ratei de succes, ar ajuta mai multe variabile de tipul: lungimea nasului, distanta intre anumite puncte de pe sprancene, marimea ochiului, etc. Odata cu adaugarea
	acestor marimi, creste si complexitatea stabilirii variatiei de marimi specifice pentru fiecare expresie in parte, dar nu vor exista la fel de multe cazuri de false positive.

	*/

	imshow("Original", img);

	waitKey(0);
	return 0;
}