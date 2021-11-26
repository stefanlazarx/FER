//Proiect destinat recunoasterii expresiei faciale - Stefan Lazar & Bianca Popescu
//Etape: - Detectia fetei dintr-o imagine din cadrul bazei de date
//       - Scoaterea in evidenta a zonelor de interes si a punctelor de interes din cadrul acestor zone
//       - Clasificarea intr-una din cele 7 emotii, in functie de pozitionarea punctelor de interes fata de un anumit 'standard' de normalitate

#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

using namespace cv::face;


