#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

Mat homomorphicFilter(const Mat& Y, double gammaL, double gammaH, double c, double D0) {
    Mat Yfloat;
    Y.convertTo(Yfloat, CV_32F);
    log(Yfloat + 1.0, Yfloat);  // log(Y + 1)

    // FFT
    Mat planes[] = {Yfloat.clone(), Mat::zeros(Yfloat.size(), CV_32F)};
    Mat complexImg;
    merge(planes, 2, complexImg);
    dft(complexImg, complexImg);

    // Construção do filtro
    Mat H = Mat::zeros(Y.size(), CV_32F);
    int M = Y.rows;
    int N = Y.cols;

    for (int u = 0; u < M; u++) {
        for (int v = 0; v < N; v++) {
            double du = u - M / 2;
            double dv = v - N / 2;
            double D = sqrt(du*du + dv*dv);
            H.at<float>(u, v) = 
                (gammaH - gammaL) * (1.0 - exp(-c * (D*D) / (D0*D0)))
                + gammaL;
        }
    }

    // Aplicar filtro
    Mat planesH[2];
    split(complexImg, planesH);
    Mat realF = planesH[0];
    Mat imagF = planesH[1];

    realF = realF.mul(H);
    imagF = imagF.mul(H);

    merge(planesH, 2, complexImg);

    // IDFT
    idft(complexImg, complexImg, DFT_SCALE | DFT_REAL_OUTPUT);

    // exponencial
    exp(complexImg, complexImg);
    complexImg = complexImg - 1.0;

    // normalizar para 0–255
    Mat Yout;
    normalize(complexImg, Yout, 0, 255, NORM_MINMAX);
    Yout.convertTo(Yout, CV_8U);

    return Yout;
}

int main() {
    string filename = "../iluminacao_irregular.jpg";
    Mat img = imread(filename);
    if (img.empty()) {
        cout << "Erro ao carregar imagem\n";
        return -1;
    }

    // Conversão para YCrCb
    Mat imgYCrCb;
    cvtColor(img, imgYCrCb, COLOR_BGR2YCrCb);

    vector<Mat> canais;
    split(imgYCrCb, canais);
    Mat Y = canais[0];

    // Trackbars
    namedWindow("Homomorphic", WINDOW_NORMAL);
    resizeWindow("Homomorphic", 900, 700);
    int gammaL_slider = 10;   // 0.1
    int gammaH_slider = 200;  // 2.0
    int c_slider      = 10;   // 0.1
    int D0_slider     = 30;

    createTrackbar("gammaL (x0.01)", "Homomorphic", &gammaL_slider, 100);
    createTrackbar("gammaH (x0.01)", "Homomorphic", &gammaH_slider, 500);
    createTrackbar("c (x0.1)",       "Homomorphic", &c_slider, 100);
    createTrackbar("D0",             "Homomorphic", &D0_slider, 200);

    while (true) {
        double gammaL = gammaL_slider * 0.01;
        double gammaH = gammaH_slider * 0.01;
        double c      = c_slider * 0.1;
        double D0     = (double)D0_slider;

        if (D0 < 1) D0 = 1;

        Mat Y_filtered = homomorphicFilter(Y, gammaL, gammaH, c, D0);

        // Reconstruir imagem colorida
        Mat outYCrCb;
        vector<Mat> channelsOut = {Y_filtered, canais[1], canais[2]};
        merge(channelsOut, outYCrCb);

        Mat outBGR;
        cvtColor(outYCrCb, outBGR, COLOR_YCrCb2BGR);

        imshow("Homomorphic", outBGR);

        if (waitKey(10) == 27) break; // ESC
    }

    return 0;
}

