#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <chrono>

cv::Mat imageGray, edges;
int canny_thresh = 50;
int STEP = 5;
int JITTER = 3;
int RADIUS = 3;

// --- Ajuste automático da imagem na tela ---
cv::Mat resizeToScreen(const cv::Mat &img) {
    int screenW = 1366;
    int screenH = 768;

    float scaleW = (float)screenW / img.cols;
    float scaleH = (float)screenH / img.rows;
    float scale = std::min(std::min(scaleW, scaleH), 1.0f);

    cv::Mat out;
    cv::resize(img, out, cv::Size(), scale, scale);
    return out;
}

// --- Função principal de atualização ---
void update(int, void*) {
    if (imageGray.empty()) return;

    // 1) Rodar Canny
    cv::Canny(imageGray, edges, canny_thresh, canny_thresh * 3);

    // 2) Criar imagem branca
    cv::Mat points(imageGray.rows, imageGray.cols, CV_8U, cv::Scalar(255));

    // Preparar ranges
    std::vector<int> xrange(imageGray.rows / STEP);
    std::vector<int> yrange(imageGray.cols / STEP);
    std::iota(xrange.begin(), xrange.end(), 0);
    std::iota(yrange.begin(), yrange.end(), 0);

    for (auto &i : xrange) i = i * STEP + STEP / 2;
    for (auto &j : yrange) j = j * STEP + STEP / 2;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine rng(seed);

    std::shuffle(xrange.begin(), xrange.end(), rng);

    // 3) Desenhar pontos grandes (pontilhismo base)
    for (auto x : xrange) {
        std::shuffle(yrange.begin(), yrange.end(), rng);
        for (auto y : yrange) {
            int xx = x + (std::rand() % (2 * JITTER) - JITTER);
            int yy = y + (std::rand() % (2 * JITTER) - JITTER);

            xx = std::clamp(xx, 0, imageGray.rows - 1);
            yy = std::clamp(yy, 0, imageGray.cols - 1);

            int gray = imageGray.at<uchar>(xx, yy);
            cv::circle(points, cv::Point(yy, xx), RADIUS, cv::Scalar(gray), cv::FILLED, cv::LINE_AA);
        }
    }

    // 4) Pontos pequenos nas bordas do Canny
    for (int i = 0; i < edges.rows; i++) {
        for (int j = 0; j < edges.cols; j++) {
            if (edges.at<uchar>(i, j) > 0) {
                int gray = imageGray.at<uchar>(i, j);
                cv::circle(points, cv::Point(j, i), std::max(1, RADIUS / 2), cv::Scalar(gray), cv::FILLED);
            }
        }
    }

    // 5) Redimensionar para caber na tela
    cv::Mat disp = resizeToScreen(points);

    cv::imshow("Canny Points", disp);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Uso: ./cannypoints_interactive imagem.png\n";
        return 0;
    }

    imageGray = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (imageGray.empty()) {
        std::cout << "Erro ao carregar imagem.\n";
        return -1;
    }

    cv::namedWindow("Canny Points", cv::WINDOW_NORMAL);

    // Trackbars
    cv::createTrackbar("Canny Thresh", "Canny Points", &canny_thresh, 300, update);
    cv::createTrackbar("STEP", "Canny Points", &STEP, 20, update);
    cv::createTrackbar("JITTER", "Canny Points", &JITTER, 20, update);
    cv::createTrackbar("RADIUS", "Canny Points", &RADIUS, 20, update);

    update(0, 0);
    cv::waitKey();

    return 0;
}

