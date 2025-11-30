#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

double distancia(const std::vector<double>& a, const std::vector<double>& b) {
    double soma = 0.0;
    for (int i = 0; i < 7; i++) {
        double d = a[i] - b[i];
        soma += d * d;
    }
    return std::sqrt(soma);
}

std::vector<double> calculaHu(const cv::Mat& img) {
    cv::Mat bin;
    // Threshold adaptativo
    cv::adaptiveThreshold(img, bin, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY, 11, 2);

    cv::Moments moment = cv::moments(bin, true);

    double hu[7];
    cv::HuMoments(moment, hu);

    std::vector<double> logHu(7);
    for (int i = 0; i < 7; i++) {
        logHu[i] = -1.0 * std::copysign(1.0, hu[i]) * std::log10(std::abs(hu[i]) + 1e-30); // evitar log10(0)
    }
    return logHu;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Uso: ./localiza pessoa.jpg multidao.jpg\n";
        return 0;
    }

    cv::Mat imgPessoa = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat imgMultidao = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    if (imgPessoa.empty() || imgMultidao.empty()) {
        std::cout << "Erro ao carregar imagens.\n";
        return 0;
    }

    // Redimensiona a pessoa para acelerar
    double scalePessoa = 0.5;
    cv::resize(imgPessoa, imgPessoa, cv::Size(), scalePessoa, scalePessoa);

    int pw = imgPessoa.cols;
    int ph = imgPessoa.rows;

    std::vector<double> huPessoa = calculaHu(imgPessoa);

    double melhorDist = 1e18;
    cv::Rect melhorRect;

    // Sliding window multi-escala
    std::vector<double> escalas = {0.8, 0.9, 1.0, 1.1, 1.2};

    for (int y = 0; y <= imgMultidao.rows - ph; y += 2) {   // passo menor = mais preciso
        for (int x = 0; x <= imgMultidao.cols - pw; x += 2) {
            cv::Rect roi(x, y, pw, ph);
            cv::Mat janela = imgMultidao(roi);

            for (double s : escalas) {
                cv::Mat janelaRedim;
                cv::resize(janela, janelaRedim, cv::Size(), s, s);
                if (janelaRedim.cols != pw || janelaRedim.rows != ph) continue;

                std::vector<double> huJanela = calculaHu(janelaRedim);
                double d = distancia(huPessoa, huJanela);

                if (d < melhorDist) {
                    melhorDist = d;
                    melhorRect = roi;
                }
            }
        }
    }

    // Mostra resultado colorido
    cv::Mat imgColor = cv::imread(argv[2]);
    cv::Mat imgDisplay = imgColor.clone();

    // desenha retângulo
    cv::rectangle(imgDisplay, melhorRect, cv::Scalar(0, 0, 255), 2);

    // desenha círculo no centro
    cv::Point centro(
        melhorRect.x + melhorRect.width / 2,
        melhorRect.y + melhorRect.height / 2
    );
    cv::circle(imgDisplay, centro, 10, cv::Scalar(0, 255, 0), 3);

    // Impressão das coordenadas
    std::cout << "Melhor correspondencia: distancia = " << melhorDist << "\n";
    std::cout << "Centro encontrado: x=" << centro.x
              << " y=" << centro.y << "\n";

    // Ajusta a janela para caber na tela
    int screenWidth = 1366;  
    int screenHeight = 768;  

    double fx = (double)screenWidth / imgDisplay.cols;
    double fy = (double)screenHeight / imgDisplay.rows;
    double f = std::min(fx, fy);

    cv::Mat imgShow;
    cv::resize(imgDisplay, imgShow, cv::Size(), f, f);

    cv::imshow("Localizacao Encontrada", imgShow);
    cv::waitKey(0);

    return 0;
}

