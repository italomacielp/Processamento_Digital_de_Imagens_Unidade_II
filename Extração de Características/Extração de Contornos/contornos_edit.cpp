#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

// Função para calcular distância entre vetores de Hu Moments
double distancia(const std::vector<double>& a, const std::vector<double>& b) {
    double soma = 0.0;
    for (int i = 0; i < 7; i++) {
        double d = a[i] - b[i];
        soma += d * d;
    }
    return std::sqrt(soma);
}

// Função para calcular Hu Moments em escala logaritmica
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
        logHu[i] = -1.0 * std::copysign(1.0, hu[i]) * std::log10(std::abs(hu[i]) + 1e-30);
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

    // Redimensiona a imagem da pessoa para acelerar
    double scalePessoa = 0.5;
    cv::Mat imgPessoaRed;
    cv::resize(imgPessoa, imgPessoaRed, cv::Size(), scalePessoa, scalePessoa);

    int pw = imgPessoaRed.cols;
    int ph = imgPessoaRed.rows;

    std::vector<double> huPessoa = calculaHu(imgPessoaRed);

    double melhorDist = 1e18;
    cv::Rect melhorRect;

    // Sliding window simples
    for (int y = 0; y <= imgMultidao.rows - ph; y += 5) {
        for (int x = 0; x <= imgMultidao.cols - pw; x += 5) {
            cv::Rect roi(x, y, pw, ph);
            cv::Mat janela = imgMultidao(roi);

            std::vector<double> huJanela = calculaHu(janela);
            double d = distancia(huPessoa, huJanela);

            if (d < melhorDist) {
                melhorDist = d;
                melhorRect = roi;
            }
        }
    }

    // Mostra resultado colorido
    cv::Mat imgColor = cv::imread(argv[2]);
    cv::Mat imgDisplay = imgColor.clone();

    cv::rectangle(imgDisplay, melhorRect, cv::Scalar(0, 0, 255), 2);

    cv::Point centro(
        melhorRect.x + melhorRect.width / 2,
        melhorRect.y + melhorRect.height / 2
    );
    cv::circle(imgDisplay, centro, 10, cv::Scalar(0, 255, 0), 3);

    std::cout << "Melhor correspondencia: distancia = " << melhorDist << "\n";
    std::cout << "Centro encontrado: x=" << centro.x << " y=" << centro.y << "\n";

    // Ajuste para caber na tela
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

