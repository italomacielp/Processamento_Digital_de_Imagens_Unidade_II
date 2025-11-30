#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
  cv::Mat image, smooth, corrected, thresh;

  image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

  if (image.empty()) {
    std::cout << "imagem nao carregou corretamente\n";
    return -1;
  }

  // --- 1. Suavização da imagem para obter o componente de iluminação ---
  cv::GaussianBlur(image, smooth, cv::Size(51, 51), 0);

  // --- 2. Correção da iluminação usando divisão ---
  cv::Mat imageFloat, smoothFloat;
  image.convertTo(imageFloat, CV_32F);
  smooth.convertTo(smoothFloat, CV_32F);

  corrected = imageFloat / (smoothFloat + 1);   // evita divisão por zero
  cv::normalize(corrected, corrected, 0, 255, cv::NORM_MINMAX);
  corrected.convertTo(corrected, CV_8U);

  // --- 3. Limiarização por Otsu ---
  cv::threshold(corrected, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

  // --- 4. Exibição e salvamento ---
  cv::imshow("Imagem original", image);
  cv::imshow("Suavizada (Gauss)", smooth);
  cv::imshow("Corrigida", corrected);
  cv::imshow("Otsu corrigido", thresh);

  cv::imwrite("corrigida.png", corrected);
  cv::imwrite("otsu_corrigido.png", thresh);

  cv::waitKey();
  return 0;
}

