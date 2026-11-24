// Directly exercise the C++ OpenCV adapter's text, confidence, and word boxes.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/text/ocr.hpp>

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: qualify_native <tessdata directory> <8-bit PGM crop> [...]\n";
        return 2;
    }
    try {
        auto ocr = cv::text::OCRTesseract::create(argv[1], "eng", "", 1, 7);
        std::cout << "crop\ttext\tconfidence\tleft\ttop\twidth\theight\n";
        for (int imageIndex = 2; imageIndex < argc; ++imageIndex) {
            std::ifstream stream(argv[imageIndex], std::ios::binary);
            std::string format;
            int width = 0, height = 0, maximum = 0;
            if (!(stream >> format >> width >> height >> maximum) ||
                format != "P5" || width <= 0 || height <= 0 ||
                width > 4096 || height > 4096 || maximum != 255 || stream.get() != '\n') {
                std::cerr << "Invalid PGM crop " << argv[imageIndex] << '\n';
                return 3;
            }
            cv::Mat image(height, width, CV_8UC1);
            if (!stream.read(reinterpret_cast<char*>(image.data), width * height)) {
                std::cerr << "Incomplete PGM crop " << argv[imageIndex] << '\n';
                return 3;
            }
            std::string text;
            std::vector<cv::Rect> boxes;
            std::vector<std::string> words;
            std::vector<float> confidence;
            ocr->run(image, text, &boxes, &words, &confidence, cv::text::OCR_LEVEL_WORD);
            if (words.empty() || words.size() != boxes.size() || words.size() != confidence.size()) {
                std::cerr << "OpenCV OCR adapter returned incomplete word data\n";
                return 4;
            }
            for (std::size_t i = 0; i < words.size(); ++i) {
                const auto& box = boxes[i];
                std::cout << imageIndex - 2 << '\t' << words[i] << '\t' << std::setprecision(9)
                          << confidence[i] << '\t' << box.x << '\t' << box.y << '\t'
                          << box.width << '\t' << box.height << '\n';
            }
        }
    } catch (const cv::Exception& error) {
        std::cerr << error.what() << '\n';
        return 5;
    } catch (...) {
        std::cerr << "OpenCV Tesseract initialization or recognition failed\n";
        return 5;
    }
}
