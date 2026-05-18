#ifndef INFERENCE_H
#define INFERENCE_H

#include <MNN/ImageProcess.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include <memory>
#include <string>
#include <vector>

#include "result.h"

class YoloInference {
 public:
  /**
   * @brief Constructor initializes the inference engine.
   * @param model_path Path to the .mnn model file.
   * @param forward_type Forward type (CPU, OPENCL, etc.).
   * @param precision_mode Precision mode (Normal, High, Low).
   * @param num_threads Number of threads for inference.
   */
  YoloInference(const std::string& model_path, int forward_type = MNN_FORWARD_CPU, int precision_mode = 0, int num_threads = 1);

  ~YoloInference();

  /**
   * @brief Run inference on a single image file.
   * @param image_path Path to the input image.
   * @return Vector of DetectionResult objects.
   */
  std::vector<DetectionResult> run_inference(const std::string& image_path);

  /**
   * @brief Run inference on all images in a directory.
   * @param input_dir Path to the input directory containing images.
   * @return Map of image filename to its detection results.
   */
  std::map<std::string, std::vector<DetectionResult>> run_inference_on_directory(const std::string& input_dir);

 private:
  std::shared_ptr<MNN::Express::Module> m_net;
  std::shared_ptr<MNN::Express::Executor::RuntimeManager> m_rtmgr;
  int m_forward_type;
  int m_precision_mode;
  int m_num_threads;

  // Preprocessing parameters (hardcoded based on C reference)
  static constexpr int TARGET_SIZE = 640;

  /**
   * @brief Internal preprocessing and inference logic.
   * @param image The loaded MNN CV Mat.
   * @return Vector of DetectionResult objects.
   */
  std::vector<DetectionResult> process_image(const MNN::Express::VARP& image);
};

#endif  // INFERENCE_H