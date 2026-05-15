#include "inference.h"

#include <MNN/cv/cv.hpp>
#include <algorithm>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

YoloInference::YoloInference(const std::string& model_path, int forward_type, int precision_mode, int num_threads)
    : m_forward_type(forward_type), m_precision_mode(precision_mode), m_num_threads(num_threads) {
  MNN::BackendConfig backendConfig;
  backendConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(precision_mode);

  m_executor = std::shared_ptr<MNN::Express::Executor>(
      MNN::Express::Executor::newExecutor(static_cast<MNNForwardType>(forward_type), backendConfig, num_threads));

  m_net = std::shared_ptr<MNN::Express::Module>(MNN::Express::Module::load({}, {}, model_path.c_str(), m_rtmgr));
  if (!m_net) {
    throw std::runtime_error("Failed to load model: " + model_path);
  }

  auto rt = m_net->getInfo()->runTimeManager;
  if (rt) {
    rt->setCache(".cachefile");
  }
}

YoloInference::~YoloInference() {
  if (m_rtmgr) {
    m_rtmgr->updateCache();
  }
}

std::vector<DetectionResult> YoloInference::run_inference(const std::string& image_path) {
  auto image = MNN::CV::imread(image_path);
  return process_image(image);
}

std::map<std::string, std::vector<DetectionResult>> YoloInference::run_inference_on_directory(const std::string& input_dir) {
  std::map<std::string, std::vector<DetectionResult>> results_map;

  for (const auto& entry : fs::directory_iterator(input_dir)) {
    if (entry.is_regular_file()) {
      const std::string ext = entry.path().extension().string();
      if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
        const std::string image_path = entry.path().string();
        const std::string filename = entry.path().filename().string();
        std::cout << "Processing: " << image_path << std::endl;
        results_map[filename] = run_inference(image_path);
      }
    }
  }
  return results_map;
}

std::vector<DetectionResult> YoloInference::process_image(const MNN::Express::VARP& original_image) {
  auto dims = original_image->getInfo()->dim;
  int ih = dims[0];
  int iw = dims[1];
  int len = std::max(ih, iw);
  float scale = len / static_cast<float>(TARGET_SIZE);

  // Padding
  std::vector<int> padvals{0, len - ih, 0, len - iw, 0, 0};
  auto pads = MNN::Express::_Const(static_cast<void*>(padvals.data()), {3, 2}, MNN::Express::NCHW, halide_type_of<int>());
  auto image = MNN::Express::_Pad(original_image, pads, MNN::Express::CONSTANT);

  // Resize, Normalize, Convert Color
  image = MNN::CV::resize(image, MNN::CV::Size(TARGET_SIZE, TARGET_SIZE), 0, 0, MNN::CV::INTER_LINEAR, -1, {0., 0., 0.},
                          {1. / 255., 1. / 255., 1. / 255.});
  image = MNN::CV::cvtColor(image, MNN::CV::COLOR_BGR2RGB);

  // Prepare input tensor
  auto input = MNN::Express::_Unsqueeze(image, {0});
  input = MNN::Express::_Convert(input, MNN::Express::NC4HW4);  // input->getInfo()->order, input->getInfo()->dim

  // Run inference
  auto outputs = m_net->onForward({input});
  auto output = MNN::Express::_Convert(outputs[0], MNN::Express::NCHW);
  output = MNN::Express::_Squeeze(output);
  output = MNN::Express::_Transpose(output, {1, 0});

  // Post-process based on C reference code structure
  auto x0 = MNN::Express::_Gather(output, MNN::Express::_Scalar<int>(0));
  auto y0 = MNN::Express::_Gather(output, MNN::Express::_Scalar<int>(1));
  auto x1 = MNN::Express::_Gather(output, MNN::Express::_Scalar<int>(2));
  auto y1 = MNN::Express::_Gather(output, MNN::Express::_Scalar<int>(3));
  auto scores = MNN::Express::_Gather(output, MNN::Express::_Scalar<int>(4));
  auto class_ids = MNN::Express::_Gather(output, MNN::Express::_Scalar<int>(5));
  auto boxes = MNN::Express::_Stack({x0, y0, x1, y1}, 1);

  // NMS
  auto result_ids = MNN::Express::_Nms(boxes, scores, 100, 0.8f, 0.25f);

  auto result_ptr = result_ids->readMap<int>();
  auto box_ptr = boxes->readMap<float>();
  auto ids_ptr = class_ids->readMap<int>();
  auto score_ptr = scores->readMap<float>();

  std::vector<DetectionResult> results;
  for (int i = 0; i < 100; ++i) {  // Max detections limit
    auto idx = result_ptr[i];
    if (idx < 0) break;

    float x0 = box_ptr[idx * 4 + 0] * scale;
    float y0 = box_ptr[idx * 4 + 1] * scale;
    float x1 = box_ptr[idx * 4 + 2] * scale;
    float y1 = box_ptr[idx * 4 + 3] * scale;

    // Clamp back to original image size
    x1 = std::min(static_cast<float>(iw), x1);
    y1 = std::min(static_cast<float>(ih), y1);

    int class_idx = ids_ptr[idx];
    float conf = score_ptr[idx];

    results.push_back({x0, y0, x1, y1, class_idx, conf});
  }

  return results;
}