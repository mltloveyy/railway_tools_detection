#include <filesystem>
#include <iostream>
#include <vector>

#include "inference.h"
#include "result.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <model_path> <input_image_or_dir> <config_path>\n";
    return -1;
  }

  std::string model_path = argv[1];
  std::string input_path = argv[2];
  std::string config_path = argv[3];

  YoloInference infer(model_path);
  ResultProcessor processor(config_path);

  if (fs::is_directory(input_path)) {
    auto dir_results = infer.run_inference_on_directory(input_path);
    for (const auto& [filename, detections] : dir_results) {
      std::string output_img_path = input_path + "/results_" + filename;
      std::string output_json_path = input_path + "/" + fs::path(filename).replace_extension(".json").string();
      processor.draw_and_save_image(input_path + "/" + filename, detections, output_img_path);
      processor.save_labelme_json(input_path + "/" + filename, detections, output_json_path);
    }
  } else {
    auto detections = infer.run_inference(input_path);
    std::string output_img_path = "annotated_" + fs::path(input_path).filename().string();
    std::string output_json_path = fs::path(input_path).replace_extension(".json").string();
    processor.draw_and_save_image(input_path, detections, output_img_path);
    processor.save_labelme_json(input_path, detections, output_json_path);
  }

  return 0;
}