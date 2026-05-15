#include "result.h"

#include <yaml-cpp/yaml.h>

#include <MNN/cv/cv.hpp>
#include <fstream>
#include <iostream>

ResultProcessor::ResultProcessor(const std::string& config_path) {
  YAML::Node config = YAML::LoadFile(config_path);
  if (config["names"]) {
    for (auto it = config["names"].begin(); it != config["names"].end(); ++it) {
      m_class_names[it->first.as<int>()] = it->second.as<std::string>();
    }
  } else {
    std::cerr << "Error: 'names' key not found in the YAML file." << std::endl;
    return;
  }
}

void ResultProcessor::draw_and_save_image(const std::string& image_path, const std::vector<DetectionResult>& detections,
                                          const std::string& output_path) {
  auto image = MNN::CV::imread(image_path);
  for (const auto& det : detections) {
    MNN::CV::rectangle(image, {det.x0, det.y0}, {det.x1, det.y1}, {0, 0, 255}, 2);  // Red color, thickness 2
  }

  MNN::CV::imwrite(output_path, image);
  std::cout << "Annotated image saved to: " << output_path << std::endl;
}

std::vector<LabelMeShape> ResultProcessor::convert_to_labelme_shapes(const std::vector<DetectionResult>& detections) {
  std::vector<LabelMeShape> shapes;
  for (const auto& det : detections) {
    LabelMeShape shape;
    std::string name = "unknown_" + std::to_string(det.class_id);
    auto it = m_class_names.find(det.class_id);
    if (it != m_class_names.end()) {
      name = it->second;
    }
    shape.points = {{det.x0, det.y0}, {det.x1, det.y1}};
    shapes.push_back(shape);
  }
  return shapes;
}

void ResultProcessor::save_labelme_json(const std::string& image_path, const std::vector<DetectionResult>& detections,
                                        const std::string& json_output_path) {
  Json::Value root;
  root["version"] = "4.5.6";  // Example version
  root["flags"] = Json::objectValue;
  root["shapes"] = Json::arrayValue;

  auto shapes_data = convert_to_labelme_shapes(detections);
  for (const auto& shape : shapes_data) {
    Json::Value shape_obj;
    shape_obj["label"] = shape.label;
    shape_obj["points"] = Json::arrayValue;
    for (const auto& point : shape.points) {
      Json::Value pt(Json::arrayValue);
      pt.append(point[0]);
      pt.append(point[1]);
      shape_obj["points"].append(pt);
    }
    shape_obj["shape_type"] = shape.shape_type;
    shape_obj["flags"] = shape.flags;
    root["shapes"].append(shape_obj);
  }

  // Add image info (optional but common)
  auto temp_img = MNN::CV::imread(image_path);
  auto dims = temp_img->getInfo()->dim;
  root["imageHeight"] = dims[0];
  root["imageWidth"] = dims[1];

  std::ofstream file(json_output_path);
  if (file.is_open()) {
    Json::StreamWriterBuilder builder;
    builder.settings_["indentation"] = "  ";  // Pretty print
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(root, &file);
    file.close();
    std::cout << "LabelMe JSON saved to: " << json_output_path << std::endl;
  } else {
    std::cerr << "Failed to open file for writing: " << json_output_path << std::endl;
  }
}