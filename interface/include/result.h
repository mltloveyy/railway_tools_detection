#ifndef RESULT_H
#define RESULT_H

#include <json/json.h>

#include <string>
#include <vector>

struct DetectionResult {
  float x0, y0, x1, y1;
  int class_id;
  float confidence;
};

struct LabelMeShape {
  std::string label;
  std::vector<std::vector<double>> points;  // [[x1, y1], [x2, y2]]
  std::string shape_type = "rectangle";
  int flags = 0;
};

class ResultProcessor {
 public:
  /**
   * @brief Constructor.
   * @param class_names A vector of class names corresponding to class IDs from the model.
   */
  ResultProcessor(const std::string& config_path);

  /**
   * @brief Draw detection results on an image and save it.
   * @param image_path Path to the input image.
   * @param detections Vector of detection results.
   * @param output_path Path to save the annotated image.
   */
  void draw_and_save_image(const std::string& image_path, const std::vector<DetectionResult>& detections,
                           const std::string& output_path);

  /**
   * @brief Generate a LabelMe JSON string from detections and save it.
   * @param image_path Path to the source image (used for metadata).
   * @param detections Vector of detection results.
   * @param json_output_path Path to save the JSON file.
   */
  void save_labelme_json(const std::string& image_path, const std::vector<DetectionResult>& detections,
                         const std::string& json_output_path);

 private:
  std::map<int, std::string> m_class_names;
  std::vector<LabelMeShape> convert_to_labelme_shapes(const std::vector<DetectionResult>& detections);
};

#endif  // RESULT_H