#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "math.h"

class CameraFrame {
 public:
  CameraFrame() = default;
  ~CameraFrame() = default;

  CameraFrame(uint32_t index, double timestamp,
              const std::string& image_filename)
      : index_(index), timestamp_(timestamp), image_filename_(image_filename) {}

  uint32_t index() const { return index_; }
  double timestamp() const { return timestamp_; }
  std::string image_filename() const { return image_filename_; }

 private:
  uint32_t index_;
  double timestamp_;
  std::string image_filename_;
};
