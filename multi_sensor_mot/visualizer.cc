
#include "visualizer.h"

#include <opencv4/opencv2/imgproc/imgproc_c.h>

#include <chrono>
#include <random>

namespace {

const cv::Scalar kYellowColor = cv::Scalar(0, 255, 255);
const cv::Scalar kGreenColor = cv::Scalar(0, 255, 0);
const cv::Scalar kRedColor = cv::Scalar(0, 0, 255);
const cv::Scalar kBlueColor = cv::Scalar(255, 0, 0);
const cv::Scalar kBlackColor = cv::Scalar(0, 0, 0);
const cv::Scalar kWhileColor = cv::Scalar(255, 255, 255);
const cv::Scalar kGrayColor = cv::Scalar(80, 80, 80);

std::string CategoryToString(const Category& category) {
  switch (category) {
    case Category::kUnknwon:
      return "Unknown";
    case Category::kPerson:
      return "Person";
    case Category::kVehicle:
      return "Vehicle";
    default:
      return "Unknown";
  }
}

}  // namespace

Visualizer::Visualizer(const CameraFrame& camera_frame,
                       const std::vector<Track>& tracks,
                       const std::string scene_name)
    : timestamp_(camera_frame.timestamp()),
      world_to_vehicle_(camera_frame.world_to_vehicle()),
      bev_image_(cv::Mat(kImageHeight, kImageWidth, CV_8UC3, kBlackColor)),
      camera_image_(cv::imread(camera_frame.image_filename())) {
  cv::putText(camera_image_, "CAM_FRONT", cv::Point(30, 60),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 2, kGreenColor, 2, CV_AA);
  cv::putText(camera_image_, std::to_string(camera_frame.timestamp()),
              cv::Point(30, 120), cv::FONT_HERSHEY_COMPLEX_SMALL, 2,
              kGreenColor, 2, CV_AA);
  DrawGrids();
  DrawCoordinates();
  DrawTracks(tracks);

  DrawDetectionsOnImage(camera_frame);
  DrawTrackBBoxOnImage(camera_frame, tracks);

  SaveImages(camera_frame.index(), tracks.size(), scene_name);
}

void Visualizer::DrawTracks(const std::vector<Track>& tracks) {
  for (const Track& track : tracks) {
    // if (!track.IsConfirmed()) continue;
    const std::vector<Vec2d> corners = track.GetCorners();
    std::vector<cv::Point> corners_image;
    cv::Point id_pos;
    for (int i = 0; i < corners.size(); ++i) {
      const Vec2d corner_vehicle = world_to_vehicle_.Transform(corners[i]);
      const cv::Point p = FromVehicleToImage(corner_vehicle);
      if (IsPointBeyondGridImage(p)) break;
      corners_image.push_back(p);
      if (i == 3) id_pos = p;
    }
    if (corners_image.size() < 4) continue;
    cv::polylines(bev_image_, corners_image, true, kRedColor, 1, 8, 0);
    cv::putText(bev_image_, std::to_string(track.id()),
                cv::Point(id_pos.x + 2, id_pos.y - 2), cv::FONT_HERSHEY_PLAIN,
                1, kRedColor, 1, CV_AA);
  }
}

void Visualizer::DrawGTTracks(const std::vector<Track>& tracks) {
  for (const Track& track : tracks) {
    const std::vector<Vec2d> corners = track.GetCorners();
    std::vector<cv::Point> corners_image;
    cv::Point id_pos;
    for (int i = 0; i < corners.size(); ++i) {
      const Vec2d corner_vehicle = world_to_vehicle_.Transform(corners[i]);
      const cv::Point p = FromVehicleToImage(corner_vehicle);
      if (IsPointBeyondGridImage(p)) break;
      corners_image.push_back(p);
      if (i == 1) id_pos = p;
    }
    if (corners_image.size() < 4) continue;
    cv::polylines(bev_image_, corners_image, true, kGreenColor, 1, 8, 0);
    cv::putText(bev_image_, std::to_string(track.id()),
                cv::Point(id_pos.x + 2, id_pos.y - 2), cv::FONT_HERSHEY_PLAIN,
                1, kGreenColor, 1, CV_AA);
  }
}

void Visualizer::DrawCoordinates() {
  cv::putText(bev_image_, "T: " + std::to_string(timestamp_), cv::Point(20, 50),
              cv::FONT_HERSHEY_PLAIN, 2, kRedColor, 1, CV_AA);

  const cv::Point frame_x_axis_min =
      FromVehicleToImage(Vec2d(kMaxBackRange, 0));
  const cv::Point frame_x_axis_max =
      FromVehicleToImage(Vec2d(kMaxFrontRange, 0));
  const cv::Point frame_y_axis_min =
      FromVehicleToImage(Vec2d(0, kMaxRightRange));
  const cv::Point frame_y_axis_max =
      FromVehicleToImage(Vec2d(0, kMaxLeftRange));

  const cv::Point label_30m = FromVehicleToImage(Vec2d(30, 0));
  const cv::Point label_60m = FromVehicleToImage(Vec2d(60, 0));

  cv::arrowedLine(bev_image_, frame_x_axis_min, frame_x_axis_max, kRedColor, 2,
                  8, 0, 0.05);
  cv::arrowedLine(bev_image_, frame_y_axis_min, frame_y_axis_max, kBlueColor, 2,
                  8, 0, 0.05);
  cv::putText(bev_image_, "x",
              cv::Point(frame_x_axis_max.x + 25, frame_x_axis_max.y + 25),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, kRedColor, 1, CV_AA);
  cv::putText(bev_image_, "y",
              cv::Point(frame_y_axis_max.x, frame_y_axis_max.y - 15),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, kBlueColor, 1, CV_AA);

  cv::putText(bev_image_, "30m", cv::Point(label_30m.x + 5, label_30m.y),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, kRedColor, 1, CV_AA);
  cv::putText(bev_image_, "60m", cv::Point(label_60m.x + 5, label_60m.y),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, kRedColor, 1, CV_AA);
}

void Visualizer::DrawGrids() {
  const double base_size = 5.0;  // meters
  for (double row = kMaxBackRange; row <= kMaxFrontRange; row += base_size) {
    const int row_p = static_cast<int>((row - kMaxBackRange) / kGridSize);
    const cv::Point p1 = cv::Point(0, row_p);
    const cv::Point p2 = cv::Point(kImageWidth, row_p);
    cv::line(bev_image_, p1, p2, kGrayColor, 1);
  }
  for (double col = kMaxRightRange; col <= kMaxLeftRange; col += base_size) {
    const int col_p = static_cast<int>((col - kMaxRightRange) / kGridSize);
    const cv::Point p1 = cv::Point(col_p, 0);
    const cv::Point p2 = cv::Point(col_p, kImageHeight);
    cv::line(bev_image_, p1, p2, kGrayColor, 1);
  }
}

void Visualizer::SaveImages(const int frame_index, const int tracks_num,
                            const std::string scene_name) {
  const int final_rows = std::max(bev_image_.rows, camera_image_.rows);
  const int final_cols = bev_image_.cols + camera_image_.cols;

  final_image_ = cv::Mat(final_rows, final_cols, CV_8UC3, kBlackColor);

  const int row_offset =
      std::max(static_cast<int>((bev_image_.rows - camera_image_.rows) / 2), 0);

  const cv::Rect camera_region =
      cv::Rect(0, row_offset, camera_image_.cols, camera_image_.rows);
  const cv::Rect bev_region =
      cv::Rect(camera_image_.cols, 0, bev_image_.cols, bev_image_.rows);

  camera_image_.copyTo(final_image_(camera_region));
  bev_image_.copyTo(final_image_(bev_region));

  const std::string filename = "../visualization/" + scene_name + "_frame_" +
                               std::to_string(frame_index) + "_tracks_num_" +
                               std::to_string(tracks_num) + "_" +
                               std::to_string(timestamp_) + ".png";
  cv::imwrite(filename, final_image_);

  std::cout << "Save image as " << filename << std::endl;
}

// 0061
// std::map<int, int> id_time = {{19, 1}, {66, 1},  {37, 1},  {53, 1},  {37, 1},
//                               {17, 1}, {41, 1},  {21, 1},  {57, 1},  {47, 1},
//                               {52, 1}, {84, 10}, {88, 17}, {44, 16}, {90,
//                               19}};

// 0103
// std::map<int, int> id_time = {
//     {5, 1},    {6, 1},   {15, 1},  {29, 2},  {31, 9},   {25, 9},  {49, 9},
//     {28, 9},   {44, 9},  {21, 9},  {49, 9},  {28, 18},  {33, 21}, {67, 23},
//     {55, 23},  {75, 30}, {78, 30}, {79, 30}, {88, 33},  {72, 33}, {74, 35},
//     {108, 35}, {95, 35}, {84, 37}, {63, 37}, {113, 39}, {117, 39}};

// 0053
// std::map<int, int> id_time = {{17, 7},  {44, 9},  {33, 9},  {34, 11}, {50,
// 13},
//                               {14, 15}, {51, 18}, {54, 22}, {38, 22}, {35,
//                               25}, {55, 27}, {31, 29}, {36, 29}};

// 0655
// std::map<int, int> id_time = {
//     {30, 1},  {27, 1},  {7, 1},   {11, 1},  {17, 1},  {22, 1},  {8, 1},
//     {4, 1},   {3, 1},   {22, 3},  {24, 3},  {13, 4},  {17, 4},  {39, 4},
//     {35, 7},  {40, 7},  {49, 7},  {65, 9},  {5, 10},  {59, 13}, {41, 13},
//     {73, 19}, {70, 19}, {31, 20}, {90, 22}, {122, 32}};

// 0757
// std::map<int, int> id_time = {{9, 1},   {12, 4},  {8, 7},   {16, 10}, {15,
// 15},
//                               {17, 21}, {22, 21}, {18, 26}, {19, 26}, {23,
//                               26}, {24, 29}, {26, 30}, {25, 32}, {28, 39},
//                               {29, 39}};
// std::map<int, std::vector<int>> occlusion = {
//     {15, {16, 21}}, {17, {23}}, {8, {23, 24}}};

// 0796
// std::map<int, int> id_time = {{7, 1},   {9, 1},   {3, 1},   {13, 1},  {22,
// 2},
//                               {23, 2},  {25, 4},  {30, 8},  {33, 9},  {28,
//                               11}, {36, 12}, {40, 13}, {41, 17}, {42, 21},
//                               {49, 28}, {50, 30}, {6, 39}};
// std::map<int, std::vector<int>> occlusion = {};

// 0916
// std::map<int, int> id_time = {{18, 1},  {19, 4},  {15, 6},  {33, 6},  {56,
// 8},
//                               {2, 8},   {67, 17}, {65, 18}, {55, 20}, {61,
//                               21}, {68, 21}, {8, 21},  {86, 22}, {85, 22},
//                               {76, 22}, {81, 22}, {90, 38}, {73, 39}};
// std::map<int, std::vector<int>> occlusion = {};

// // 1077
// std::map<int, int> id_time = {
//     {9, 1},   {1, 1},   {3, 1},   {13, 2},  {16, 2},  {8, 3},   {15, 4},
//     {20, 5},  {21, 6},  {24, 8},  {25, 9},  {23, 9},  {18, 10}, {26, 13},
//     {31, 13}, {32, 16}, {39, 20}, {33, 20}, {36, 23}, {40, 25}, {45, 25},
//     {46, 25}, {47, 25}, {50, 27}, {51, 31}, {52, 31}, {61, 34}, {66, 36}};
// std::map<int, std::vector<int>> occlusion = {};

// 1094
// std::map<int, int> id_time = {{19, 1},  {3, 1},   {44, 1},  {45, 1},  {2, 2},
//                               {27, 9},  {43, 13}, {26, 15}, {61, 15}, {65,
//                               15}, {68, 16}, {58, 16}, {70, 19}, {71, 23},
//                               {74, 23}, {78, 26}, {80, 30}, {81, 34}, {82,
//                               36}};
// std::map<int, std::vector<int>> occlusion = {{45, {5, 6, 7, 8}}};

// 1100
// std::map<int, int> id_time = {{21, 1},  {8, 1},   {5, 1},  {14, 1},  {24, 1},
//                               {12, 2},  {2, 5},   {26, 9}, {28, 20}, {20,
//                               26}, {30, 30}, {29, 30}, {32, 35}};
// std::map<int, std::vector<int>> occlusion = {{14, {5}}};

void Visualizer::DrawTrackBBoxOnImage(const CameraFrame& camera_frame,
                                      const std::vector<Track>& tracks) {
  // int count = 0;
  // std::vector<std::vector<int>> data;
  for (const Track& track : tracks) {
    if (track.last_camera_update_timestamp() < camera_frame.timestamp() - 1e-6)
      continue;
    // if (track.category() == Category::kUnknwon) {
    //   continue;
    // }
    // if (id_time.count(track.id()) == 0) {
    //   continue;
    // } else if (camera_frame.index() < id_time[track.id()]) {
    //   continue;
    // } else if (occlusion.count(track.id()) > 0) {
    //   auto iter = std::find(occlusion[track.id()].begin(),
    //                         occlusion[track.id()].end(),
    //                         camera_frame.index());
    //   if (iter != occlusion[track.id()].end()) {
    //     continue;
    //   }
    // }

    AABox bbox;
    if (track.GetProjectionOnImage(world_to_vehicle_, &bbox)) {
      const cv::Rect rect = cv::Rect(bbox.top_left_x(), bbox.top_left_y(),
                                     bbox.width(), bbox.height());
      cv::rectangle(camera_image_, rect, kRedColor, 2, 1, 0);
      cv::putText(
          camera_image_,
          std::to_string(track.id()) + "," + CategoryToString(track.category()),
          cv::Point(bbox.top_left_x(), bbox.top_left_y() + bbox.height() - 5),
          cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, kRedColor, 1, CV_AA);
    }
  }

  // std::string output_filename = "../data/scene-1100-bboxes.dat";
  // std::ofstream output_file(output_filename, std::ios::out | std::ios::app);
  // output_file << count << std::endl;
  // for (const auto& d : data) {
  //   output_file << d[0] << " " << d[1] << " " << d[2] << " " << d[3] << " "
  //               << d[4] << " " << d[5] << std::endl;
  // }
  // output_file << std::endl;
  // output_file.close();
}

void Visualizer::DrawDetectionsOnImage(const CameraFrame& camera_frame) {
  for (const CameraDetection& detection : camera_frame.detections()) {
    const AABox& bbox = detection.bbox();
    const cv::Rect rect = cv::Rect(bbox.top_left_x(), bbox.top_left_y(),
                                   bbox.width(), bbox.height());
    cv::rectangle(camera_image_, rect, kGreenColor, 2, 1, 0);
    cv::putText(camera_image_,
                std::to_string(detection.id()) + "," +
                    CategoryToString(detection.category()),
                cv::Point(bbox.top_left_x(), bbox.top_left_y() - 5),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, kGreenColor, 1, CV_AA);
  }
}