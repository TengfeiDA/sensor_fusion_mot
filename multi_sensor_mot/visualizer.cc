
#include "visualizer.h"

#include <opencv4/opencv2/imgproc/imgproc_c.h>

namespace {

const cv::Scalar kYellowColor = cv::Scalar(0, 255, 255);
const cv::Scalar kGreenColor = cv::Scalar(0, 255, 0);
const cv::Scalar kRedColor = cv::Scalar(0, 0, 255);
const cv::Scalar kBlueColor = cv::Scalar(255, 0, 0);
const cv::Scalar kBlackColor = cv::Scalar(0, 0, 0);
const cv::Scalar kWhileColor = cv::Scalar(255, 255, 255);
const cv::Scalar kGrayColor = cv::Scalar(80, 80, 80);

}  // namespace

void Visualizer::DrawTracks(const std::vector<Track>& tracks) {
  for (const Track& track : tracks) {
    if (track.size().x() < 0.7 && track.size().y() < 0.7) continue;
    if (!track.IsConfirmed()) continue;
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
    // if (track.size().x() < 0.7 && track.size().y() < 0.7) continue;
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

Visualizer::Visualizer(const LidarFrame& lidar_frame,
                       const CameraFrame& camera_frame,
                       const std::vector<Track>& tracks)
    : timestamp_(lidar_frame.timestamp()),
      world_to_vehicle_(lidar_frame.world_to_vehicle()),
      bev_image_(cv::Mat(kImageHeight, kImageWidth, CV_8UC3, kBlackColor)) {
  DrawGrids();
  DrawCoordinates();
  DrawTracks(tracks);

  DrawBBoxOnImage(camera_frame);

  SaveImages(lidar_frame.index(), tracks.size());
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

void Visualizer::SaveImages(const int frame_index, const int tracks_num) {
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

  const std::string filename =
      "../visualization/frame_" + std::to_string(frame_index) + "_tracks_num_" +
      std::to_string(tracks_num) + "_" + std::to_string(timestamp_) + ".png";
  cv::imwrite(filename, final_image_);

  std::cout << "Save image as " << filename << std::endl;
}

void Visualizer::DrawBBoxOnImage(const CameraFrame& camera_frame) {
  camera_image_ = cv::imread(camera_frame.image_filename());

  cv::putText(camera_image_, "CAM_FRONT", cv::Point(30, 60),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 3, kGreenColor, 2, CV_AA);

  cv::putText(camera_image_, std::to_string(camera_frame.timestamp()),
              cv::Point(30, 140), cv::FONT_HERSHEY_SIMPLEX, 2, kGreenColor, 2,
              CV_AA);
}