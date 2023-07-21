#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "evaluation.h"
#include "hungarian.h"
#include "lidar_detection.h"
#include "tracker.h"

namespace {

static uint64_t output_filename_timestamp = 0;
std::string GetOutputFilename(const std::string& scene_name) {
  if (output_filename_timestamp == 0) {
    time_t nowtime;
    time(&nowtime);
    output_filename_timestamp = static_cast<uint64_t>(nowtime);
  }
  const std::string output_filename =
      "../results/" + scene_name + "_results_" +
      std::to_string(output_filename_timestamp) + ".dat";
  return output_filename;
}

}  // namespace

std::vector<std::vector<Track>> GetGroundTruthTracksFromFile(
    const std::string& scene_name) {
  const std::string gt_tracks_filename =
      "../data/" + scene_name + "_gt_tracks.dat";
  std::ifstream srcFile(gt_tracks_filename, std::ios::in);
  if (!srcFile.is_open()) {
    std::cout << "Fail to open " << gt_tracks_filename << std::endl;
    return std::vector<std::vector<Track>>();
  }

  int frames_num;
  srcFile >> frames_num;
  std::vector<std::vector<Track>> tracks_list;
  tracks_list.reserve(frames_num);
  for (int i = 0; i < frames_num; ++i) {
    uint32_t frame_index = 0, tracks_num = 0;
    double timestamp = 0;
    srcFile >> frame_index >> tracks_num >> timestamp;

    std::vector<Track> tracks;
    tracks.reserve(tracks_num);
    for (int j = 0; j < tracks_num; ++j) {
      uint32_t id;
      double x, y, z, l, w, h, yaw;
      srcFile >> id >> x >> y >> z >> l >> w >> h >> yaw;
      tracks.emplace_back(
          LidarDetection(id, timestamp, Vec3d(x, y, z), Vec3d(l, w, h), yaw));
      tracks.back().SetId(id);
    }
    tracks_list.push_back(tracks);
  }
  srcFile.close();

  return tracks_list;
}

std::vector<LidarFrame> GetDetectionsFromFile(const std::string& scene_name) {
  const std::string lidar_detection_filename =
      "../data/" + scene_name + "_detections.dat";
  std::ifstream srcFile(lidar_detection_filename, std::ios::in);
  if (!srcFile.is_open()) {
    std::cout << "Fail to open " << lidar_detection_filename << std::endl;
    return std::vector<LidarFrame>();
  }

  int frames_num;
  srcFile >> frames_num;
  std::vector<LidarFrame> lidar_frames;
  lidar_frames.reserve(frames_num);
  for (int i = 0; i < frames_num; ++i) {
    uint32_t frame_index = 0, detections_num = 0;
    double timestamp = 0;
    srcFile >> frame_index >> detections_num >> timestamp;

    double ego_x, ego_y, ego_yaw;
    srcFile >> ego_x >> ego_y >> ego_yaw;
    const Vec2d ego_position(ego_x, ego_y);

    std::vector<LidarDetection> detections;
    detections.reserve(detections_num);
    for (int j = 0; j < detections_num; ++j) {
      uint32_t id;
      double x, y, z, l, w, h, yaw;
      srcFile >> id >> x >> y >> z >> l >> w >> h >> yaw;
      detections.emplace_back(id, timestamp, Vec3d(x, y, z), Vec3d(l, w, h),
                              yaw);
      detections.back().GetDistanceToEgo(ego_position);
    }
    const Transformation2d world_to_vehicle(ego_position, ego_yaw);
    lidar_frames.emplace_back(frame_index, timestamp, world_to_vehicle,
                              detections);
  }
  srcFile.close();

  return lidar_frames;
}

int main(int argc, char** argv) {
  std::cout << "Multiple Object Tracking Based on Lidar Detections.\n";

  const std::string scene_name = argc > 1 ? argv[1] : "scene-0061";
  std::cout << "scene_name : " << scene_name << "\n";

  const std::vector<LidarFrame> lidar_frames =
      GetDetectionsFromFile(scene_name);

  Tracker tracker(GetOutputFilename(scene_name));
  std::vector<std::vector<Track>> published_tracks_list;
  for (const LidarFrame& frame : lidar_frames) {
    tracker.Run(frame);
    published_tracks_list.push_back(tracker.PublishTracks(frame));
  }

  std::vector<std::vector<Track>> gt_tracks_list =
      GetGroundTruthTracksFromFile(scene_name);
  const double mota =
      PerformanceEvaluation(published_tracks_list, gt_tracks_list);

  return 0;
}