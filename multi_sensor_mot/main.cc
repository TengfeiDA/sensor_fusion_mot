#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "camera_detection.h"
#include "evaluation.h"
#include "lidar_detection.h"
#include "math.h"
#include "tracker.h"
#include "visualizer.h"

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
      int category_int;
      double x, y, z, l, w, h, yaw;
      srcFile >> id >> category_int >> x >> y >> z >> l >> w >> h >> yaw;

      Category category = category_int == 0   ? Category::kPerson
                          : category_int == 1 ? Category::kVehicle
                                              : Category::kUnknwon;
      tracks.emplace_back(
          LidarDetection(id, timestamp, Vec3d(x, y, z), Vec3d(l, w, h), yaw),
          category);
      tracks.back().SetId(id);
    }
    tracks_list.push_back(tracks);
  }
  srcFile.close();

  return tracks_list;
}

std::vector<LidarFrame> GetLidarDetectionsFromFile(
    const std::string& scene_name) {
  const std::string lidar_detection_filename =
      "../data/" + scene_name + "_lidar_detections.dat";
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

std::vector<CameraFrame> GetCameraDetectionsFromFile(
    const std::string& scene_name) {
  const std::string camera_detection_filename =
      "../data/" + scene_name + "_camera_detections.dat";
  std::ifstream srcFile(camera_detection_filename, std::ios::in);
  if (!srcFile.is_open()) {
    std::cout << "Fail to open " << camera_detection_filename << std::endl;
    return std::vector<CameraFrame>();
  }

  int frames_num;
  srcFile >> frames_num;

  double x, y, z;
  srcFile >> x >> y >> z;
  const Vec3d translation(x, y, z);
  double rotation[4];
  double intrinsic[4];
  srcFile >> rotation[0] >> rotation[1] >> rotation[2] >> rotation[3];
  srcFile >> intrinsic[0] >> intrinsic[1] >> intrinsic[2] >> intrinsic[3];
  Track::SetCameraProjection(translation, rotation, intrinsic);

  std::vector<CameraFrame> camera_frames;
  camera_frames.reserve(frames_num);
  for (int i = 0; i < frames_num; ++i) {
    uint32_t frame_index = 0, detections_num = 0;
    double timestamp = 0;
    srcFile >> frame_index >> detections_num >> timestamp;

    std::string filename;
    srcFile >> filename;

    double ego_x, ego_y, ego_yaw;
    srcFile >> ego_x >> ego_y >> ego_yaw;
    const Vec2d ego_position(ego_x, ego_y);
    const Transformation2d world_to_vehicle(ego_position, ego_yaw);

    std::vector<CameraDetection> detections;
    detections.reserve(detections_num);
    for (int j = 0; j < detections_num; ++j) {
      int id, x, y, w, h, type;
      srcFile >> id >> x >> y >> w >> h >> type;
      detections.emplace_back(id, timestamp, type, AABox(x, y, w, h));
    }

    camera_frames.emplace_back(frame_index, timestamp, filename,
                               world_to_vehicle, detections);
  }
  srcFile.close();

  return camera_frames;
}

// void GenerateCameraDetections(
//     const std::vector<std::vector<Track>>& tracks_list,
//     const std::vector<LidarFrame>& lidar_frames,
//     const std::vector<CameraFrame>& camera_frames) {
//   for (int i = 0; i < tracks_list.size(); ++i) {
//     Visualizer(lidar_frames[i], camera_frames[i], tracks_list[i]);
//   }
// }

int main(int argc, char** argv) {
  std::cout << "Multiple Object Tracking Based on Lidar Detections.\n";

  const std::string scene_name = argc > 1 ? argv[1] : "scene-0103";
  std::cout << "scene_name : " << scene_name << "\n";

  const std::vector<LidarFrame> lidar_frames =
      GetLidarDetectionsFromFile(scene_name);

  const std::vector<CameraFrame> camera_frames =
      GetCameraDetectionsFromFile(scene_name);

  Tracker tracker(scene_name);
  std::vector<std::vector<Track>> published_tracks_list;

  int lidar_frame_idx = 0;
  int camera_frame_idx = 0;
  while (lidar_frame_idx < lidar_frames.size() ||
         camera_frame_idx < camera_frames.size()) {
    if (camera_frame_idx >= camera_frames.size()) {
      tracker.Run(lidar_frames[lidar_frame_idx++]);
      published_tracks_list.push_back(tracker.PublishTracks());
    } else if (lidar_frame_idx >= lidar_frames.size()) {
      tracker.Run(camera_frames[camera_frame_idx++]);
    } else if (lidar_frames[lidar_frame_idx].timestamp() <=
               camera_frames[camera_frame_idx].timestamp()) {
      tracker.Run(lidar_frames[lidar_frame_idx++]);
      published_tracks_list.push_back(tracker.PublishTracks());
    } else {
      tracker.Run(camera_frames[camera_frame_idx++]);
    }
  }

  std::vector<std::vector<Track>> gt_tracks_list =
      GetGroundTruthTracksFromFile(scene_name);
  const double mota =
      PerformanceEvaluation(published_tracks_list, gt_tracks_list);

  // GenerateCameraDetections(gt_tracks_list, lidar_frames, camera_frames);

  return 0;
}