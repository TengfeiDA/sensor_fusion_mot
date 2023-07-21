
#include "tracker.h"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <string>

#include "visualizer.h"

namespace {

static constexpr double kTrackLidarDetectionDistanceGating = 3.0;
double GetTrackLidarDetectionDistance(const LidarDetection& detection,
                                      const Track& track) {
  const Vec3d error = detection.position() - track.position();
  return error.Length();
}

}  // namespace

void Tracker::Run(const LidarFrame& lidar_frame,
                  const CameraFrame& camera_frame) {
  std::cout << "\nLidar frame_index: " << lidar_frame.index()
            << " detections_num: " << lidar_frame.detections().size()
            << " timestamp: " << std::fixed << std::setw(16)
            << std::setprecision(6) << lidar_frame.timestamp() << std::endl;

  PredictTracks(lidar_frame.timestamp());

  std::vector<std::pair<int, int>> association_pairs;
  std::vector<int> unassociated_track_indices;
  std::vector<int> unassociated_detection_indices;
  DataAssociation(lidar_frame, &association_pairs, &unassociated_track_indices,
                  &unassociated_detection_indices);

  UpdateTracks(lidar_frame, association_pairs);

  ManagementTracks(lidar_frame, unassociated_track_indices,
                   unassociated_detection_indices);

  Visualizer(lidar_frame, camera_frame, tracks_);
}

void Tracker::PredictTracks(const double& timestamp) {
  for (Track& track : tracks_) {
    track.Predict(timestamp);
  }
}

void Tracker::DataAssociation(
    const LidarFrame& lidar_frame,
    std::vector<std::pair<int, int>>* association_pairs,
    std::vector<int>* unassociated_track_indices,
    std::vector<int>* unassociated_detection_indices) {
  if (association_pairs == nullptr || unassociated_track_indices == nullptr ||
      unassociated_detection_indices == nullptr) {
    return;
  }

  if (lidar_frame.detections().empty()) {
    std::cout << "lidar_frame is empty! index: " << lidar_frame.index()
              << " timestamp: " << std::fixed << std::setw(16)
              << std::setprecision(6) << lidar_frame.timestamp() << std::endl;
    return;
  }

  if (tracks_.empty()) {
    std::cout << "tracks is empty! lidar_frame index: " << lidar_frame.index()
              << " timestamp: " << std::fixed << std::setw(16)
              << std::setprecision(6) << lidar_frame.timestamp() << std::endl;
    unassociated_detection_indices->resize(lidar_frame.detections().size());
    std::iota(unassociated_detection_indices->begin(),
              unassociated_detection_indices->end(), 0);
    return;
  }

  const std::vector<LidarDetection>& detections = lidar_frame.detections();
  std::vector<bool> is_track_associated(tracks_.size(), false);
  for (int i = 0; i < detections.size(); ++i) {
    double min_distance = kTrackLidarDetectionDistanceGating;
    int nearest_track_index = -1;
    for (int j = 0; j < tracks_.size(); ++j) {
      if (is_track_associated[j]) continue;
      const double distance =
          GetTrackLidarDetectionDistance(detections[i], tracks_[j]);
      if (distance < min_distance) {
        min_distance = distance;
        nearest_track_index = j;
      }
    }
    if (nearest_track_index > 0) {
      association_pairs->emplace_back(nearest_track_index, i);
      is_track_associated[nearest_track_index] = true;
    } else {
      unassociated_detection_indices->push_back(i);
    }
  }

  for (int i = 0; i < is_track_associated.size(); ++i) {
    if (!is_track_associated[i]) {
      unassociated_track_indices->push_back(i);
    }
  }
}

void Tracker::UpdateTracks(
    const LidarFrame& lidar_frame,
    const std::vector<std::pair<int, int>>& association_pairs) {
  const std::vector<LidarDetection>& detections = lidar_frame.detections();
  for (const auto& pair : association_pairs) {
    const int track_index = pair.first;
    const int detection_index = pair.second;
    tracks_[track_index].Update(detections[detection_index]);
  }
}

void Tracker::ManagementTracks(
    const LidarFrame& lidar_frame,
    const std::vector<int>& unassociated_track_indices,
    const std::vector<int>& unassociated_detection_indices) {
  // Create new tracks
  const std::vector<LidarDetection>& detections = lidar_frame.detections();
  for (const int index : unassociated_detection_indices) {
    tracks_.emplace_back(detections[index]);
  }

  // Remove lost tracks
  auto iter = tracks_.begin();
  while (iter != tracks_.end()) {
    if (iter->IsLost()) {
      iter = tracks_.erase(iter);
    } else {
      ++iter;
    }
  }
}

std::vector<Track> Tracker::PublishTracks(const LidarFrame& lidar_frame) {
  std::vector<Track> published_tracks;
  for (const Track& track : tracks_) {
    if (track.IsConfirmed()) {
      published_tracks.push_back(track);
    }
  }

  std::ofstream output_file(output_filename_, std::ios::out | std::ios::app);

  if (!output_file.is_open()) {
    std::cout << "Fail to open " << output_filename_ << std::endl;
    return published_tracks;
  }

  output_file << lidar_frame.index() << " " << published_tracks.size() << " "
              << std::fixed << std::setw(16) << std::setprecision(6)
              << lidar_frame.timestamp() << std::endl;

  if (!published_tracks.empty()) {
    for (int i = 0; i < tracks_.size(); ++i) {
      if (tracks_[i].IsConfirmed()) {
        output_file << tracks_[i].id() << "  " << tracks_[i].position().x()
                    << " " << tracks_[i].position().y() << " "
                    << tracks_[i].position().z() << " "
                    << tracks_[i].velocity().x() << " "
                    << tracks_[i].velocity().y() << " "
                    << tracks_[i].velocity().z() << " "
                    << tracks_[i].acceleration().x() << " "
                    << tracks_[i].acceleration().y() << " "
                    << tracks_[i].acceleration().z() << " "
                    << tracks_[i].size().x() << " " << tracks_[i].size().y()
                    << " " << tracks_[i].size().z() << " " << tracks_[i].yaw()
                    << " " << tracks_[i].yaw_rate() << std::endl;
      }
    }
  }

  output_file << std::endl;
  output_file.close();
  std::cout << "Publish " << published_tracks.size() << " tracks to "
            << output_filename_ << " at timestamp: " << std::fixed
            << std::setw(16) << std::setprecision(6) << lidar_frame.timestamp()
            << std::endl;
  return published_tracks;
}