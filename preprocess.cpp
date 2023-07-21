#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "json/json.h"

using namespace std;

static constexpr double Pi = 3.141562954;

struct Vec3d {
  Vec3d() = default;
  Vec3d(double xx, double yy, double zz) : x(xx), y(yy), z(zz) {}
  Vec3d(Json::Value json_vec_3d)
      : x(json_vec_3d[0].asDouble()),
        y(json_vec_3d[1].asDouble()),
        z(json_vec_3d[2].asDouble()) {}

  double Length() { return sqrt(x * x + y * y + z * z); }

  double x;
  double y;
  double z;
};

Vec3d operator+(const Vec3d& a, const Vec3d& b) {
  return Vec3d(a.x + b.x, a.y + b.y, a.z + b.z);
}

Vec3d operator-(const Vec3d& a, const Vec3d& b) {
  return Vec3d(a.x - b.x, a.y - b.y, a.z - b.z);
}

struct Quaternion {
  Quaternion() = default;
  Quaternion(Json::Value json_quaternion)
      : w(json_quaternion[0].asDouble()),
        x(json_quaternion[1].asDouble()),
        y(json_quaternion[2].asDouble()),
        z(json_quaternion[3].asDouble()) {}

  double w;
  double x;
  double y;
  double z;
};

Vec3d QuaternionToEulerAngles(Quaternion q) {
  Vec3d angles;

  // roll (x-axis rotation)
  double sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
  double cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
  angles.x = std::atan2(sinr_cosp, cosr_cosp);

  // pitch (y-axis rotation)
  double sinp = 2 * (q.w * q.y - q.z * q.x);
  if (std::abs(sinp) >= 1)
    angles.y = std::copysign(M_PI / 2, sinp);  // use 90 degrees if out of range
  else
    angles.y = std::asin(sinp);

  // yaw (z-axis rotation)
  double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
  double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
  angles.z = std::atan2(siny_cosp, cosy_cosp);

  return angles;
}

struct EgoPose {
  EgoPose() = default;
  EgoPose(uint64_t timestamp, Vec3d position, double yaw)
      : timestamp(timestamp), position(position), yaw(yaw) {}

  uint64_t timestamp;
  Vec3d position;
  double yaw;
};

struct Detection {
  Detection() = default;
  Detection(uint32_t id, double timestamp, Vec3d position, Vec3d size,
            Vec3d angle)
      : id(id),
        timestamp(timestamp),
        position(position),
        size(size),
        angle(angle) {}

  uint32_t id;
  double timestamp;
  Vec3d position;
  Vec3d size;
  Vec3d angle;
};

struct Track {
  Track() = default;
  Track(uint32_t id, string token, Detection detection)
      : id(id), instance_token(token), detection(detection) {}

  uint32_t id;
  string instance_token;
  Detection detection;
};

void AddRandomNoise(const Vec3d ego_position, Vec3d* position, Vec3d* size,
                    Vec3d* angles) {
  const double distance_x = position->x - ego_position.x;
  const double distance_y = position->y - ego_position.y;
  const double distance =
      sqrt(distance_x * distance_x + distance_y * distance_y);
  const double distance_scale = distance < 30 ? 1.0 : 0.05 * distance - 0.5;

  const double area = size->x * size->y;
  const double area_scale = area < 4.0 ? 1.0 : area < 10.0 ? 1.1 : 1.2;

  const double scale = distance_scale * area_scale;

  // 从epoch（1970年1月1日00:00:00
  // UTC）开始经过的纳秒数，unsigned类型会截断这个值
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);

  const double position_std = 0.2 * scale;
  const double size_std = 0.1 * scale;
  const double angle_std = (2.0 / 180.0 * Pi) * scale;

  // 第一个参数为高斯分布的平均值，第二个参数为标准差
  std::normal_distribution<double> position_gaussian(0.0, position_std);
  std::normal_distribution<double> size_gaussian(0.0, size_std);
  std::normal_distribution<double> angle_gaussian(0.0, angle_std);

  const Vec3d position_noise(position_gaussian(generator),
                             position_gaussian(generator), 0);
  const Vec3d size_noise(size_gaussian(generator), size_gaussian(generator), 0);
  const Vec3d angle_noise(angle_gaussian(generator), angle_gaussian(generator),
                          angle_gaussian(generator));

  *position = *position + position_noise;
  *size = *size + size_noise;
  *angles = *angles + angle_noise;
}

void GetSamplesFromScene(const int& num_of_samples,
                         const string& first_sample_token,
                         const string& last_sample_token,
                         vector<string>* all_samples_token,
                         vector<double>* all_samples_timestamps,
                         vector<uint64_t>* all_samples_timestamps_uint64,
                         vector<EgoPose>* all_samples_ego_poses) {
  const string sample_filename = "data/v1.0-mini/sample.json";
  ifstream sample_file(sample_filename, ios::binary);
  if (!sample_file.is_open()) {
    cout << "Fail to open " << sample_filename << endl;
    return;
  }

  Json::Reader reader;
  Json::Value sample;
  if (reader.parse(sample_file, sample)) {
    int sample_cnt = 0;
    string current_sample_token = first_sample_token;
    while (sample_cnt < num_of_samples) {
      for (int i = 0; i < sample.size(); ++i) {
        string token = sample[i]["token"].asString();
        if (token == current_sample_token) {
          ++sample_cnt;
          all_samples_token->push_back(token);
          all_samples_timestamps_uint64->push_back(
              sample[i]["timestamp"].asUInt64());
          all_samples_timestamps->push_back(
              static_cast<double>(sample[i]["timestamp"].asUInt64()) / 1e6);

          current_sample_token = sample[i]["next"].asString();
          if (current_sample_token.empty() || token == last_sample_token) {
            break;
          }
        }
      }
    }
  }
  sample_file.close();

  const string ego_pose_filename = "data/v1.0-mini/ego_pose.json";
  ifstream ego_pose_file(ego_pose_filename, ios::binary);
  if (!ego_pose_file.is_open()) {
    cout << "Fail to open " << ego_pose_filename << endl;
    return;
  }

  Json::Value ego_pose;
  if (reader.parse(ego_pose_file, ego_pose)) {
    for (int t = 0; t < all_samples_timestamps_uint64->size(); ++t) {
      for (int i = 0; i < ego_pose.size(); ++i) {
        const uint64_t ego_pose_timestamp = ego_pose[i]["timestamp"].asUInt64();
        if ((*all_samples_timestamps_uint64)[t] == ego_pose_timestamp) {
          Vec3d angles =
              QuaternionToEulerAngles(Quaternion(ego_pose[i]["rotation"]));
          all_samples_ego_poses->emplace_back(
              ego_pose_timestamp, Vec3d(ego_pose[i]["translation"]), angles.z);
        }
      }
    }
  }
  ego_pose_file.close();
}

void GetTracks(const vector<string>& all_samples_token,
               const vector<double>& all_samples_timestamps,
               const string& scene_name) {
  const string sample_annotation_filename =
      "data/v1.0-mini/sample_annotation.json";
  ifstream sample_annotation_file(sample_annotation_filename, ios::binary);
  if (!sample_annotation_file.is_open()) {
    cout << "Fail to open " << sample_annotation_filename << endl;
    return;
  }

  uint32_t next_id = 1;
  map<string, uint32_t> instance_token_id_map;
  vector<vector<Track>> tracks(all_samples_token.size());

  Json::Reader reader;
  Json::Value sample_annotation;
  if (reader.parse(sample_annotation_file, sample_annotation)) {
    for (int t = 0; t < all_samples_token.size(); ++t) {
      auto& current_sample_token = all_samples_token[t];
      uint32_t detection_id = 0;
      for (int i = 0; i < sample_annotation.size(); ++i) {
        const string sample_token =
            sample_annotation[i]["sample_token"].asString();
        if (current_sample_token == sample_token) {
          const string instance_token =
              sample_annotation[i]["instance_token"].asString();
          Vec3d position(sample_annotation[i]["translation"]);
          Vec3d size(sample_annotation[i]["size"]);
          Vec3d angles = QuaternionToEulerAngles(
              Quaternion(sample_annotation[i]["rotation"]));
          Detection detection(++detection_id, all_samples_timestamps[t],
                              position, size, angles);
          auto iter = instance_token_id_map.find(instance_token);
          if (iter == instance_token_id_map.end()) {
            tracks[t].emplace_back(next_id, instance_token, detection);
            instance_token_id_map.insert(make_pair(instance_token, next_id++));
          } else {
            const uint32_t current_track_index = iter->second - 1;
            tracks[t].emplace_back(iter->second, instance_token, detection);
          }
        }
      }
    }
  }

  string output_filename = "./lidar_mot/data/" + scene_name + "_gt_tracks.dat";
  ofstream output_file(output_filename, ios::out | ios::trunc);
  if (!output_file.is_open()) {
    cout << "Fail to open " << output_filename << endl;
    return;
  }

  output_file << all_samples_timestamps.size() << std::endl;
  for (int t = 0; t < all_samples_timestamps.size(); ++t) {
    output_file << t + 1 << " " << tracks[t].size() << " " << fixed << setw(16)
                << setprecision(6) << all_samples_timestamps[t] << endl;

    for (int i = 0; i < tracks[t].size(); ++i) {
      const auto& track = tracks[t][i];
      const auto& detection = track.detection;
      output_file << track.id << "  " << detection.position.x << " "
                  << detection.position.y << " " << detection.position.z << " "
                  << detection.size.x << " " << detection.size.y << " "
                  << detection.size.z << " " << detection.angle.z << endl;
    }
    output_file << endl;
  }

  output_file.close();
  cout << "Save ground truth tracks to " << output_filename << endl;
}

void GetDetections(const vector<string>& all_samples_token,
                   const vector<double>& all_samples_timestamps,
                   const vector<EgoPose>& all_samples_ego_poses,
                   const string& scene_name, const bool add_noise = true) {
  const string sample_annotation_filename =
      "data/v1.0-mini/sample_annotation.json";
  ifstream sample_annotation_file(sample_annotation_filename, ios::binary);
  if (!sample_annotation_file.is_open()) {
    cout << "Fail to open " << sample_annotation_filename << endl;
    return;
  }

  vector<vector<Detection>> detections(all_samples_token.size());

  Json::Reader reader;
  Json::Value sample_annotation;
  if (reader.parse(sample_annotation_file, sample_annotation)) {
    for (int t = 0; t < all_samples_token.size(); ++t) {
      auto& current_sample_token = all_samples_token[t];
      uint32_t detection_id = 0;
      for (int i = 0; i < sample_annotation.size(); ++i) {
        const string sample_token =
            sample_annotation[i]["sample_token"].asString();
        if (current_sample_token == sample_token) {
          const string instance_token =
              sample_annotation[i]["instance_token"].asString();
          Vec3d position(sample_annotation[i]["translation"]);
          Vec3d size(sample_annotation[i]["size"]);
          Vec3d angles = QuaternionToEulerAngles(
              Quaternion(sample_annotation[i]["rotation"]));
          if (add_noise) {
            AddRandomNoise(all_samples_ego_poses[t].position, &position, &size,
                           &angles);
          }
          Detection detection(++detection_id, all_samples_timestamps[t],
                              position, size, angles);
          detections[t].push_back(detection);
        }
      }
    }
  }

  string output_filename = "./lidar_mot/data/" + scene_name + "_detections.dat";
  ofstream output_file(output_filename, ios::out | ios::trunc);
  if (!output_file.is_open()) {
    cout << "Fail to open " << output_filename << endl;
    return;
  }

  output_file << all_samples_timestamps.size() << std::endl;
  for (int t = 0; t < all_samples_timestamps.size(); ++t) {
    output_file << t + 1 << "  " << detections[t].size() << "  " << fixed
                << setw(16) << setprecision(6) << all_samples_timestamps[t]
                << endl;
    output_file << all_samples_ego_poses[t].position.x << " "
                << all_samples_ego_poses[t].position.y << " "
                << all_samples_ego_poses[t].yaw << endl;

    for (int i = 0; i < detections[t].size(); ++i) {
      const auto& detection = detections[t][i];
      output_file << detection.id << "  " << detection.position.x << " "
                  << detection.position.y << " " << detection.position.z << "  "
                  << detection.size.x << " " << detection.size.y << " "
                  << detection.size.z << "  " << detection.angle.z << endl;
    }
    output_file << endl;
  }
  output_file.close();
  cout << "Save raw detections to " << output_filename << endl;
}

void GetFrontCameraImages(const vector<string>& all_samples_token,
                          const vector<double>& all_samples_timestamps,
                          const string& scene_name) {
  const string sample_data_filename = "data/v1.0-mini/sample_data.json";
  ifstream sample_data_file(sample_data_filename, ios::binary);
  if (!sample_data_file.is_open()) {
    cout << "Fail to open " << sample_data_filename << endl;
    return;
  }

  vector<double> all_camera_timestamps;
  vector<string> all_image_filenames;

  Json::Reader reader;
  Json::Value sample_data;
  if (reader.parse(sample_data_file, sample_data)) {
    for (int t = 0; t < all_samples_token.size(); ++t) {
      for (int i = 0; i < sample_data.size(); ++i) {
        if (all_samples_token[t] == sample_data[i]["sample_token"].asString()) {
          const bool is_key_frame = sample_data[i]["is_key_frame"].asBool();
          const string filename = sample_data[i]["filename"].asString();
          if (is_key_frame && filename.find("/CAM_FRONT/") != string::npos) {
            const string new_filename =
                "../data/" + filename.substr(8, filename.length());
            all_image_filenames.push_back(new_filename);
            all_camera_timestamps.push_back(
                static_cast<double>(sample_data[i]["timestamp"].asUInt64()) /
                1e6);
            break;
          }
        }
      }
    }
  }

  string output_filename =
      "./multi_sensor_mot/data/" + scene_name + "_camera_detections.dat";
  ofstream output_file(output_filename, ios::out | ios::trunc);
  if (!output_file.is_open()) {
    cout << "Fail to open " << output_filename << endl;
    return;
  }

  output_file << all_camera_timestamps.size() << std::endl;
  for (int t = 0; t < all_camera_timestamps.size(); ++t) {
    output_file << t + 1 << "  " << fixed << setw(16) << setprecision(6)
                << all_camera_timestamps[t] << endl;
    output_file << all_image_filenames[t] << endl;
    output_file << endl;
  }
  output_file.close();
  cout << "Save camera detections to " << output_filename << endl;
}

void readDataFromJsonFile() {
  const string scene_filename = "data/v1.0-mini/scene.json";
  ifstream srcFile(scene_filename, ios::binary);
  if (!srcFile.is_open()) {
    cout << "Fail to open " << scene_filename << endl;
    return;
  }

  Json::Reader reader;
  Json::Value scene;
  if (reader.parse(srcFile, scene)) {
    for (int scene_index = 0; scene_index < scene.size(); ++scene_index) {
      string scene_name = scene[scene_index]["name"].asString();
      string token = scene[scene_index]["token"].asString();
      int num_of_samples = scene[scene_index]["nbr_samples"].asInt();
      string first_sample_token =
          scene[scene_index]["first_sample_token"].asString();
      string last_sample_token =
          scene[scene_index]["last_sample_token"].asString();
      cout << "scene name: " << scene_name << " num_of_samples "
           << num_of_samples << " first_sample_token " << first_sample_token
           << " last_sample_token " << last_sample_token << endl;

      vector<string> all_samples_token;
      vector<double> all_samples_timestamps;
      vector<EgoPose> all_samples_ego_poses;
      vector<uint64_t> all_samples_timestamps_uint64;
      GetSamplesFromScene(num_of_samples, first_sample_token, last_sample_token,
                          &all_samples_token, &all_samples_timestamps,
                          &all_samples_timestamps_uint64,
                          &all_samples_ego_poses);

      // GetDetections(all_samples_token, all_samples_timestamps,
      //               all_samples_ego_poses, scene_name);

      // GetTracks(all_samples_token, all_samples_timestamps, scene_name);

      GetFrontCameraImages(all_samples_token, all_samples_timestamps,
                           scene_name);
    }
  }
  srcFile.close();
}

int main(int argc, char** argv) {
  cout << "It's preprocess.\n";

  readDataFromJsonFile();

  return 0;
}