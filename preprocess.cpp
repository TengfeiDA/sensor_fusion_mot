#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "json/json.h"

using namespace std;

static constexpr double Pi = 3.141562954;

struct Vec2d {
  Vec2d() = default;
  Vec2d(double xx, double yy) : x(xx), y(yy) {}
  Vec2d(Json::Value json_vec_2d)
      : x(json_vec_2d[0].asDouble()), y(json_vec_2d[1].asDouble()) {}

  double Length() { return sqrt(x * x + y * y); }

  double x;
  double y;
};

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

struct CameraCalibration {
  CameraCalibration() = default;
  CameraCalibration(Vec3d translation, Quaternion rotation,
                    double _intrinsic[4])
      : translation(translation), rotation(rotation) {
    for (int i = 0; i < 4; ++i) {
      intrinsic[i] = _intrinsic[i];
    }
  }

  Vec3d translation;
  Quaternion rotation;
  double intrinsic[4];
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

constexpr int kPerson = 0, kVehicle = 1, kUnknwon = 2;

struct Track {
  Track() = default;
  Track(uint32_t id, string token, Detection detection)
      : id(id), instance_token(token), detection(detection) {}

  uint32_t id;
  string instance_token;
  int category;
  Detection detection;
};

struct RadarDetection {
  RadarDetection() = default;
  RadarDetection(uint32_t id, double timestamp, Vec2d position, Vec2d velocity)
      : id(id), timestamp(timestamp), position(position), velocity(velocity) {}

  uint32_t id;
  double timestamp;
  Vec2d position;
  Vec2d velocity;
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
          break;
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

  const string category_filename = "data/v1.0-mini/category.json";
  ifstream category_file(category_filename, ios::binary);
  if (!category_file.is_open()) {
    cout << "Fail to open " << category_filename << endl;
    return;
  }

  Json::Value category_json;
  map<string, int> token_category_map;
  if (reader.parse(category_file, category_json)) {
    for (int i = 0; i < category_json.size(); ++i) {
      const string token = category_json[i]["token"].asString();
      const string name = category_json[i]["name"].asString();
      int category = kUnknwon;
      if (name.find("human") != string::npos) {
        category = kPerson;
      } else if (name.find("vehicle") != string::npos) {
        category = kVehicle;
      }
      token_category_map.insert(make_pair(token, category));
    }
  }

  const string instance_filename = "data/v1.0-mini/instance.json";
  ifstream instance_file(instance_filename, ios::binary);
  if (!instance_file.is_open()) {
    cout << "Fail to open " << instance_filename << endl;
    return;
  }

  Json::Value instance_json;
  if (reader.parse(instance_file, instance_json)) {
    for (int t = 0; t < tracks.size(); ++t) {
      for (Track& track : tracks[t]) {
        for (int i = 0; i < instance_json.size(); ++i) {
          if (instance_json[i]["token"].asString() == track.instance_token) {
            const string category_token =
                instance_json[i]["category_token"].asString();
            if (token_category_map.find(category_token) !=
                token_category_map.end()) {
              track.category = token_category_map[category_token];
            } else {
              track.category = kUnknwon;
            }
            break;
          }
        }
      }
    }
  }

  string output_filename =
      "./multi_sensor_mot/data/" + scene_name + "_gt_tracks.dat";
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
      output_file << track.id << "  " << track.category << "  "
                  << detection.position.x << " " << detection.position.y << " "
                  << detection.position.z << " " << detection.size.x << " "
                  << detection.size.y << " " << detection.size.z << " "
                  << detection.angle.z << endl;
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

  string output_filename =
      "./multi_sensor_mot/data/" + scene_name + "_lidar_detections.dat";
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
                          const string& scene_name) {
  const string sample_data_filename = "data/v1.0-mini/sample_data.json";
  ifstream sample_data_file(sample_data_filename, ios::binary);
  if (!sample_data_file.is_open()) {
    cout << "Fail to open " << sample_data_filename << endl;
    return;
  }

  vector<uint64_t> all_camera_timestamps_uint64;
  vector<double> all_camera_timestamps;
  vector<string> all_image_filenames;
  string calibration_token;

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
            uint64_t timestamp = sample_data[i]["timestamp"].asUInt64();
            all_camera_timestamps_uint64.push_back(timestamp);
            all_camera_timestamps.push_back(static_cast<double>(timestamp) /
                                            1e6);
            calibration_token =
                sample_data[i]["calibrated_sensor_token"].asString();
            break;
          }
        }
      }
    }
  }

  const string calibrated_sensor_filename =
      "data/v1.0-mini/calibrated_sensor.json";
  ifstream calibrated_sensor_file(calibrated_sensor_filename, ios::binary);
  if (!calibrated_sensor_file.is_open()) {
    cout << "Fail to open " << calibrated_sensor_filename << endl;
    return;
  }

  CameraCalibration camera_calibration;

  Json::Value calibration;
  if (reader.parse(calibrated_sensor_file, calibration)) {
    for (int i = 0; i < calibration.size(); ++i) {
      if (calibration_token == calibration[i]["token"].asString()) {
        Vec3d translation(calibration[i]["translation"]);
        Quaternion rotation(calibration[i]["rotation"]);
        double camera_intrinsic[4];
        camera_intrinsic[0] =
            calibration[i]["camera_intrinsic"][0][0].asDouble();
        camera_intrinsic[1] =
            calibration[i]["camera_intrinsic"][0][2].asDouble();
        camera_intrinsic[2] =
            calibration[i]["camera_intrinsic"][1][1].asDouble();
        camera_intrinsic[3] =
            calibration[i]["camera_intrinsic"][1][2].asDouble();
        camera_calibration =
            CameraCalibration(translation, rotation, camera_intrinsic);
        break;
      }
    }
  }

  const string ego_pose_filename = "data/v1.0-mini/ego_pose.json";
  ifstream ego_pose_file(ego_pose_filename, ios::binary);
  if (!ego_pose_file.is_open()) {
    cout << "Fail to open " << ego_pose_filename << endl;
    return;
  }

  vector<EgoPose> all_samples_ego_poses;
  Json::Value ego_pose;
  if (reader.parse(ego_pose_file, ego_pose)) {
    for (int t = 0; t < all_camera_timestamps_uint64.size(); ++t) {
      for (int i = 0; i < ego_pose.size(); ++i) {
        const uint64_t ego_pose_timestamp = ego_pose[i]["timestamp"].asUInt64();
        if (all_camera_timestamps_uint64[t] == ego_pose_timestamp) {
          Vec3d angles =
              QuaternionToEulerAngles(Quaternion(ego_pose[i]["rotation"]));
          all_samples_ego_poses.emplace_back(
              ego_pose_timestamp, Vec3d(ego_pose[i]["translation"]), angles.z);
          break;
        }
      }
    }
  }
  ego_pose_file.close();

  const std::string camera_detection_filename =
      "./multi_sensor_mot/data/" + scene_name + "-bboxes.dat";
  std::ifstream srcFile(camera_detection_filename, std::ios::in);

  std::vector<std::vector<std::vector<int>>> camera_detections;
  for (int t = 0; t < all_camera_timestamps.size(); ++t) {
    int detections_num;
    srcFile >> detections_num;
    std::vector<std::vector<int>> detections(detections_num);
    for (int i = 0; i < detections_num; ++i) {
      int id, x, y, w, h, type;
      srcFile >> id >> x >> y >> w >> h >> type;
      detections[i] = {id, x, y, w, h, type};
    }
    camera_detections.push_back(detections);
  }
  srcFile.close();

  string output_filename =
      "./data/samples/" + scene_name + "_camera_detections.dat";
  ofstream output_file(output_filename, ios::out | ios::trunc);
  if (!output_file.is_open()) {
    cout << "Fail to open " << output_filename << endl;
    return;
  }

  output_file << all_camera_timestamps.size() << std::endl;

  output_file << camera_calibration.translation.x << "  "
              << camera_calibration.translation.y << "  "
              << camera_calibration.translation.z << "  "
              << camera_calibration.rotation.w << "  "
              << camera_calibration.rotation.x << "  "
              << camera_calibration.rotation.y << "  "
              << camera_calibration.rotation.z << endl;
  for (int i = 0; i < 4; ++i)
    output_file << camera_calibration.intrinsic[i] << "  ";
  output_file << endl;

  for (int t = 0; t < all_camera_timestamps.size(); ++t) {
    output_file << t + 1 << " " << camera_detections[t].size() << " " << fixed
                << setw(16) << setprecision(6) << all_camera_timestamps[t]
                << endl;
    output_file << all_image_filenames[t] << endl;
    output_file << all_samples_ego_poses[t].position.x << " "
                << all_samples_ego_poses[t].position.y << " "
                << all_samples_ego_poses[t].yaw << endl;
    for (const auto& detections : camera_detections[t]) {
      output_file << detections[0] << " " << detections[1] << " "
                  << detections[2] << " " << detections[3] << " "
                  << detections[4] << " " << detections[5] << endl;
    }
    output_file << endl;
  }
  output_file.close();
  cout << "Save camera detections to " << output_filename << endl;
}

void GetRadarDetections(const vector<string>& all_samples_token,
                        const string& scene_name) {
  const string sample_data_filename = "data/v1.0-mini/sample_data.json";
  ifstream sample_data_file(sample_data_filename, ios::binary);
  if (!sample_data_file.is_open()) {
    cout << "Fail to open " << sample_data_filename << endl;
    return;
  }

  vector<uint64_t> all_radar_timestamps_uint64;
  vector<double> all_radar_timestamps;

  Json::Reader reader;
  Json::Value sample_data;
  if (reader.parse(sample_data_file, sample_data)) {
    for (int t = 0; t < all_samples_token.size(); ++t) {
      for (int i = 0; i < sample_data.size(); ++i) {
        if (all_samples_token[t] == sample_data[i]["sample_token"].asString()) {
          const bool is_key_frame = sample_data[i]["is_key_frame"].asBool();
          const string filename = sample_data[i]["filename"].asString();
          if (is_key_frame && filename.find("/RADAR_FRONT/") != string::npos) {
            uint64_t timestamp = sample_data[i]["timestamp"].asUInt64();
            all_radar_timestamps_uint64.push_back(timestamp);
            all_radar_timestamps.push_back(static_cast<double>(timestamp) /
                                           1e6);
            break;
          }
        }
      }
    }
  }

  const string ego_pose_filename = "data/v1.0-mini/ego_pose.json";
  ifstream ego_pose_file(ego_pose_filename, ios::binary);
  if (!ego_pose_file.is_open()) {
    cout << "Fail to open " << ego_pose_filename << endl;
    return;
  }

  vector<EgoPose> all_samples_ego_poses;
  Json::Value ego_pose;
  if (reader.parse(ego_pose_file, ego_pose)) {
    for (int t = 0; t < all_radar_timestamps_uint64.size(); ++t) {
      for (int i = 0; i < ego_pose.size(); ++i) {
        const uint64_t ego_pose_timestamp = ego_pose[i]["timestamp"].asUInt64();
        if (all_radar_timestamps_uint64[t] == ego_pose_timestamp) {
          Vec3d angles =
              QuaternionToEulerAngles(Quaternion(ego_pose[i]["rotation"]));
          all_samples_ego_poses.emplace_back(
              ego_pose_timestamp, Vec3d(ego_pose[i]["translation"]), angles.z);
          break;
        }
      }
    }
  }
  ego_pose_file.close();

  const std::string gt_tracks_filename =
      "./multi_sensor_mot/data/" + scene_name + "_gt_tracks.dat";
  std::ifstream srcFile(gt_tracks_filename, std::ios::in);

  std::vector<std::vector<Detection>> gt_detections;
  int frames_num = 0;
  srcFile >> frames_num;
  vector<double> all_gt_timestamps;
  vector<int> all_gt_detection_nums;
  for (int t = 0; t < frames_num; ++t) {
    int frame_index, detections_num;
    double timestamp;
    srcFile >> frame_index >> detections_num >> timestamp;
    all_gt_detection_nums.push_back(detections_num);
    all_gt_timestamps.push_back(timestamp);

    std::vector<Detection> detections(detections_num);
    for (int i = 0; i < detections_num; ++i) {
      int t_id, type;
      double x, y, z, w, l, h, yaw;
      srcFile >> t_id >> type >> x >> y >> z >> w >> l >> h >> yaw;
      detections[i] = Detection(t_id, timestamp, Vec3d(x, y, z), Vec3d(w, l, h),
                                Vec3d(0, 0, yaw));
    }
    gt_detections.push_back(detections);
  }
  srcFile.close();

  std::srand(std::time(0));
  std::vector<std::vector<RadarDetection>> radar_detections;
  for (int t = 0; t < all_radar_timestamps.size(); ++t) {
    std::vector<RadarDetection> detections;
    int radar_id = 0;
    int k = 0;
    for (; k < all_gt_timestamps.size(); ++k) {
      if (all_gt_timestamps[k] > all_radar_timestamps[t]) break;
    }
    if (k > 0 && k < all_gt_timestamps.size()) {
      const double time_interval =
          all_gt_timestamps[k] - all_gt_timestamps[k - 1];
      const double weight =
          (all_radar_timestamps[t] - all_gt_timestamps[k - 1]) / time_interval;
      for (int i = 0; i < gt_detections[k - 1].size(); ++i) {
        const Detection& prev = gt_detections[k - 1][i];
        auto iter =
            std::find_if(gt_detections[k].begin(), gt_detections[k].end(),
                         [&prev](const Detection& next) -> bool {
                           return prev.id == next.id;
                         });
        if (iter != gt_detections[k].end()) {
          const double random =
              static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
          if (random < 0.8) {
            const double x =
                prev.position.x + weight * (iter->position.x - prev.position.x);
            const double y =
                prev.position.y + weight * (iter->position.y - prev.position.y);
            const double vx =
                (iter->position.x - prev.position.x) / time_interval;
            const double vy =
                (iter->position.y - prev.position.y) / time_interval;
            detections.emplace_back(++radar_id, all_radar_timestamps[t],
                                    Vec2d(x, y), Vec2d(vx, vy));
          }
        }
      }
    }
    radar_detections.push_back(detections);
  }

  string output_filename =
      "./multi_sensor_mot/data/" + scene_name + "_radar_detections.dat";
  ofstream output_file(output_filename, ios::out | ios::trunc);
  if (!output_file.is_open()) {
    cout << "Fail to open " << output_filename << endl;
    return;
  }

  output_file << all_radar_timestamps.size() << std::endl;

  for (int t = 0; t < all_radar_timestamps.size(); ++t) {
    output_file << t + 1 << " " << radar_detections[t].size() << " " << fixed
                << setw(16) << setprecision(6) << all_radar_timestamps[t]
                << endl;
    output_file << all_samples_ego_poses[t].position.x << " "
                << all_samples_ego_poses[t].position.y << " "
                << all_samples_ego_poses[t].yaw << endl;

    for (int i = 0; i < radar_detections[t].size(); ++i) {
      output_file << radar_detections[t][i].id << " "
                  << radar_detections[t][i].position.x << " "
                  << radar_detections[t][i].position.y << " "
                  << radar_detections[t][i].velocity.x << " "
                  << radar_detections[t][i].velocity.y << endl;
    }
    output_file << endl;
  }
  output_file.close();
  cout << "Save radar detections to " << output_filename << endl;
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
      cout << "scene_index: " << scene_index + 1 << " name: " << scene_name
           << " num_of_samples " << num_of_samples << endl;

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

      // GetFrontCameraImages(all_samples_token,
      //                      scene_name);

      GetRadarDetections(all_samples_token, scene_name);
    }
  }
  srcFile.close();
}

int main(int argc, char** argv) {
  cout << "It's preprocess.\n";

  readDataFromJsonFile();

  return 0;
}