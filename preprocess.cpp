#include <fstream>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "json/json.h"

using namespace std;

struct Vec3d {
  Vec3d() = default;
  Vec3d(double xx, double yy, double zz) : x(xx), y(yy), z(zz) {}
  Vec3d(Json::Value json_vec_3d)
      : x(json_vec_3d[0].asDouble()),
        y(json_vec_3d[1].asDouble()),
        z(json_vec_3d[2].asDouble()) {}

  double Length() {
	return sqrt(x*x + y*y + z*z);
  }

  double x;
  double y;
  double z;
};

Vec3d operator-(const Vec3d& a, const Vec3d& b) {
	return Vec3d(a.x-b.x, a.y-b.y, a.z-b.z);
}

struct Detection {
	Detection() = default;
	Detection(uint32_t id, double timestamp, Vec3d position, Vec3d size, Vec3d angle) : 
		id(id), timestamp(timestamp), position(position), size(size), angle(angle) {}

	uint32_t id;
	double timestamp;
	Vec3d position;
	Vec3d size;
	Vec3d angle;
};

struct Track {
  Track() = default;
  Track(uint32_t id, string token, Detection detection)
      : track_id(id), instance_token(token) {
		detections.push_back(detection);
	  }

  uint32_t track_id;
  string instance_token;
  vector<Detection> detections;
};

struct Quaternion {
	Quaternion() = default;
	Quaternion(Json::Value json_quaternion) : 
		w(json_quaternion[0].asDouble()),
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
        angles.y = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        angles.y = std::asin(sinp);
 
    // yaw (z-axis rotation)
    double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    angles.z = std::atan2(siny_cosp, cosy_cosp);
 
    return angles;
}

void GetSamplesFromScene(const int& num_of_samples,
                         const string& first_sample_token,
                         const string& last_sample_token,
                         vector<string>* all_samples_token,
                         vector<double>* all_samples_timestamps) {
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
          all_samples_timestamps->push_back(static_cast<double>(sample[i]["timestamp"].asUInt64()) / 1e6);

          cout << "sample_cnt: " << sample_cnt << " token: " << token
               << " timestamp: " << fixed << setw(16) << setprecision(6) << all_samples_timestamps->back()
               << endl;

          current_sample_token = sample[i]["next"].asString();
          if (current_sample_token.empty() || token == last_sample_token) {
            break;
          }
        }
      }
    }
  }
  sample_file.close();
}

void GetTracks(vector<string> all_samples_token, vector<double> all_samples_timestamps) {
    const string sample_annotation_filename =
        "data/v1.0-mini/sample_annotation.json";
    ifstream sample_annotation_file(sample_annotation_filename, ios::binary);
    if (!sample_annotation_file.is_open()) {
      cout << "Fail to open " << sample_annotation_filename << endl;
      return;
    }

    uint32_t next_id = 1;
    map<string, uint32_t> instance_token_id_map;
    vector<Track> tracks;

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
			Vec3d angles = QuaternionToEulerAngles(Quaternion(sample_annotation[i]["rotation"]));
			Detection detection(++detection_id, all_samples_timestamps[t], position, size, angles);
            auto iter = instance_token_id_map.find(instance_token);
            if (iter == instance_token_id_map.end()) {
              tracks.emplace_back(next_id, instance_token, detection);
              instance_token_id_map.insert(
                  make_pair(instance_token, next_id++));
            } else {
              const uint32_t current_track_index = iter->second - 1;
              tracks[current_track_index].detections.push_back(detection);
            }
          }
        }
      }
    }
    for (const Track& track : tracks) {
	  if (track.detections.size() < 10) continue;
	  Vec3d distance = track.detections.back().position - track.detections.front().position;
	  if (distance.Length() < 10.0) continue;

      cout << "track: id " << track.track_id << " instance_token "
           << track.instance_token << " num_of_positions "
           << track.detections.size() << endl;
      for (int i = 0; i < track.detections.size(); ++i) {
        const auto& detection = track.detections[i];
        cout << "T: " << fixed << setw(16) << setprecision(6) << detection.timestamp
		<< "  position: " << detection.position.x << " " << detection.position.y << " " << detection.position.z
		<< "  size: " << detection.size.x << " " << detection.size.y << " " << detection.size.z
		<< "  angle: " << detection.angle.x << " " << detection.angle.y << " " << detection.angle.z << endl;
      }
      cout << endl;
    }
}


void GetDetections(vector<string> all_samples_token, vector<double> all_samples_timestamps) {
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
			Vec3d angles = QuaternionToEulerAngles(Quaternion(sample_annotation[i]["rotation"]));
			Detection detection(++detection_id, all_samples_timestamps[t], position, size, angles);
            detections[t].push_back(detection);
          }
        }
      }
    }

	string output_filename = "detections.dat";
	ofstream output_file(output_filename, ios::out | ios::trunc);
    if (!output_file.is_open()) {
      cout << "Fail to open " << output_filename << endl;
      return;
    }

    for (int t = 0; t < all_samples_timestamps.size(); ++t) {
      output_file << "frame: " << t+1 << " detections_num: " << detections[t].size() << " timestamp: "
           << fixed << setw(16) << setprecision(6) << all_samples_timestamps[t] << endl;

      for (int i = 0; i < detections[t].size(); ++i) {
        const auto& detection = detections[t][i];
        output_file << "detection: id: " << detection.id
		<< "  position: " << detection.position.x << " " << detection.position.y << " " << detection.position.z
		<< "  size: " << detection.size.x << " " << detection.size.y << " " << detection.size.z
		<< "  angle: " << detection.angle.x << " " << detection.angle.y << " " << detection.angle.z << endl;
      }
      output_file << endl;
    }
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
    string name = scene[0]["name"].asString();
    string token = scene[0]["token"].asString();
    int num_of_samples = scene[0]["nbr_samples"].asInt();
    string first_sample_token = scene[0]["first_sample_token"].asString();
    string last_sample_token = scene[0]["last_sample_token"].asString();
    cout << "scene name: " << name << " num_of_samples " << num_of_samples
         << " first_sample_token " << first_sample_token
         << " last_sample_token " << last_sample_token << endl;

    vector<string> all_samples_token;
    vector<double> all_samples_timestamps;
    GetSamplesFromScene(num_of_samples, first_sample_token, last_sample_token,
                        &all_samples_token, &all_samples_timestamps);

	GetDetections(all_samples_token, all_samples_timestamps);

  }
  srcFile.close();
}

int main(int argc, char** argv) {
  cout << "It's preprocess.\n";

  readDataFromJsonFile();

  return 0;
}