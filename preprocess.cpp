#include <fstream>
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

  double x;
  double y;
  double z;
};

struct Track {
  Track() = default;
  Track(uint32_t id, string token, vector<Vec3d> ps, vector<Vec3d> ss)
      : track_id(id), instance_token(token), positions(ps), sizes(ss) {}

  uint32_t track_id;
  string instance_token;
  vector<Vec3d> positions;
  vector<Vec3d> sizes;
};

void GetSamplesFromScene(const int& num_of_samples,
                         const string& first_sample_token,
                         const string& last_sample_token,
                         vector<string>* all_samples_token,
                         vector<uint64_t>* all_samples_timestamps) {
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
          all_samples_timestamps->push_back(sample[i]["timestamp"].asUInt64());

          const double t =
              static_cast<double>(all_samples_timestamps->back()) / 1e6;
          cout << "sample_cnt: " << sample_cnt << " token: " << token
               << " timestamp: " << fixed << setw(17) << setprecision(6) << t
               << endl;

          current_sample_token = sample[i]["next"].asString();
          if (current_sample_token.empty() || token == last_sample_token) {
            cout << "token: " << token << " last_sample_token "
                 << last_sample_token << " current_sample_token "
                 << current_sample_token << " current_sample_token.empty() "
                 << current_sample_token.empty() << endl;
            break;
          }
        }
      }
    }
  }
  sample_file.close();
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
    vector<uint64_t> all_samples_timestamps;
    GetSamplesFromScene(num_of_samples, first_sample_token, last_sample_token,
                        &all_samples_token, &all_samples_timestamps);

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

    Json::Value sample_annotation;
    if (reader.parse(sample_annotation_file, sample_annotation)) {
      for (const string& current_sample_token : all_samples_token) {
        cout << "current_sample_token: " << current_sample_token << endl;
        for (int i = 0; i < sample_annotation.size(); ++i) {
          const string sample_token =
              sample_annotation[i]["sample_token"].asString();
          //   cout << "i: " << i
          //        << "  current_sample_token: " << current_sample_token
          //        << "  sample_token: " << sample_token
          //        << "  current_sample_token == sample_token: "
          //        << (current_sample_token == sample_token) << endl;
          if (current_sample_token == sample_token) {
            const string instance_token =
                sample_annotation[i]["instance_token"].asString();
            Vec3d position(sample_annotation[i]["translation"]);
            Vec3d size(sample_annotation[i]["size"]);
            auto iter = instance_token_id_map.find(instance_token);
            if (iter == instance_token_id_map.end()) {
              vector<Vec3d> positions;
              positions.push_back(position);
              vector<Vec3d> sizes;
              sizes.push_back(size);
              tracks.emplace_back(next_id, instance_token, positions, sizes);
              instance_token_id_map.insert(
                  make_pair(instance_token, next_id++));
              cout << "Add new track: id " << next_id - 1 << " instance_token "
                   << instance_token << endl;
            } else {
              //   cout << "instance_token in map: " << iter->first << endl;
              const uint32_t current_track_index = iter->second - 1;
              tracks[current_track_index].positions.push_back(position);
              tracks[current_track_index].sizes.push_back(size);
            }
          }
        }
      }
    }
    for (const Track& track : tracks) {
      cout << "track: id " << track.track_id << " instance_token "
           << track.instance_token << " num_of_positions "
           << track.positions.size() << endl;
      for (int i = 0; i < track.positions.size(); ++i) {
        const auto& position = track.positions[i];
        const auto& size = track.sizes[i];
        cout << position.x << " " << position.y << " " << position.z << " "
             << size.x << " " << size.y << " " << size.z << endl;
      }
      cout << endl;
    }
  }
  srcFile.close();
}

int main(int argc, char** argv) {
  cout << "It's preprocess.\n";

  readDataFromJsonFile();

  return 0;
}