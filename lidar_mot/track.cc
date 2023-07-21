
#include "track.h"

namespace {
static constexpr double kInitPositionVariance = 1.0;
static constexpr double kInitVelocityVariance = 4.0;
static constexpr double kInitAccelerationVariance = 1.0;
static constexpr double kInitYawVariance = 0.01;
static constexpr double kInitYawRateVariance = 0.01;

static constexpr double kPredictPositionVariance = 0.25;
static constexpr double kPredictVelocityVariance = 1.0;
static constexpr double kPredictAccelerationVariance = 0.25;
static constexpr double kPredictYawVariance = 0.01;
static constexpr double kPredictYawRateVariance = 0.01;

static constexpr double kLidarPositionNoiseVariance = 0.25;
static constexpr double kLidarYawNoiseVariance = 0.01;

static constexpr double kSmoothFilterCoefficient = 0.7;

static constexpr double kReliableDetectionRange = 50.0;  // meter

Vec2d RotateByTheta(const Vec2d p, const double theta) {
  return Vec2d(p.x() * std::cos(theta) + p.y() * std::sin(theta),
               p.y() * std::cos(theta) - p.x() * std::sin(theta));
}

}  // namespace

uint32_t Track::next_track_id_ = 1;

Track::Track(const LidarDetection& detection)
    : id_(next_track_id_++),
      created_timestamp_(detection.timestamp()),
      timestamp_(detection.timestamp()),
      last_update_timestamp_(detection.timestamp()),
      position_(detection.position()),
      size_(detection.size()),
      yaw_(detection.yaw()),
      associated_lidar_detections_cnt_(1) {
  velocity_ = Vec3d();
  acceleration_ = Vec3d();
  const Eigen::Matrix<double, 9, 1>& state{
      position_.x(), position_.y(), position_.z(), 0.0, 0.0,
      0.0,           0.0,           0.0,           0.0};
  Eigen::Matrix<double, 9, 9> covariance = Eigen::Matrix<double, 9, 9>::Zero();
  covariance(0, 0) = covariance(1, 1) = covariance(2, 2) =
      kInitPositionVariance;
  covariance(3, 3) = covariance(4, 4) = covariance(5, 5) =
      kInitVelocityVariance;
  covariance(6, 6) = covariance(7, 7) = covariance(8, 8) =
      kInitAccelerationVariance;

  motion_kf_.set_state(state);
  motion_kf_.set_covariance(covariance);

  yaw_rate_ = 0.0;
  const Eigen::Matrix<double, 2, 1>& yaw_state{yaw_, yaw_rate_};
  Eigen::Matrix<double, 2, 2> yaw_covariance =
      Eigen::Matrix<double, 2, 2>::Zero();
  yaw_covariance(0, 0) = kInitYawVariance;
  yaw_covariance(1, 1) = kInitYawRateVariance;

  yaw_kf_.set_state(yaw_state);
  yaw_kf_.set_covariance(yaw_covariance);
}

void Track::Predict(const double timestamp) {
  if (timestamp <= timestamp_) return;
  const double dt = timestamp - timestamp_;
  const double dt2 = dt * dt;

  timestamp_ = timestamp;

  Eigen::Matrix<double, 9, 9> transition_matrix;
  transition_matrix << 1, 0, 0, dt, 0, 0, dt2, 0, 0, 0, 1, 0, 0, dt, 0, 0, dt2,
      0, 0, 0, 1, 0, 0, dt, 0, 0, dt2, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0,
      1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;

  Eigen::Matrix<double, 9, 9> predict_noise_cov =
      Eigen::Matrix<double, 9, 9>::Zero();
  predict_noise_cov(0, 0) = predict_noise_cov(1, 1) = predict_noise_cov(2, 2) =
      kPredictPositionVariance;
  predict_noise_cov(3, 3) = predict_noise_cov(4, 4) = predict_noise_cov(5, 5) =
      kPredictVelocityVariance;
  predict_noise_cov(6, 6) = predict_noise_cov(7, 7) = predict_noise_cov(8, 8) =
      kPredictAccelerationVariance;

  motion_kf_.Predict(transition_matrix, predict_noise_cov);
  position_ = Vec3d(motion_kf_.state()[0], motion_kf_.state()[1],
                    motion_kf_.state()[2]);
  velocity_ = Vec3d(motion_kf_.state()[3], motion_kf_.state()[4],
                    motion_kf_.state()[5]);
  acceleration_ = Vec3d(motion_kf_.state()[6], motion_kf_.state()[7],
                        motion_kf_.state()[8]);

  const Eigen::Matrix<double, 2, 2> yaw_transition_matrix{{1, dt}, {0, 1}};

  Eigen::Matrix<double, 2, 2> yaw_predict_noise_cov{
      {kPredictYawVariance, 0}, {0, kPredictYawRateVariance}};

  yaw_kf_.Predict(yaw_transition_matrix, yaw_predict_noise_cov);
  yaw_ = yaw_kf_.state()[0];
  yaw_rate_ = yaw_kf_.state()[1];
}

void Track::Update(const LidarDetection& detection) {
  const Eigen::Matrix<double, 3, 1> measurement{detection.position().x(),
                                                detection.position().y(),
                                                detection.position().z()};
  const double noise_scale =
      detection.distance_to_ego() < kReliableDetectionRange ? 1.0 : 2.0;
  const Eigen::Matrix<double, 3, 3> measure_noise_cov =
      noise_scale * kLidarPositionNoiseVariance *
      Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::Matrix<double, 3, 9> measure_matrix =
      Eigen::Matrix<double, 3, 9>::Zero();
  measure_matrix.block(0, 0, 3, 3) = Eigen::Matrix<double, 3, 3>::Identity();
  motion_kf_.Update<3>(measurement, measure_noise_cov, measure_matrix);

  position_ = Vec3d(motion_kf_.state()[0], motion_kf_.state()[1],
                    motion_kf_.state()[2]);
  velocity_ = Vec3d(motion_kf_.state()[3], motion_kf_.state()[4],
                    motion_kf_.state()[5]);
  acceleration_ = Vec3d(motion_kf_.state()[6], motion_kf_.state()[7],
                        motion_kf_.state()[8]);

  const Eigen::Matrix<double, 1, 1> yaw_measurement{detection.yaw()};
  const Eigen::Matrix<double, 1, 1> yaw_measure_noise_cov{
      noise_scale * kLidarYawNoiseVariance};
  Eigen::Matrix<double, 1, 2> yaw_measure_matrix{1, 0};
  yaw_kf_.Update<1>(yaw_measurement, yaw_measure_noise_cov, yaw_measure_matrix);
  yaw_ = yaw_kf_.state()[0];
  yaw_rate_ = yaw_kf_.state()[1];

  size_ = size_ * kSmoothFilterCoefficient +
          detection.size() * (1 - kSmoothFilterCoefficient);

  last_update_timestamp_ = detection.timestamp();
  ++associated_lidar_detections_cnt_;
}

bool Track::IsLost() const {
  const double kTrackLostTimeGating = IsConfirmed() ? 1.5 : 0.1;  // second
  return timestamp_ - last_update_timestamp_ > kTrackLostTimeGating;
}

bool Track::IsConfirmed() const { return associated_lidar_detections_cnt_ > 2; }

std::vector<Vec2d> Track::GetCorners() const {
  const double theta = -yaw_;
  const Vec2d center(position_.x(), position_.y());
  const Vec2d tl_shift(0.5 * size_.y(), 0.5 * size_.x());
  const Vec2d tr_shift(0.5 * size_.y(), -0.5 * size_.x());
  const Vec2d bl_shift(-0.5 * size_.y(), 0.5 * size_.x());
  const Vec2d br_shift(-0.5 * size_.y(), -0.5 * size_.x());
  const Vec2d top_left = center + RotateByTheta(tl_shift, theta);
  const Vec2d top_right = center + RotateByTheta(tr_shift, theta);
  const Vec2d bottom_left = center + RotateByTheta(bl_shift, theta);
  const Vec2d bottom_right = center + RotateByTheta(br_shift, theta);
  return std::vector<Vec2d>{top_left, bottom_left, bottom_right, top_right};
}