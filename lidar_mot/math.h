#pragma once

#include <cmath>
#include <eigen3/Eigen/Dense>
#include <vector>

class Vec2d {
 public:
  Vec2d() : x_(0.0), y_(0.0) {}
  Vec2d(double x, double y) : x_(x), y_(y) {}

  double x() const { return x_; }
  double y() const { return y_; }

  double Length() const { return std::sqrt(x_ * x_ + y_ * y_); }
  Vec2d operator+(const Vec2d& b) const {
    return Vec2d(x_ + b.x(), y_ + b.y());
  }
  Vec2d operator-(const Vec2d& b) const {
    return Vec2d(x_ - b.x(), y_ - b.y());
  }
  Vec2d operator*(const double k) const { return Vec2d(k * x_, k * y_); }

 private:
  double x_;
  double y_;
};

class Vec3d {
 public:
  Vec3d() : x_(0.0), y_(0.0), z_(0.0) {}
  Vec3d(double x, double y, double z) : x_(x), y_(y), z_(z) {}
  Vec3d(const Eigen::Vector3d& p) : x_(p[0]), y_(p[1]), z_(p[2]) {}

  double x() const { return x_; }
  double y() const { return y_; }
  double z() const { return z_; }

  Vec2d xy() const { return Vec2d(x_, y_); }

  double Length() const { return std::sqrt(x_ * x_ + y_ * y_ + z_ * z_); }
  Vec3d operator+(const Vec3d& b) const {
    return Vec3d(x_ + b.x(), y_ + b.y(), z_ + b.z());
  }
  Vec3d operator-(const Vec3d& b) const {
    return Vec3d(x_ - b.x(), y_ - b.y(), z_ - b.z());
  }
  Vec3d operator*(const double k) const {
    return Vec3d(k * x_, k * y_, k * z_);
  }

 private:
  double x_;
  double y_;
  double z_;
};

class Transformation2d {
 public:
  Transformation2d() = default;
  ~Transformation2d() = default;

  Transformation2d(const Vec2d translation, const double theta)
      : translation_(translation), theta_(theta) {}

  Vec2d translation() const { return translation_; }
  double theta() const { return theta_; }

  Vec2d Transform(const Vec2d point) const {
    const Vec2d p = point - translation_;
    return Vec2d(p.x() * std::cos(theta_) + p.y() * std::sin(theta_),
                 p.y() * std::cos(theta_) - p.x() * std::sin(theta_));
  }

 private:
  Vec2d translation_;
  double theta_;
};
