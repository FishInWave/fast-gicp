#ifndef GICP_COST_HPP
#define GICP_COST_HPP

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
// L^T*error
struct GICP_FACTOR
{
    GICP_FACTOR(Eigen::Vector3d p_mean, Eigen::Vector3d q_mean, Eigen::Matrix3d p_cov, Eigen::Matrix3d q_cov) : p_mean_(p_mean), q_mean_(q_mean), p_cov_(p_cov), q_cov_(q_cov) {}

    template <typename T>
    bool operator()(const T *const q, const T *const t, T *residuals) const
    {
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_map(residuals);
        Eigen::Matrix<T, 3, 1> p_m{T(p_mean_.x()), T(p_mean_.y()), T(p_mean_.z())};
        Eigen::Matrix<T, 3, 1> q_m{T(q_mean_.x()), T(q_mean_.y()), T(q_mean_.z())};
        Eigen::Matrix<T, 3, 3> p_c = p_cov_.cast<T>();
        Eigen::Matrix<T, 3, 3> q_c = q_cov_.cast<T>();

        Eigen::Quaternion<T> quat{q[3], q[0], q[1], q[2]};
        Eigen::Matrix<T, 3, 1> translation{t[0], t[1], t[2]};
        Eigen::Matrix<T, 3, 3> rotation(quat.toRotationMatrix());

        Eigen::Matrix<T, 3, 3> mahalanobis = (q_c + rotation * p_c * (rotation.transpose())).inverse();
        Eigen::Matrix<T, 3, 3> LT = mahalanobis.llt().matrixL().transpose();
        residuals_map = LT * (q_m - (rotation * p_m + translation));

        return true;
    }

    static ceres::CostFunction *Create(Eigen::Vector3d p_mean_, Eigen::Vector3d q_mean_, Eigen::Matrix3d p_cov_,
                                       Eigen::Matrix3d q_cov_)
    {
        // 分别是残差，q，t的维度
        return (new ceres::AutoDiffCostFunction<GICP_FACTOR, 3, 4, 3>(new GICP_FACTOR(p_mean_, q_mean_, p_cov_, q_cov_)));
    }

    Eigen::Vector3d p_mean_, q_mean_;
    Eigen::Matrix3d p_cov_, q_cov_;
};
#endif