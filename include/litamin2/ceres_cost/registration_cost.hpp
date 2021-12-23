#ifndef REGISTRATION_COST_HPP
#define REGISTRATION_COST_HPP

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

struct REGISTRATION_FACTOR_GICP
{
    REGISTRATION_FACTOR_GICP(Eigen::Vector3d p_mean, Eigen::Vector3d q_mean, Eigen::Matrix3d p_cov, Eigen::Matrix3d q_cov,
                           Eigen::Matrix3d lambdaI,double sigma_ICP) :
                        p_mean_(p_mean), q_mean_(q_mean), p_cov_(p_cov), q_cov_(q_cov),lambdaI_(lambdaI),sigma_ICP_(sigma_ICP) {}

    template <typename T>
    bool operator()(const T *const q, const T *const t, T *residuals) const
    {
        Eigen::Map<Eigen::Matrix<T,3,1>> residuals_map(residuals);
        Eigen::Matrix<T, 3, 1> p_m{T(p_mean_.x()), T(p_mean_.y()), T(p_mean_.z())};
        Eigen::Matrix<T, 3, 1> q_m{T(q_mean_.x()), T(q_mean_.y()), T(q_mean_.z())};
        Eigen::Matrix<T, 3, 3> p_c = p_cov_.cast<T>();
        Eigen::Matrix<T, 3, 3> q_c = q_cov_.cast<T>();

        Eigen::Matrix<T, 3, 3> lambI = lambdaI_.cast<T>();       
        Eigen::Quaternion<T> quat{q[3], q[0], q[1], q[2]};
        Eigen::Matrix<T, 3, 1> translation{t[0], t[1], t[2]};
        Eigen::Matrix<T,3,3> rotation;
        rotation = quat.matrix();

        Eigen::Matrix<T, 3, 3> mahalanobis = (q_c + rotation * p_c * (rotation.transpose())).inverse();
        // mahalanobis.normalize();
        //  llt分解被验证为是一种错误的方式
        // Eigen::Matrix<T,3,3> LT = mahalanobis.llt().matrixL().transpose();
        residuals_map = mahalanobis * (q_m - (rotation * p_m + translation));
        // T EICP = residuals_map.squaredNorm();
        // T sigma_square = T(sigma_ICP_)*T(sigma_ICP_);
        // residuals_map = sqrt((T(1.)-(EICP/(EICP+sigma_square))))*residuals_map;

        return true;
    }

    static ceres::CostFunction *Create(Eigen::Vector3d p_mean_, Eigen::Vector3d q_mean_, Eigen::Matrix3d p_cov_, 
    Eigen::Matrix3d q_cov_,Eigen::Matrix3d lambdaI_, double sigma_ICP_){
        // 残差是三维的,变量分别是思维和三维
        return (new ceres::AutoDiffCostFunction<REGISTRATION_FACTOR_GICP,3,4,3>
        (new REGISTRATION_FACTOR_GICP(p_mean_,q_mean_,p_cov_,q_cov_,lambdaI_,sigma_ICP_)));
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector3d p_mean_, q_mean_;
    Eigen::Matrix3d p_cov_, q_cov_;
    Eigen::Matrix3d lambdaI_;
    double sigma_ICP_;
};
#endif