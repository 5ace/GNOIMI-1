#ifndef NOVUGNO_IMI_INCLUDE_KMEAN_H_
#define NOVUGNO_IMI_INCLUDE_KMEAN_H_

#include <stdio.h>

#include "common_type.h"

#ifdef __cplusplus
extern "C"{
#endif

#include <yael/vector.h>
#include <yael/kmeans.h>
#include <yael/machinedeps.h>

float* fvec_new_rand(long);
float* fvec_new(long);
int* ivec_new(long);
double getmillisecs();
float kmeans(int, int, int, int, const float*, int, long, int, float*, float*, int*, int*);
void ivec_print(int const*, int);

#ifdef __cplusplus
}
#endif

/* --------------------------------------------------------------------------*/
/**
 * K mean info
 */
/* ----------------------------------------------------------------------------*/
struct t_K_mean_info {
	t_v<int> v_num_assign;			    //!< number of assigned points for each centroid
	t_v<int> v_pt_assigned_centroid_idx;        //!< each point's assigned centroid's idx
	t_v<float> v_pt_dis_to_assigned_centroid;   //!< the distance between a point and its assigned centorid
	t_vv<float> vv_centroid;		    //!< all estimate K mean centroids
};


/* --------------------------------------------------------------------------*/
/**
 * Normalizing the features
 *
 * @param[in] vv_features	all feature points
 * 
 * @return			normalized features   
 */
/* ----------------------------------------------------------------------------*/
t_vv<float> normalize_features(const t_vv<float>& vv_features);

/* --------------------------------------------------------------------------*/
/**
 * estimate the K-mean by using yael lib
 *
 * @param[in] vv_features		all feature points
 * @param[in] K				the number of the centroid
 *
 * @return				t_K_mean_info
 */
/* ----------------------------------------------------------------------------*/
t_K_mean_info kmean_yael_wrapper(const t_vv<float>& vv_features, const int& K);

/* --------------------------------------------------------------------------*/
/**
 * calculate the displacement bewteen a point and its corresponding assigend 
 * centroid
 *
 * @param[in] vv_features			original feature points
 * @param[in] v_pt_assigned_centroid_idx	assigned centroid idx
 * @param[in] vv_centroid			centroid points
 *
 * @return					the displacements  
 */
/* ----------------------------------------------------------------------------*/
t_vv<float> cal_displacement(const t_vv<float>& vv_features, const t_v<int>& v_pt_assigned_centroid_idx, const t_vv<float>& vv_centroid);	

#endif 
