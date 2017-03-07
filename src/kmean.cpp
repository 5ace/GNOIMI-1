#include <algorithm>
#include <cstring>
#include <mkl.h>
#include <assert.h>

#include "kmean.h"

t_vv<float> normalize_features(const t_vv<float>& vv_features) {

    int D = vv_features[0].size();        /* dimensionality of the vectors */
    int N = vv_features.size();		  /* number of vectors */

    float* pt_org_feature = new float[D];
    float* pt_normalized_feature = new float[D];
    float* temp_f = new float[D];

    t_vv<float> vv_normalized_f;

    //normalize all features
    for (int i = 0; i < N; ++i) {
	memset(pt_normalized_feature, (float)0.0f, D * sizeof(float));
	memcpy(pt_org_feature, vv_features[i].data(), D * sizeof(float));

	//normalize the feature
	float cur_norm = cblas_snrm2(D, pt_org_feature, 1);
	cblas_saxpy(D, 1.0f/cur_norm, pt_org_feature, 1, pt_normalized_feature, 1);

	memcpy(temp_f, pt_normalized_feature, D * sizeof(float));
	vv_normalized_f.emplace_back(t_v<float>(pt_normalized_feature, pt_normalized_feature + D));
    }

    delete[] pt_org_feature;
    delete[] pt_normalized_feature;
    delete[] temp_f;

    return vv_normalized_f;
}

t_K_mean_info kmean_yael_wrapper(const t_vv<float>& vv_features, const int& K) {
    int D = vv_features[0].size();        /* dimensionality of the vectors */
    int N = vv_features.size();		  /* number of vectors */
    int nt = 4;                           /* number of threads to use */
    int niter = 0;                        /* number of iterations (0 for convergence)*/
    int redo = 1;                         /* number of redo */

    float * v = fvec_new(D * N);

    for (int i = 0; i < N; ++i) {
	memcpy(v + i * D, vv_features[i].data(), D * sizeof(float));
    }

    /* variables are allocated externaly */
    float * centroids = fvec_new (D * K); /* output: centroids */
    float * dis = fvec_new (N);           /* point-to-cluster distance */
    int * assign = ivec_new (N);          /* quantization index of each point */
    int * nassign = ivec_new (K);         /* output: number of vectors assigned to each centroid */

    double t1 = getmillisecs();
    kmeans (D, N, K, niter, v, 1, 1, redo, centroids, dis, assign, nassign);
    double t2 = getmillisecs();

    printf ("kmeans performed in %.3fs\n", (t2 - t1)  / 1000);

    t_K_mean_info res;	
    for (int i = 0; i < K; ++i) {
	t_v<float> v_temp;
	v_temp.assign(centroids + i * D, centroids + (i + 1) * D);
	res.vv_centroid.push_back(v_temp);
    }

    res.v_num_assign.assign(nassign, nassign + K);
    res.v_pt_assigned_centroid_idx.assign(assign, assign + N);
    res.v_pt_dis_to_assigned_centroid.assign(dis, dis + N);

    //ivec_print (nassign, K);
    //ivec_print (assign, n);
    //fvec_print (dis, n);

    free(v); 
    free(centroids); 
    free(dis); 
    free(assign); 
    free(nassign);

    return res;
}

/* --------------------------------------------------------------------------*/
/**
 * calculate all displacements between all feature points and their corresponding
 * nearest centroid
 *
 * @param[in] vv_features		    original feature vectors
 * @param[in] v_pt_assigned_centroid_idx    the corresponding nearest k-mean centroid idx
 * @param[in] vv_centroid		    all k-mean centroid
 *
 * @return				    vector of displacements 	 
 */
/* ----------------------------------------------------------------------------*/
t_vv<float> cal_displacement(const t_vv<float>& vv_features, const t_v<int>& v_pt_assigned_centroid_idx, const t_vv<float>& vv_centroid) {	
    t_vv<float> vv_displacement;

    const int f_num = vv_features.size(); 
    const int f_dim = vv_features[0].size();
    assert(f_num == v_pt_assigned_centroid_idx.size());

    for (int i = 0; i < f_num; ++i) {
	t_v<float> v_temp(f_dim);
	transform(vv_features[i].begin(), vv_features[i].end(), vv_centroid[v_pt_assigned_centroid_idx[i]].begin(), v_temp.begin(), std::minus<float>());	
	vv_displacement.emplace_back(v_temp);
    }

    return vv_displacement;
}
