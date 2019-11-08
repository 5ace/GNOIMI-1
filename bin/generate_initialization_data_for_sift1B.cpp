#include <iostream>
#include <functional>
#include <ctime>
#include <chrono>

#include "common_type.h"
#include "file_utility.h"
#include "kmean.h"

using namespace std;

/* --------------------------------------------------------------------------*/
/**
 * This program is for generating all the required data for learn_GNOIMI 
 * 
 * feature file list:	    input original features (with format feature.bin label)
 * K                :	    the K-mean's K
 * coarse kmean file name:  the estimated K-mean centroids and stored as fvecs format
 * fine kmean file name:    the fine K-mean centroids and stored as fvecs format 
 *  
 */
/* ----------------------------------------------------------------------------*/
int main(int argc, char** argv) {
    if (argc != 5) {
	    std::cerr << "Usage: " << argv[0]
	    << " [feature file list] [cluster# (i.e., K of Kmean)] [coarsefile .fvecs] [finefile .fvecs]" << std::endl;
	    exit(1);
    }

    int pos = 1;
    string f_file_list   = argv[pos++];
    int K = atoi(argv[pos++]);
    string coarse_file   = argv[pos++];
    string fine_file   = argv[pos++];

    t_vv<float> vv_features;
    t_v<int> v_ID;

    int D = 0;
    //load_feature_ID_fromFiles<float>(f_file_list, v_ID, vv_features);
    if(f_file_list.find(".bvecs") == string::npos) {
      std::cout << "this must input bvecs\n";
      exit(1);
    }
    float *vecs = nullptr; 
    int num = b2fvecs_new_read_limit(f_file_list.c_str(),&D,&vecs,100000);
    for(int i = 0;i<num && i<100000;i++) {
      t_v<float> t;
      for(int j=0;j<D;j++) {
        t.push_back(vecs[i*D+j]); 
      }
      vv_features.push_back(t);
    }
    cout << "read " << num << " end\n";
    t_vv<float> vv_normalized_features = normalize_features(vv_features);
    print_elements(vv_normalized_features[0].data(),D);
    print_elements(vv_normalized_features[1].data(),D);
    print_elements(vv_normalized_features[2].data(),D);


    cout << "Estimate coarse kmean" << endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); 

    //estimate coarse kmean
    t_K_mean_info coarse_kmean_res = kmean_yael_wrapper(vv_normalized_features, K);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Coarse Kmean time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
    print_elements(coarse_kmean_res.vv_centroid[0].data(),D);

    cout << "calculate displacement" << endl;
    t_vv<float> vv_displacements = cal_displacement(vv_normalized_features, coarse_kmean_res.v_pt_assigned_centroid_idx, coarse_kmean_res.vv_centroid);
    print_elements(coarse_kmean_res.v_pt_assigned_centroid_idx.data(),100);

    cout << "Estimate fine kmean" << endl;
    std::chrono::steady_clock::time_point fine_begin = std::chrono::steady_clock::now(); 

    //estimate fine kmean i.e., displacement
    t_K_mean_info fine_kmean_res = kmean_yael_wrapper(vv_displacements, K);

    std::chrono::steady_clock::time_point fine_end = std::chrono::steady_clock::now();
    std::cout << "Fine Kmean time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(fine_end - fine_begin).count() << std::endl;
    print_elements(fine_kmean_res.vv_centroid[0].data(),D);

    //store the coarse centroids (i.e., S)
    vector<float> flatten_centroid;
    for (auto vvit = std::begin(coarse_kmean_res.vv_centroid); vvit != std::end(coarse_kmean_res.vv_centroid); ++vvit) {
      for (auto vit = vvit->begin(); vit != vvit->end(); ++vit) {
        flatten_centroid.emplace_back(*vit);
      }
    } 
    float* vv_coarse_centroid_ptr = flatten_centroid.data();
    fvecs_write(coarse_file.c_str(),D,K,vv_coarse_centroid_ptr);
    //store the fine centroids (i.e., T)
    flatten_centroid.clear();
    for (auto vvit = std::begin(fine_kmean_res.vv_centroid); vvit != std::end(fine_kmean_res.vv_centroid); ++vvit) {
      for (auto vit = vvit->begin(); vit != vvit->end(); ++vit) {
        flatten_centroid.emplace_back(*vit);
      }
    } 
    float* vv_fine_centroid_ptr = flatten_centroid.data();
    fvecs_write(fine_file.c_str(),D,K,vv_fine_centroid_ptr);
}
