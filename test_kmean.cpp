#include <iostream>
#include <functional>
#include <ctime>
#include <chrono>

#include "file_utility.h"
#include "kmean.h"

using namespace std;

//struct t_K_mean_info {
	//t_v<int> v_num_assign;
	//t_v<int> v_pt_assigned_centroid_idx;
	//t_v<float> v_pt_dis_to_assigned_centroid;
	//t_vv<float> vv_centroid;
//}

int main(int argc, char** argv) {
	if (argc != 6) {
		std::cerr << "Usage: " << argv[0]
			<< " [feature file list] [cluster# (i.e., K of Kmean)] [coarse_kmean file name (i.e., S)] [fine_kmean file name (i.e., T)] [normalized feature file name]" << std::endl;
		return 1;
	}

	int pos = 1;
	string f_file_list   = argv[pos++];
	int K = atoi(argv[pos++]);
	string coarseFile_name = argv[pos++];
	string fineFile_name = argv[pos++];
	string normalized_f_name = argv[pos++];

	t_vv<float> vv_features;
	t_v<int> v_ID;

	load_feature_ID_fromFiles<float>(f_file_list, v_ID, vv_features);

	//write out the IDs
	ofstream of;
	of.open("label.txt");
	for (auto it = begin(v_ID); it != end(v_ID); ++it) {
		of << *it << endl;	
	}
	of.close();

	t_vv<float> vv_normalized_features = normalize_features(vv_features);
	
	const int D = vv_features[0].size();

	cout << "Estimate coarse kmean" << endl;
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); 
	
	//estimate coarse kmean
	t_K_mean_info coarse_kmean_res = kmean_yael_wrapper(vv_normalized_features, K);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Coarse Kmean time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

	cout << "calculate displacement" << endl;
	t_vv<float> vv_displacements = cal_displacement(vv_normalized_features, coarse_kmean_res.v_pt_assigned_centroid_idx, coarse_kmean_res.vv_centroid);


	cout << "Estimate fine kmean" << endl;
	std::chrono::steady_clock::time_point fine_begin = std::chrono::steady_clock::now(); 
	
	//estimate fine kmean i.e., displacement
	t_K_mean_info fine_kmean_res = kmean_yael_wrapper(vv_displacements, K);

	std::chrono::steady_clock::time_point fine_end = std::chrono::steady_clock::now();
	std::cout << "Fine Kmean time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(fine_end - fine_begin).count() << std::endl;


	//store the coarse centroids
	vector<float> flatten_centroid;
	for (auto vvit = std::begin(coarse_kmean_res.vv_centroid); vvit != std::end(coarse_kmean_res.vv_centroid); ++vvit) {
		for (auto vit = vvit->begin(); vit != vvit->end(); ++vit) {
			flatten_centroid.emplace_back(*vit);
		}
	}	
	float* vv_coarse_centroid_ptr = flatten_centroid.data();
	fvecs_write(coarseFile_name.c_str(), D, K, vv_coarse_centroid_ptr);

	//store the fine centroids
	flatten_centroid.clear();
	for (auto vvit = std::begin(fine_kmean_res.vv_centroid); vvit != std::end(fine_kmean_res.vv_centroid); ++vvit) {
		for (auto vit = vvit->begin(); vit != vvit->end(); ++vit) {
			flatten_centroid.emplace_back(*vit);
		}
	}	
	float* vv_fine_centroid_ptr = flatten_centroid.data();
	fvecs_write(fineFile_name.c_str(), D, K, vv_fine_centroid_ptr);

	//change normalized feature to fvecs
	flatten_centroid.clear();
	for (auto vvit = std::begin(vv_normalized_features); vvit != std::end(vv_normalized_features); ++vvit) {
		for (auto vit = vvit->begin(); vit != vvit->end(); ++vit) {
			flatten_centroid.emplace_back(*vit);
		}
	}	
	float* vv_original_features_ptr = flatten_centroid.data();
	string original_feature_file_name = normalized_f_name;
	fvecs_write(original_feature_file_name.c_str(), D, vv_features.size(), vv_original_features_ptr);

	//cout << kmean_res.v_pt_assigned_centroid_idx.size() << endl;
	//for_each(kmean_res.v_pt_assigned_centroid_idx.begin(), kmean_res.v_pt_assigned_centroid_idx.end(), [](const int& i){cout << i << " ";});
	//cout << endl;
	
}
