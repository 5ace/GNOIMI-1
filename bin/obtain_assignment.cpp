#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <functional>
#include <ctime>
#include <chrono>
#include <mutex>
#include <map>

#include <cblas.h>
#include <boost/algorithm/string.hpp>

#include "file_utility.h"

#ifdef __cplusplus
extern "C"{
#endif

#include <yael/kmeans.h>
#include <yael/vector.h>
#include <yael/matrix.c>

int fvecs_read(const char *fname, int d, int n, float *v);
void fmat_mul_full(const float *left, const float *right,
	int m, int n, int k,
	const char *transp,
	float *result);
#ifdef __cplusplus
}
#endif

using namespace std;

//int D = 512;
//int K = 1289;
//int totalLearnCount = 500000;
//int L = 6;
//string pointFilename = "./normalized_features_500k.fvecs";
//string labelFilename = "./label_500K.txt"; 
//string learnedCoarseFilename = "./learned_coarse_19.fvecs";
//string learnedFineFilename = "./learned_fine_19.fvecs";
//string learnedAlphaFilename = "./learned_alpha_19.fvecs";
//int threadsCount = 20;
//int trainThreadChunkSize = 5000;

const int D = 512;
const int K = 657;
const int totalLearnCount = 260000;
const int L = 6;
const int first_R_fineCentroid = 4;
string pointFilename = "./normalized_features_260k.fvecs";
string labelFilename = "./label_260K.txt"; 
string learnedCoarseFilename = "./learned_260K_coarse_19.fvecs";
string learnedFineFilename = "./learned_260K_fine_19.fvecs";
string learnedAlphaFilename = "./learned_260K_alpha_19.fvecs";

int threadsCount = 3;
int totalPoints_perThread = 0;
const int Init_chunkSize = 3000; 

float* alpha = (float*)malloc(K * K * sizeof(float));

float* coarseVocab = (float*)malloc(D * K * sizeof(float));
float* fineVocab = (float*)malloc(D * K * sizeof(float));

float* coarseNorms = (float*)malloc(K * sizeof(float));
float* fineNorms = (float*)malloc(K * sizeof(float));
float* coarseFineProducts = (float*)malloc(K * K * sizeof(float));

vector<int> v_ID;

//map<coarse_ID * K + fine_ID>, vector<person IDs assigned to the centroid defined by coarseID and fine_ID> 
map<int, vector<int>> m_inverted_idx;

//map<coarse_ID * K + fine_ID>, map<person IDs assigned to the centroid defined by coarseID and fine_ID, number of the corresponding ID> 
map<int, map<int, int>> mm_inverted_idx;

vector<std::pair<float, int> > scores(L * K);

mutex mtx;

///////////////////////////


vector<float> normalize_features(const vector<float>& v_feature) {

    int D = v_feature.size();        /* dimensionality of the vectors */

    float* pt_org_feature = new float[D];
    float* pt_normalized_feature = new float[D];
    float* temp_f = new float[D];

    vector<float> v_normalized_f;

    //normalize all features
    memset(pt_normalized_feature, (float)0.0f, D * sizeof(float));
    memcpy(pt_org_feature, v_feature.data(), D * sizeof(float));

    //normalize the feature
    float cur_norm = cblas_snrm2(D, pt_org_feature, 1);
    cblas_saxpy(D, 1.0f/cur_norm, pt_org_feature, 1, pt_normalized_feature, 1);

    memcpy(temp_f, pt_normalized_feature, D * sizeof(float));
    v_normalized_f = vector<float>(pt_normalized_feature, pt_normalized_feature + D);

    delete[] pt_org_feature;
    delete[] pt_normalized_feature;
    delete[] temp_f;

    return v_normalized_f;
}

struct threadInfo {
    long long startId;
    int pointsCount;
    int chunksCount;
    int trainThreadChunkSize;
    int extra_ChunkSize;
};

threadInfo get_threadInfo(const int& threadId) {
    threadInfo t_info;
    t_info.startId = totalPoints_perThread * threadId; 
    t_info.trainThreadChunkSize = Init_chunkSize; 

    if (totalPoints_perThread * (threadId + 1) <= totalLearnCount) {
	t_info.pointsCount = totalPoints_perThread;	
    } else {
	t_info.pointsCount = totalLearnCount - totalPoints_perThread * threadId;
    }

    t_info.extra_ChunkSize = 0;
    if (t_info.pointsCount % t_info.trainThreadChunkSize == 0) {
	t_info.chunksCount = t_info.pointsCount / t_info.trainThreadChunkSize;
    } else {
	t_info.extra_ChunkSize = t_info.pointsCount % t_info.trainThreadChunkSize;
	t_info.chunksCount = (t_info.pointsCount - t_info.extra_ChunkSize) / t_info.trainThreadChunkSize + 1;
    }	

    return t_info;
}

void computeOptimalAssignsSubset(int threadId) {

    threadInfo cur_t_info = get_threadInfo(threadId);
    int trainThreadChunkSize = cur_t_info.trainThreadChunkSize;

    FILE* learnStream = fopen(pointFilename.c_str(), "r");
    fseek(learnStream, cur_t_info.startId * (D + 1) * sizeof(float), SEEK_SET);

    float* pointsCoarseTerms = (float*)malloc(trainThreadChunkSize * K * sizeof(float));
    float* pointsFineTerms = (float*)malloc(trainThreadChunkSize * K * sizeof(float));
    float* chunkPoints = (float*)malloc(trainThreadChunkSize * D * sizeof(float));

    std::vector<std::pair<float, int> > coarseScores(K);

    for(int chunkId = 0; chunkId < cur_t_info.chunksCount; ++chunkId) {
	std::cout << "[Assigns][Thread " << threadId << "] " << "processing chunk " <<  chunkId << " of " << cur_t_info.chunksCount << "\n";

	if (chunkId == cur_t_info.chunksCount - 1 && cur_t_info.extra_ChunkSize != 0) {
	    free(pointsCoarseTerms);
	    free(pointsFineTerms);
	    free(chunkPoints); 

	    trainThreadChunkSize = cur_t_info.extra_ChunkSize;
	    pointsCoarseTerms = (float*)malloc(trainThreadChunkSize * K * sizeof(float));
	    pointsFineTerms = (float*)malloc(trainThreadChunkSize * K * sizeof(float));
	    chunkPoints = (float*)malloc(trainThreadChunkSize * D * sizeof(float));
	}

	fvecs_fread(learnStream, chunkPoints, trainThreadChunkSize, D);
	fmat_mul_full(coarseVocab, chunkPoints, K, trainThreadChunkSize, D, "TN", pointsCoarseTerms);
	fmat_mul_full(fineVocab, chunkPoints, K, trainThreadChunkSize, D, "TN", pointsFineTerms);

	for(int pointId = 0; pointId < trainThreadChunkSize; ++pointId) {
	    cblas_saxpy(K, -1.0, coarseNorms, 1, pointsCoarseTerms + pointId * K, 1);

	    for(int k = 0; k < K; ++k) {
		coarseScores[k].first = (-1.0) * pointsCoarseTerms[pointId * K + k];
		coarseScores[k].second = k;
	    }

	    std::sort(coarseScores.begin(), coarseScores.end());
	    float currentMinScore = 999999999.0;
	    int currentMinCoarseId = -1;
	    int currentMinFineId = -1;

	    for(int l = 0; l < L; ++l) {
		//examine cluster l
		int currentCoarseId = coarseScores[l].second;
		float currentCoarseTerm = coarseScores[l].first;
		for(int currentFineId = 0; currentFineId < K; ++currentFineId) {
		    int centroid_ID = currentCoarseId * K + currentFineId;

		    float alphaFactor = alpha[centroid_ID];
		    float score = currentCoarseTerm + alphaFactor * coarseFineProducts[centroid_ID] + 
			(-1.0) * alphaFactor * pointsFineTerms[pointId * K + currentFineId] + 
			alphaFactor * alphaFactor * fineNorms[currentFineId];

		    if(score < currentMinScore) {
			currentMinScore = score;
			currentMinCoarseId = currentCoarseId;
			currentMinFineId = currentFineId;
		    }
		}
	    }

	    mtx.lock();
	    //vector version
	    m_inverted_idx[currentMinCoarseId * K + currentMinFineId].emplace_back(v_ID[cur_t_info.startId + chunkId * cur_t_info.trainThreadChunkSize + pointId]);

	    //map version
	    mm_inverted_idx[currentMinCoarseId * K + currentMinFineId][v_ID[cur_t_info.startId + chunkId * cur_t_info.trainThreadChunkSize + pointId]]++;
	    mtx.unlock();
	}
    }

    fclose(learnStream);
    free(chunkPoints);
    free(pointsCoarseTerms);
    free(pointsFineTerms);
}

vector<vector<int>> query_feature(const vector<float>& v_q_feature, int& query_result) {

    float* pointsCoarseTerms = (float*)malloc(K * sizeof(float));
    float* pointsFineTerms = (float*)malloc(K * sizeof(float));

    const float* queryPoint = v_q_feature.data();
    std::vector<std::pair<float, int> > coarseScores(K);

    fmat_mul_full(coarseVocab, queryPoint, K, 1, D, "TN", pointsCoarseTerms);
    fmat_mul_full(fineVocab, queryPoint, K, 1, D, "TN", pointsFineTerms);

    cblas_saxpy(K, -1.0, coarseNorms, 1, pointsCoarseTerms, 1);
    for(int k = 0; k < K; ++k) {
	coarseScores[k].first = (-1.0) * pointsCoarseTerms[k];
	coarseScores[k].second = k;
    }
    std::sort(coarseScores.begin(), coarseScores.end());

    float currentMinScore = 999999999.0;
    int currentMinCoarseId = -1;
    int currentMinFineId = -1;

    for(int l = 0; l < L; ++l) {
	//examine cluster l
	int currentCoarseId = coarseScores[l].second;
	float currentCoarseTerm = coarseScores[l].first;

	for(int currentFineId = 0; currentFineId < K; ++currentFineId) {

	    int centroid_ID = currentCoarseId * K + currentFineId;
	    float alphaFactor = alpha[centroid_ID];
	    scores[l * K + currentFineId].first = currentCoarseTerm + alphaFactor * coarseFineProducts[centroid_ID] + 
		(-1.0) * alphaFactor * pointsFineTerms[currentFineId] + 
		alphaFactor * alphaFactor * fineNorms[currentFineId];
	    scores[l * K + currentFineId].second = centroid_ID;
	}
    }

    std::nth_element(scores.begin(), scores.begin() + first_R_fineCentroid, scores.end());
    std::sort(scores.begin(), scores.begin() + first_R_fineCentroid);

    vector<vector<int>> vv_matched_IDs;
    for (int i = 0; i < first_R_fineCentroid; ++i) {
	vv_matched_IDs.push_back(m_inverted_idx[scores[i].second]);
	auto m_it = std::max_element(std::begin(mm_inverted_idx[scores[i].second]), std::end(mm_inverted_idx[scores[i].second]), 
	       [](pair<int, int> m_it1, pair<int, int> m_it2){
		    return m_it1.second < m_it2.second; 
	       });
	if (i == 0) query_result = m_it->first;
	else        query_result = m_it->second > query_result ? m_it->first : query_result;
    }

    free(pointsCoarseTerms);
    free(pointsFineTerms);

    return vv_matched_IDs; 
}

int main() {
    // load all required data
    fvecs_read(learnedCoarseFilename.c_str(), D, K, coarseVocab);
    fvecs_read(learnedFineFilename.c_str(), D, K, fineVocab);
    fvecs_read(learnedAlphaFilename.c_str(), K, K, alpha);
    load_ID_fromFile(labelFilename, v_ID);

    //precomputation
    for(int k = 0; k < K; ++k) {
	coarseNorms[k] = cblas_sdot(D, coarseVocab + k * D, 1, coarseVocab + k * D, 1) / 2;
	fineNorms[k] = cblas_sdot(D, fineVocab + k * D, 1, fineVocab + k * D, 1) / 2;
    }

    fmat_mul_full(fineVocab, coarseVocab, K, K, D, "TN", coarseFineProducts);

    totalPoints_perThread = totalLearnCount / threadsCount; 

    //increase the threadCount
    if (totalLearnCount % threadsCount != 0) {
       	threadsCount += 1;
    }

    // obtain Assigns
    vector<std::thread> workers;
    for(int threadId = 0; threadId < threadsCount; ++threadId) {
	workers.push_back(std::thread(computeOptimalAssignsSubset, threadId));
    }
    for(int threadId = 0; threadId < threadsCount; ++threadId) {
	workers[threadId].join();
    }
    workers.clear();

    //string q_f_name = "12200_feature.bin";
    string q_f_name = "lfw_122.bin";
    //string q_f_name = "lfw_118.bin";
    //string q_f_name = "1289.bin";
    //string q_f_name = "657.bin";
    
    vector<float> v_q_feature = load_feature<float>(q_f_name);
    vector<float> v_q_nf = normalize_features(v_q_feature);

    vector<vector<int>> vv_matched_IDs;
    int query_res;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); 

    vv_matched_IDs = query_feature(v_q_nf, query_res);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    for (int i = 0; i < vv_matched_IDs.size(); ++i) {
	cout << "rank " << i << ": " << endl;
	for_each(std::begin(vv_matched_IDs[i]), std::end(vv_matched_IDs[i]), [](const int& id){cout << id << " ";});
	cout << endl;
    }

    cout << "query_result: " << query_res << endl; 

    free(alpha);
    free(coarseVocab);
    free(fineVocab);
    free(coarseNorms);
    free(fineNorms);
    free(coarseFineProducts);
    return 0;
}
