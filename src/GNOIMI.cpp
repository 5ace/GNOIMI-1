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
#include <vector>
#include <ctime>
#include <chrono>

#include <mutex>
#include <cblas.h>

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

int D = 512;
int K = 657;
int totalLearnCount = 260000;
int learnIterationsCount = 20;
int L = 6;

string learnFilename = "./normalized_features_260k.fvecs";
string initCoarseFilename = "./coarse_260k.fvecs";
string initFineFilename = "./fine_260k.fvecs";
string outputFilesPrefix = "./260K_new_";

int threadsCount = 3;
int totalPoints_perThread = 0;
const int Init_chunkSize = 3000; 

int* coarseAssigns = (int*)malloc(totalLearnCount * sizeof(int));
int* fineAssigns = (int*)malloc(totalLearnCount * sizeof(int));
float* alphaNum = (float*)malloc(K * K * sizeof(float));
float* alphaDen = (float*)malloc(K * K * sizeof(float));
float* alpha = (float*)malloc(K * K * sizeof(float));

vector<float*> alphaNumerators(threadsCount);
vector<float*> alphaDenominators(threadsCount);

float* coarseVocab = (float*)malloc(D * K * sizeof(float));
float* fineVocab = (float*)malloc(D * K * sizeof(float));
float* fineVocabNum = (float*)malloc(D * K * sizeof(float));
float* fineVocabDen = (float*)malloc(K * sizeof(float));
float* coarseVocabNum = (float*)malloc(D * K * sizeof(float));
float* coarseVocabDen = (float*)malloc(K * sizeof(float));

vector<float*> fineVocabNumerators(threadsCount);
vector<float*> fineVocabDenominators(threadsCount);
vector<float*> coarseVocabNumerators(threadsCount);
vector<float*> coarseVocabDenominators(threadsCount);

float* coarseNorms = (float*)malloc(K * sizeof(float));
float* fineNorms = (float*)malloc(K * sizeof(float));
float* coarseFineProducts = (float*)malloc(K * K * sizeof(float));

float* errors = (float*)malloc(threadsCount * sizeof(float));
mutex mtx;

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

    errors[threadId] = 0.0;

    FILE* learnStream = fopen(learnFilename.c_str(), "r");
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
		    float alphaFactor = alpha[currentCoarseId * K + currentFineId];
		    float score = currentCoarseTerm + alphaFactor * coarseFineProducts[currentCoarseId * K + currentFineId] + 
			(-1.0) * alphaFactor * pointsFineTerms[pointId * K + currentFineId] + 
			alphaFactor * alphaFactor * fineNorms[currentFineId];

		    if(score < currentMinScore) {
			currentMinScore = score;
			currentMinCoarseId = currentCoarseId;
			currentMinFineId = currentFineId;
		    }
		}
	    }

	    coarseAssigns[cur_t_info.startId + chunkId * cur_t_info.trainThreadChunkSize + pointId] = currentMinCoarseId;
	    fineAssigns[cur_t_info.startId + chunkId * cur_t_info.trainThreadChunkSize + pointId] = currentMinFineId;

	    errors[threadId] += currentMinScore * 2 + 1.0; // point has a norm equals 1.0
	}
    }

    fclose(learnStream);
    free(chunkPoints);
    free(pointsCoarseTerms);
    free(pointsFineTerms);
}

void computeOptimalAlphaSubset(int threadId) {
    memset(alphaNumerators[threadId], 0, K * K * sizeof(float));
    memset(alphaDenominators[threadId], 0, K * K * sizeof(float));

    threadInfo cur_t_info = get_threadInfo(threadId);
    int trainThreadChunkSize = cur_t_info.trainThreadChunkSize;

    FILE* learnStream = fopen(learnFilename.c_str(), "r");
    fseek(learnStream, cur_t_info.startId * (D + 1) * sizeof(float), SEEK_SET);

    float* residual = (float*)malloc(D * sizeof(float));
    float* chunkPoints = (float*)malloc(trainThreadChunkSize * D * sizeof(float));

    for(int chunkId = 0; chunkId < cur_t_info.chunksCount; ++chunkId) {
	std::cout << "[Alpha][Thread " << threadId << "] " << "processing chunk " <<  chunkId << " of " << cur_t_info.chunksCount << "\n";

	if (chunkId == cur_t_info.chunksCount - 1 && cur_t_info.extra_ChunkSize != 0) {
	    free(chunkPoints); 

	    trainThreadChunkSize = cur_t_info.extra_ChunkSize;
	    chunkPoints = (float*)malloc(trainThreadChunkSize * D * sizeof(float));
	}

	fvecs_fread(learnStream, chunkPoints, trainThreadChunkSize, D);
	for(int pointId = 0; pointId < trainThreadChunkSize; ++pointId) {
	    //S_k
	    int coarseAssign = coarseAssigns[cur_t_info.startId + chunkId * cur_t_info.trainThreadChunkSize + pointId];
	    //T_l
	    int fineAssign = fineAssigns[cur_t_info.startId + chunkId * cur_t_info.trainThreadChunkSize + pointId];

	    memcpy(residual, chunkPoints + pointId * D, D * sizeof(float));
	    //P_i - S_k
	    cblas_saxpy(D, -1.0, coarseVocab + coarseAssign * D, 1, residual, 1);

	    //<P_i - S_k, T_l>
	    alphaNumerators[threadId][coarseAssign * K + fineAssign] += 
		cblas_sdot(D, residual, 1, fineVocab + fineAssign * D, 1);

	    //||T||^2
	    alphaDenominators[threadId][coarseAssign * K + fineAssign] += fineNorms[fineAssign] * 2; // we keep halves of norms 
	}
    }
    fclose(learnStream);
    free(chunkPoints);
    free(residual);
}

void computeOptimalFineVocabSubset(int threadId) {
    memset(fineVocabNumerators[threadId], 0, K * D * sizeof(float));
    memset(fineVocabDenominators[threadId], 0, K * sizeof(float));

    threadInfo cur_t_info = get_threadInfo(threadId);
    int trainThreadChunkSize = cur_t_info.trainThreadChunkSize;

    FILE* learnStream = fopen(learnFilename.c_str(), "r");
    fseek(learnStream, cur_t_info.startId * (D + 1) * sizeof(float), SEEK_SET);

    float* residual = (float*)malloc(D * sizeof(float));
    float* chunkPoints = (float*)malloc(trainThreadChunkSize * D * sizeof(float));

    for(int chunkId = 0; chunkId < cur_t_info.chunksCount; ++chunkId) {
	//std::cout << "[Fine vocabs][Thread " << threadId << "] " << "processing chunk " <<  chunkId << " of " << chunksCount << "\n";
	
	if (chunkId == cur_t_info.chunksCount - 1 && cur_t_info.extra_ChunkSize != 0) {
	    free(chunkPoints); 

	    trainThreadChunkSize = cur_t_info.extra_ChunkSize;
	    chunkPoints = (float*)malloc(trainThreadChunkSize * D * sizeof(float));
	}

	fvecs_fread(learnStream, chunkPoints, trainThreadChunkSize, D);
	for(int pointId = 0; pointId < trainThreadChunkSize; ++pointId) {

	    int coarseAssign = coarseAssigns[cur_t_info.startId + chunkId * cur_t_info.trainThreadChunkSize + pointId];
	    int fineAssign = fineAssigns[cur_t_info.startId + chunkId * cur_t_info.trainThreadChunkSize + pointId];

	    float alphaFactor = alpha[coarseAssign * K + fineAssign];
	    memcpy(residual, chunkPoints + pointId * D, D * sizeof(float));

	    cblas_saxpy(D, -1.0, coarseVocab + coarseAssign * D, 1, residual, 1);
	    cblas_saxpy(D, alphaFactor, residual, 1, fineVocabNumerators[threadId] + fineAssign * D, 1);
	    fineVocabDenominators[threadId][fineAssign] += alphaFactor * alphaFactor;
	}
    }

    fclose(learnStream);
    free(chunkPoints);
    free(residual);
}

void computeOptimalCoarseVocabSubset(int threadId) {
    memset(coarseVocabNumerators[threadId], 0, K * D * sizeof(float));
    memset(coarseVocabDenominators[threadId], 0, K * sizeof(float));

    threadInfo cur_t_info = get_threadInfo(threadId);
    int trainThreadChunkSize = cur_t_info.trainThreadChunkSize;

    FILE* learnStream = fopen(learnFilename.c_str(), "r");
    fseek(learnStream, cur_t_info.startId * (D + 1) * sizeof(float), SEEK_SET);
    
    float* residual = (float*)malloc(D * sizeof(float));
    float* chunkPoints = (float*)malloc(trainThreadChunkSize * D * sizeof(float));

    for(int chunkId = 0; chunkId < cur_t_info.chunksCount; ++chunkId) {
	//std::cout << "[Coarse vocabs][Thread " << threadId << "] " << "processing chunk " <<  chunkId << " of " << chunksCount << "\n";
	
	if (chunkId == cur_t_info.chunksCount - 1 && cur_t_info.extra_ChunkSize != 0) {
	    free(chunkPoints); 

	    trainThreadChunkSize = cur_t_info.extra_ChunkSize;
	    chunkPoints = (float*)malloc(trainThreadChunkSize * D * sizeof(float));
	}

	fvecs_fread(learnStream, chunkPoints, trainThreadChunkSize, D);
	for(int pointId = 0; pointId < trainThreadChunkSize; ++pointId) {
	    int coarseAssign = coarseAssigns[cur_t_info.startId + chunkId * cur_t_info.trainThreadChunkSize + pointId];
	    int fineAssign = fineAssigns[cur_t_info.startId + chunkId * cur_t_info.trainThreadChunkSize + pointId];
	    float alphaFactor = alpha[coarseAssign * K + fineAssign];
	    memcpy(residual, chunkPoints + pointId * D, D * sizeof(float));
	    cblas_saxpy(D, -1.0 * alphaFactor, fineVocab + fineAssign * D, 1, residual, 1);
	    cblas_saxpy(D, 1, residual, 1, coarseVocabNumerators[threadId] + coarseAssign * D, 1);
	    coarseVocabDenominators[threadId][coarseAssign] += 1.0;
	}
    }

    fclose(learnStream);
    free(chunkPoints);
    free(residual);
}

int main() {
    totalPoints_perThread = totalLearnCount / threadsCount; 

    //increase the threadCount
    if (totalLearnCount % threadsCount != 0) {
       	threadsCount += 1;
	alphaNumerators.resize(threadsCount);
	alphaDenominators.resize(threadsCount);
	fineVocabNumerators.resize(threadsCount);
	fineVocabDenominators.resize(threadsCount);
	coarseVocabNumerators.resize(threadsCount);
	coarseVocabDenominators.resize(threadsCount);
	free(errors);
	errors = (float*)malloc(threadsCount * sizeof(float));
    }

    for(int threadId = 0; threadId < threadsCount; ++threadId) {
	alphaNumerators[threadId] = (float*)malloc(K * K * sizeof(float*));
	alphaDenominators[threadId] = (float*)malloc(K * K * sizeof(float*));
    }

    for(int threadId = 0; threadId < threadsCount; ++threadId) {
	fineVocabNumerators[threadId] = (float*)malloc(K * D * sizeof(float*));
	fineVocabDenominators[threadId] = (float*)malloc(K * sizeof(float*));
    }  

    for(int threadId = 0; threadId < threadsCount; ++threadId) {
	coarseVocabNumerators[threadId] = (float*)malloc(K * D * sizeof(float));
	coarseVocabDenominators[threadId] = (float*)malloc(K * sizeof(float));
    }

    // init vocabs
    fvecs_read(initCoarseFilename.c_str(), D, K, coarseVocab);
    fvecs_read(initFineFilename.c_str(), D, K, fineVocab);

    // init alpha
    for(int i = 0; i < K * K; ++i) {
	alpha[i] = 1.0f;
    }

    // learn iterations
    std::cout << "Start learning iterations...\n";
    for(int it = 0; it < learnIterationsCount; ++it) {
	cout << "learning iteration " << it << endl;
	for(int k = 0; k < K; ++k) {
	    coarseNorms[k] = cblas_sdot(D, coarseVocab + k * D, 1, coarseVocab + k * D, 1) / 2;
	    fineNorms[k] = cblas_sdot(D, fineVocab + k * D, 1, fineVocab + k * D, 1) / 2;
	}

	fmat_mul_full(fineVocab, coarseVocab, K, K, D, "TN", coarseFineProducts);

	// update Assigns
	vector<std::thread> workers;
	memset(errors, 0, threadsCount * sizeof(float));
	for(int threadId = 0; threadId < threadsCount; ++threadId) {
	    workers.push_back(std::thread(computeOptimalAssignsSubset, threadId));
	}

	for(int threadId = 0; threadId < threadsCount; ++threadId) {
	    workers[threadId].join();
	}

	float totalError = 0.0;
	for(int threadId = 0; threadId < threadsCount; ++threadId) {
	    cout << "threadId " << threadId << ": err " << errors[threadId] << endl;
	    totalError += errors[threadId];
	}
	std::cout << "Current reconstruction error... " << totalError / totalLearnCount << "\n";

	// update alpha
	workers.clear();
	for(int threadId = 0; threadId < threadsCount; ++threadId) {
	    workers.push_back(std::thread(computeOptimalAlphaSubset, threadId));
	}
	for(int threadId = 0; threadId < threadsCount; ++threadId) {
	    workers[threadId].join();
	}

	// update fine Vocabs
	workers.clear();
	memset(alphaNum, 0, K * K * sizeof(float));
	memset(alphaDen, 0, K * K * sizeof(float));
	
	for(int threadId = 0; threadId < threadsCount; ++threadId) {
	    cblas_saxpy(K * K, 1, alphaNumerators[threadId], 1, alphaNum, 1);
	    cblas_saxpy(K * K, 1, alphaDenominators[threadId], 1, alphaDen, 1);
	}

	for(int i = 0; i < K * K; ++i) {
	    alpha[i] = (alphaDen[i] == 0) ? 1.0 : alphaNum[i] / alphaDen[i];
	}

	for(int threadId = 0; threadId < threadsCount; ++threadId) {
	    workers.push_back(std::thread(computeOptimalFineVocabSubset, threadId));
	}

	for(int threadId = 0; threadId < threadsCount; ++threadId) {
	    workers[threadId].join();
	}

	// update coarse Vocabs
	workers.clear();
	memset(fineVocabNum, 0, K * D * sizeof(float));
	memset(fineVocabDen, 0, K * sizeof(float));

	for(int threadId = 0; threadId < threadsCount; ++threadId) {
	    cblas_saxpy(K * D, 1, fineVocabNumerators[threadId], 1, fineVocabNum, 1);
	    cblas_saxpy(K, 1, fineVocabDenominators[threadId], 1, fineVocabDen, 1);
	}

	for(int i = 0; i < K * D; ++i) {
	    fineVocab[i] = (fineVocabDen[i / D] == 0) ? 0 : fineVocabNum[i] / fineVocabDen[i / D];
	}

	for(int threadId = 0; threadId < threadsCount; ++threadId) {
	    workers.push_back(std::thread(computeOptimalCoarseVocabSubset, threadId));
	}

	for(int threadId = 0; threadId < threadsCount; ++threadId) {
	    workers[threadId].join();
	}

	workers.clear();
	memset(coarseVocabNum, 0, K * D * sizeof(float));
	memset(coarseVocabDen, 0, K * sizeof(float));

	for(int threadId = 0; threadId < threadsCount; ++threadId) {
	    cblas_saxpy(K * D, 1, coarseVocabNumerators[threadId], 1, coarseVocabNum, 1);
	    cblas_saxpy(K, 1, coarseVocabDenominators[threadId], 1, coarseVocabDen, 1);
	}

	for(int i = 0; i < K * D; ++i) {
	    coarseVocab[i] = (coarseVocabDen[i / D] == 0) ? 0 : coarseVocabNum[i] / coarseVocabDen[i / D];
	}

	// save current alpha and vocabs
	std::stringstream alphaFilename;
	alphaFilename << outputFilesPrefix << "alpha_" << it << ".fvecs";
	fvecs_write(alphaFilename.str().c_str(), K, K, alpha);
	
	std::stringstream fineVocabFilename;
	fineVocabFilename << outputFilesPrefix << "fine_" << it << ".fvecs";
	fvecs_write(fineVocabFilename.str().c_str(), D, K, fineVocab);
	
	std::stringstream coarseVocabFilename;
	coarseVocabFilename << outputFilesPrefix << "coarse_" << it << ".fvecs";
	fvecs_write(coarseVocabFilename.str().c_str(), D, K, coarseVocab);

	cout << "iteration " << it << " finished!" << endl;
    }

    free(coarseAssigns);
    free(fineAssigns);
    free(alphaNum);
    free(alphaDen);
    free(alpha);
    free(coarseVocab);
    free(coarseVocabNum);
    free(coarseVocabDen);
    free(fineVocab);
    free(fineVocabNum);
    free(fineVocabDen);
    free(coarseNorms);
    free(fineNorms);
    free(coarseFineProducts);
    free(errors);
    return 0;
}
