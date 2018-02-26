#include <iostream>
#include <memory>
#include <fstream>
#include <cstdint>
#include <typeinfo>
#include <vector>
#include <thread>
#include <algorithm>
#include <cmath>
#include <limits>
#include <array>

#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include "utils.h"
#include "Matrix.hpp"


//#include "MnistHelper.hpp"
//#include "DatasetIonosphereHelper.hpp"
//#include "DatasetTictactoeHelper.hpp"
#include "NbitHelper.hpp"



#include "MFEAChromosome.hpp"
#include "MFEA.hpp"
#include "MFEATask.hpp"


// variables for training data
DATATYPE* training_input_data;
DATATYPE* training_output_data;

// variables for testing data
DATATYPE* testing_input_data;
DATATYPE* testing_output_data;


void showMUltitasksSetting() {
	std::cout << "Multitasks setting:" << std::endl;;
    for (uint32_t task = 0; task < TASK_SIZE; ++task) {
		std::cout << "--- Task " << task << " has " << getNumberofLayersbyTask(task) << " layers (including " << getNumberofLayersbyTask(task) - 1 << " hidden layers)" << std::endl;
		for (uint32_t layer = 1; layer < getNumberofLayersbyTask(task) + 1; ++layer) {
			if (layer == getNumberofLayersbyTask(task)) {
				std::cout << "----- Layer " << layer << " (output layer): " << getNumberofUnitsbyTaskLayer(task, layer) << " units" << std::endl;
			} else {
				std::cout << "----- Layer " << layer << ": " << getNumberofUnitsbyTaskLayer(task, layer) << " units" << std::endl;
			}
			std::cout << "------- Data offset    = " << std::get<OFFSET_IDX>(getLayerWeightsandBiasesbyTaskLayer(task, layer))
							 << "\t Data size    = " << std::get<SIZE_IDX>(getLayerWeightsandBiasesbyTaskLayer(task, layer)) << std::endl;
			std::cout << "------- Weights offset = " << std::get<OFFSET_IDX>(getLayerWeightsbyTaskLayer(task, layer))
							 << "\t Weights size = " << std::get<SIZE_IDX>(getLayerWeightsbyTaskLayer(task, layer)) << std::endl;
			std::cout << "------- Biases offset  = " << std::get<OFFSET_IDX>(getLayerBiasesbyTaskLayer(task, layer))
							 << "\t Biases size  = " << std::get<SIZE_IDX>(getLayerBiasesbyTaskLayer(task, layer)) << std::endl;
		}
	}
}

void testDecode();
void testSBX();
void testPMU();
void testEval();


int main(int argc, char** argv) {
	/* manually set device for running */
	int device_id;
	if (argc > 1) {
		device_id = atoi(argv[1]);
		
	} else {
		device_id = 0;
	}
	cudaSetDevice(device_id);

	
	/* load input data */
	loadDataFile<DATATYPE>(training_input_data, training_output_data, testing_input_data, testing_output_data);


	/* Total CPU Page faults: 1384 for float */
	/* Total CPU Page faults: 2477 for double */
	
	{// limit scope for object destruct before destroy CUDA environment
		MFEA<90, 1000, 2> mfea(training_input_data, training_output_data,
							testing_input_data, testing_output_data,
							device_id);
		if (mfea.init_libraries() != 0) {
			return EXIT_FAILURE;
		}
		
		mfea.initialize();	// does not cause page fault
		mfea.evolution();	// does not cause page fault
		mfea.sumariseResults();
		mfea.writeSumaryResults();
		mfea.reEvaluateTheFinalPopulation();
		
		
		/*for (uint32_t i = 0; i < 200; ++i) {
			float __cf_distributionindex		= 1.0 * (std::rand() % 11);			// randomize between 0 - 10
			float __mf_randommatingprobability	= 1.0;
			float __mf_polynomialmutationindex	= 1.0 * (std::rand() % 11);			// randomize between 0 - 10
			float __mf_mutationratio			= 0.05 * (1 + std::rand() % 10);	// randomize between 5% - 50%
			mfea.setTunableFactors(__cf_distributionindex,
										__mf_randommatingprobability,
										__mf_polynomialmutationindex,
										__mf_mutationratio	);
			
			mfea.initialize();
			mfea.evolution();
			mfea.sumariseResults();
			mfea.writeSumaryResults();
		}*/
		

		mfea.finalize_libraries();
	}
	
	showMUltitasksSetting();

	
    /* Reset CUDA evironment */
    cudaDeviceReset();
    
	return 0;
}

/*int main(int argc, char** argv) {
	// manually set device for running
	int device_id;
	if (argc > 1) {
		device_id = atoi(argv[1]);
		
	} else {
		device_id = 0;
	}
	cudaSetDevice(device_id);

	cublasHandle_t cublas_handle;
	cublasCALL(cublasCreate(&cublas_handle));
	
	cudnnHandle_t cudnn_handle;
	cudnnCALL(cudnnCreate(&cudnn_handle));

	curandGenerator_t curand_prng;
	// Create a pseudo-random number generator
	curandCreateGenerator(&curand_prng, CURAND_RNG_PSEUDO_MTGP32);
	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(curand_prng, 0);

	
	// load input data
	loadDataFile<DATATYPE>(training_input_data, training_output_data, testing_input_data, testing_output_data);



	testEval();

	
	
	showMUltitasksSetting();

	
    // Reset CUDA evironment
    cudaDeviceReset();
    
	return 0;
}*/

void testDecode() {
	cublasHandle_t cublas_handle;
	cublasCALL(cublasCreate(&cublas_handle));
	
	cudnnHandle_t cudnn_handle;
	cudnnCALL(cudnnCreate(&cudnn_handle));

	curandGenerator_t curand_prng;
	// Create a pseudo-random number generator
	curandCreateGenerator(&curand_prng, CURAND_RNG_PSEUDO_MTGP32);
	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(curand_prng, 0);

	std::array<MFEA_Chromosome, 4> population;
	thrust::for_each(population.begin(), population.end(), MFEA_Chromosome_Randomize(curand_prng));

	cudaDeviceSynchronize();
	std::cout << population[0];


	for (uint32_t i = 0; i < getTotalLayerWeightsandBiases(); ++i) {
		population[0].rnvec[i] = i;
	}
	printMatrix<DATATYPE>(1, getTotalLayerWeightsandBiases(), population[0].rnvec);
	//cublas_transposeMatrix<DATATYPE>(6, 8, population[0].rnvec, population[1].rnvec, cublas_handle);

	DATATYPE* W;
	CUDA_M_MALLOC_MANAGED(W, DATATYPE, getTotalLayerWeightsandBiases());

	for (uint32_t task = 0; task < TASK_SIZE; ++task) {
		for (uint32_t layer = 1; layer <= getNumberofLayersbyTask(task); ++layer) {
			std::tuple<uint32_t, uint32_t> shape = population[0].decode(population[0].rnvec, W, task, layer, cublas_handle);
			cudaDeviceSynchronize();
			std::cout << "W for task " << task << " layer " << layer << " : " << std::endl;
			printMatrix<DATATYPE>(std::get<MATRIX_NROW>(shape), std::get<MATRIX_NCOL>(shape), W);


			std::tuple<uint32_t, uint32_t> bias = getLayerBiasesbyTaskLayer(task, layer);
			std::cout << "b for task " << task << " layer " << layer << " : " << std::endl;
			printMatrix<DATATYPE>(1, std::get<SIZE_IDX>(bias), population[0].rnvec + std::get<OFFSET_IDX>(bias));
		}
	}
}

void testSBX() {
	curandGenerator_t curand_prng;
	// Create a pseudo-random number generator
	curandCreateGenerator(&curand_prng, CURAND_RNG_PSEUDO_MTGP32);
	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(curand_prng, 0);
	
	std::array<MFEA_Chromosome, 4> population;
	thrust::for_each(population.begin(), population.end(), MFEA_Chromosome_Randomize(curand_prng));


	DATATYPE* ct_beta;
	cudaCALL(CUDA_M_MALLOC_MANAGED(ct_beta, DATATYPE, getTotalLayerWeightsandBiases()));



	cudaDeviceSynchronize();
	BUG(getTotalLayerWeightsandBiases());
	for (uint32_t i = 0; i < getTotalLayerWeightsandBiases(); ++i) {
		population[0].rnvec[i] = double(i) / getTotalLayerWeightsandBiases();
		population[1].rnvec[i] = double(i) / getTotalLayerWeightsandBiases();
	}

	test_crossover(population[0], population[1],
						  population[2], population[3],
						  5, ct_beta,
						  curand_prng);

	examineCrossover(population[0], population[1], population[2], population[3]);

	cudaDeviceSynchronize();
	
	std::cout << population[0];
	std::cout << population[1];
	std::cout << population[2];
	std::cout << population[3];
}

void testPMU() {
	curandGenerator_t curand_prng;
	// Create a pseudo-random number generator
	curandCreateGenerator(&curand_prng, CURAND_RNG_PSEUDO_MTGP32);
	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(curand_prng, 0);
	
	std::array<MFEA_Chromosome, 4> population;
	thrust::for_each(population.begin(), population.end(), MFEA_Chromosome_Randomize(curand_prng));


	DATATYPE* ct_beta;
	DATATYPE* rp;
	cudaCALL(CUDA_M_MALLOC_MANAGED(ct_beta, DATATYPE, getTotalLayerWeightsandBiases()));
	cudaCALL(CUDA_M_MALLOC_MANAGED(rp, DATATYPE, getTotalLayerWeightsandBiases()));


	cudaDeviceSynchronize();
	BUG(getTotalLayerWeightsandBiases());
	for (uint32_t i = 0; i < getTotalLayerWeightsandBiases(); ++i) {
		population[0].rnvec[i] = double(i) / getTotalLayerWeightsandBiases();
	}

	test_mutate(population[0], population[1],
						5, 1,
						ct_beta, rp, curand_prng);

	cudaDeviceSynchronize();
	std::cout << population[1];
}

void testEval() {
	cublasHandle_t cublas_handle;
	cublasCALL(cublasCreate(&cublas_handle));
	
	cudnnHandle_t cudnn_handle;
	cudnnCALL(cudnnCreate(&cudnn_handle));

	curandGenerator_t curand_prng;
	// Create a pseudo-random number generator
	curandCreateGenerator(&curand_prng, CURAND_RNG_PSEUDO_MTGP32);
	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(curand_prng, 0);
	
	std::array<MFEA_Chromosome, 4> population;
	thrust::for_each(population.begin(), population.end(), MFEA_Chromosome_Randomize(curand_prng));



	DATATYPE* dev_mat_temp_rnvec;
	DATATYPE* dev_mat_temp_w;
	DATATYPE* dev_mat_ones;
	std::array<DATATYPE*, LAYER_SIZE + 1> dev_mat_temp_layers;
	cudaCALL(CUDA_M_MALLOC_MANAGED(dev_mat_temp_rnvec, DATATYPE, getTotalLayerWeightsandBiases()));
	cudaCALL(CUDA_M_MALLOC_MANAGED(dev_mat_temp_w, DATATYPE, getTotalLayerWeightsandBiases()));
	cudaCALL(CUDA_M_MALLOC_MANAGED(dev_mat_ones, DATATYPE, TRAINING_SIZE));
	cuda_fillMatrix<DATATYPE>(TRAINING_SIZE, 1, dev_mat_ones, 1.0f);
	
	for (uint32_t i = 0; i < LAYER_SIZE + 1; ++i) {
		cudaCALL(CUDA_M_MALLOC_MANAGED(dev_mat_temp_layers[i], DATATYPE, TRAINING_SIZE * getNumberofUnitsbyTaskLayer(TASKINDEX_LARGEST, i)));
	}


	cudaDeviceSynchronize();
	BUG(getTotalLayerWeightsandBiases());
	for (uint32_t i = 0; i < getTotalLayerWeightsandBiases(); ++i) {
		population[0].rnvec[i] = double(i) / getTotalLayerWeightsandBiases();
	}

	printGPUArray(dev_mat_ones, TRAINING_SIZE);
	printGPUArray(dev_mat_temp_rnvec, getTotalLayerWeightsandBiases());


	int i = 0;
	//for (uint32_t i = 0; i < 1; ++i) {
		population[i].skill_factor = i % TASK_SIZE;
		population[i].evalObj(TRAINING_SIZE, OUTPUT_SIZE,
								training_input_data,
								training_output_data,
								dev_mat_temp_rnvec,
								dev_mat_temp_w,
								dev_mat_ones,
								dev_mat_temp_layers,
								cublas_handle, cudnn_handle,
								true);
	//}

	printGPUArray(dev_mat_temp_layers[LAYER_SIZE - 1], TRAINING_SIZE * getNumberofUnitsbyTaskLayer(TASKINDEX_LARGEST, LAYER_SIZE));

	cudaDeviceSynchronize();
	std::cout << population[0];
}