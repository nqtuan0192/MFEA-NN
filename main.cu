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
#include "DatasetIonosphereHelper.hpp"
//#include "DatasetTictactoeHelper.hpp"
//#include "NbitHelper.hpp"



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
