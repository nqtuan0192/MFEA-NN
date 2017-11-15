#ifndef MNIST_HELPER_H
#define MNIST_HELPER_H

#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <string>
#include <sstream>

#include "utils.h"

#define TRAINING_DATA_FILE "dataset/nbit/training_input"
#define TESTING_DATA_FILE "dataset/nbit/training_output"

#define INPUT_SIZE		8
#define OUTPUT_SIZE		1

#define DATA_SIZE		256
//uint32_t(std::pow(2, INPUT_SIZE))
#define DATA_PERCENTAGE	100

#define TRAINING_SIZE	DATA_PERCENTAGE * DATA_SIZE / 100
#define TESTING_SIZE	TRAINING_SIZE

uint32_t numberOfSetBits(uint32_t i) {
     i = i - ((i >> 1) & 0x55555555);
     i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
     return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

template<typename TYPE> void doRandomShuffle(TYPE* input_data_ptr, TYPE* output_data_ptr, size_t numberof_samples, size_t numberof_elems_per_inputrow, size_t numberof_elems_per_outputrow) {
	TYPE* temp_ptr;
	cudaCALL(CUDA_M_MALLOC_MANAGED(temp_ptr, DATATYPE, numberof_elems_per_inputrow > numberof_elems_per_outputrow ? numberof_elems_per_inputrow : numberof_elems_per_outputrow));
	
	std::random_device datahelper_random_device;
	std::mt19937 datahelper_mt_engine(datahelper_random_device());
	std::uniform_int_distribution<> ui_dist(0, numberof_samples);
	for (uint32_t i = 0; i < numberof_samples; ++i) {
		uint32_t j = ui_dist(datahelper_mt_engine);
		std::cout << "swap " << i << " and " << j << std::endl;
		
		// swap input
		// copy i => temp
		cudaCALL(cudaMemcpy(temp_ptr, input_data_ptr + i * numberof_elems_per_inputrow, numberof_elems_per_inputrow * sizeof(DATATYPE), cudaMemcpyDeviceToDevice));
		// copy j => i
		cudaCALL(cudaMemcpy(input_data_ptr + i * numberof_elems_per_inputrow, input_data_ptr + j * numberof_elems_per_inputrow, numberof_elems_per_inputrow * sizeof(DATATYPE), cudaMemcpyDeviceToDevice));
		// copy temp => j
		cudaCALL(cudaMemcpy(input_data_ptr + j * numberof_elems_per_inputrow, temp_ptr, numberof_elems_per_inputrow * sizeof(DATATYPE), cudaMemcpyDeviceToDevice));
		
		
		// swap output
		// copy i => temp
		cudaCALL(cudaMemcpy(temp_ptr, output_data_ptr + i * numberof_elems_per_outputrow, numberof_elems_per_outputrow * sizeof(DATATYPE), cudaMemcpyDeviceToDevice));
		// copy j => i
		cudaCALL(cudaMemcpy(output_data_ptr + i * numberof_elems_per_outputrow, output_data_ptr + j * numberof_elems_per_outputrow, numberof_elems_per_outputrow * sizeof(DATATYPE), cudaMemcpyDeviceToDevice));
		// copy temp => j
		cudaCALL(cudaMemcpy(output_data_ptr + j * numberof_elems_per_outputrow, temp_ptr, numberof_elems_per_outputrow * sizeof(DATATYPE), cudaMemcpyDeviceToDevice));
	}
	cudaCALL(cudaFree(temp_ptr));
}


template<typename TYPE> void loadDataFile(TYPE*& training_input_data_ptr, TYPE*& training_output_data_ptr,
											TYPE*& testing_input_data_ptr, TYPE*& testing_output_data_ptr) {
	// allocate memory
	TYPE* data_input;
	TYPE* data_output;
	cudaCALL(CUDA_M_MALLOC_MANAGED(data_input, TYPE, TRAINING_SIZE * INPUT_SIZE));
	cudaCALL(CUDA_M_MALLOC_MANAGED(data_output, TYPE, TRAINING_SIZE * OUTPUT_SIZE));
	
	for (uint32_t i = 0; i < DATA_SIZE; ++i) {
		for (uint32_t j = 0; j < INPUT_SIZE; ++j) {
			data_input[i * INPUT_SIZE + j] = (i >> j) & 0x1;
			std::cout << data_input[i * INPUT_SIZE + j] << ", ";
		}std::cout << std::endl;
		fprintTruebitOrder(stdout, &i, 1);
		std::cout << std::endl;
		data_output[i * OUTPUT_SIZE] = numberOfSetBits(i) % 2;
		BUG(data_output[i * OUTPUT_SIZE]);
	}
	
	// random shuffle
	//doRandomShuffle<TYPE>(data_input, data_output, DATA_SIZE, INPUT_SIZE, OUTPUT_SIZE);
	
	// asign return pointers
	training_input_data_ptr = data_input;
	testing_input_data_ptr = data_input;//data_input + TRAINING_SIZE * INPUT_SIZE;
	
	training_output_data_ptr = data_output;
	testing_output_data_ptr = data_output;//data_output + TRAINING_SIZE * OUTPUT_SIZE;
	
	// log datafile
	std::ofstream ofile("_____log.txt");
	for (uint32_t i = 0; i < TRAINING_SIZE; ++i) {
		for (uint32_t j = 0; j < INPUT_SIZE; ++j) {
			ofile << training_input_data_ptr[i * INPUT_SIZE + j] << ",";
		}
		ofile << training_output_data_ptr[i * OUTPUT_SIZE];
		ofile << std::endl;
	}
	for (uint32_t i = 0; i < TESTING_SIZE; ++i) {
		for (uint32_t j = 0; j < INPUT_SIZE; ++j) {
			ofile << testing_input_data_ptr[i * INPUT_SIZE + j] << ",";
		}
		ofile << testing_output_data_ptr[i * OUTPUT_SIZE];
		ofile << std::endl;
	}
	ofile.close();
}

/*
template<typename TYPE, size_t nrow, size_t ncol> void readMatrixFromFile(std::string file_name, TYPE* data_ptr) {
	std::ifstream ifile(file_name);
	#pragma unroll
	for (uint32_t i = 0; i < nrow; ++i) {
		#pragma unroll
		for (uint32_t j = 0; j < ncol; ++j) {
			ifile >> data_ptr[i * ncol + j];
		}
	}
	ifile.close();
}

template<typename TYPE> void loadDataFile(TYPE*& training_input_data_ptr, TYPE*& training_output_data_ptr,
											TYPE*& testing_input_data_ptr, TYPE*& testing_output_data_ptr) {
	// allocate memory
	TYPE* data_input;
	TYPE* data_output;
	cudaCALL(CUDA_M_MALLOC_MANAGED(data_input, TYPE, TRAINING_SIZE * INPUT_SIZE));
	cudaCALL(CUDA_M_MALLOC_MANAGED(data_output, TYPE, TRAINING_SIZE * OUTPUT_SIZE));
	
	readMatrixFromFile<TYPE, TRAINING_SIZE, INPUT_SIZE>(TRAINING_DATA_FILE, data_input);
	readMatrixFromFile<TYPE, TRAINING_SIZE, OUTPUT_SIZE>(TRAINING_DATA_FILE, data_output);
	
	// asign return pointers
	training_input_data_ptr = data_input;
	testing_input_data_ptr = data_input;//data_input + TRAINING_SIZE * INPUT_SIZE;
	
	training_output_data_ptr = data_output;
	testing_output_data_ptr = data_output;//data_output + TRAINING_SIZE * OUTPUT_SIZE;
}
*/
#endif	// MNIST_HELPER_H
