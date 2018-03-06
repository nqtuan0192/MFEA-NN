#ifndef MNIST_HELPER_H
#define MNIST_HELPER_H

#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <string>
#include <sstream>

#include "utils.h"


#define TRAINING_DATA_FILE "dataset/tic-tac-toe/tic-tac-toe.data"
#define TESTING_DATA_FILE "dataset/tic-tac-toe/tic-tac-toe.data"

#define INPUT_SIZE		9
#define OUTPUT_SIZE		1

#define DATA_SIZE		958
#define DATA_PERCENTAGE	70

#define TRAINING_SIZE	DATA_PERCENTAGE * DATA_SIZE / 100
#define TESTING_SIZE	DATA_SIZE - TRAINING_SIZE




template<typename TYPE> void loadDataFile(TYPE*& training_input_data_ptr, TYPE*& training_output_data_ptr,
											TYPE*& testing_input_data_ptr, TYPE*& testing_output_data_ptr) {
	// allocate memory
	TYPE* data_input;
	TYPE* data_output;
	cudaCALL(CUDA_M_MALLOC_MANAGED(data_input, TYPE, TRAINING_SIZE * INPUT_SIZE));
	cudaCALL(CUDA_M_MALLOC_MANAGED(data_output, TYPE, TRAINING_SIZE * OUTPUT_SIZE));
	
	// reading and asigning
	std::ifstream input(TRAINING_DATA_FILE);
	
	std::string line;
	std::string cell;
	for (uint32_t i = 0; i < DATA_SIZE; ++i) {
		std::getline(input, line);
		std::stringstream lineStream(line);
		uint32_t j = 0;
		while (std::getline(lineStream, cell, ',')) {
			if (cell == "x") {
				data_input[i * INPUT_SIZE + j] = 1;
			} else if (cell == "o") {
				data_input[i * INPUT_SIZE + j] = -1;
			} else if (cell == "b") {
				data_input[i * INPUT_SIZE + j] = 0;
			} else if (cell == "positive") {
				data_output[i * OUTPUT_SIZE] = 1;
			} else if (cell == "negative") {
				data_output[i * OUTPUT_SIZE] = 0;
			}
			++j;
		}
		// This checks for a trailing comma with no data after it.
		if (!lineStream && cell.empty()) {
			// If there was a trailing comma then add an empty element.
			// do nothing
		}
	}

	// asign return pointers
	training_input_data_ptr = data_input;
	testing_input_data_ptr = data_input + TRAINING_SIZE * INPUT_SIZE;
	
	training_output_data_ptr = data_output;
	testing_output_data_ptr = data_output + TRAINING_SIZE * OUTPUT_SIZE;
	
	input.close();
	
	
	// log datafile
	std::ofstream ofile("_____log.txt");
	for (uint32_t i = 0; i < TRAINING_SIZE; ++i) {
		for (uint32_t j = 0; j < INPUT_SIZE; ++j) {
			if (training_input_data_ptr[i * INPUT_SIZE + j] == -1) {
				ofile << "o" << ",";
			} else if (training_input_data_ptr[i * INPUT_SIZE + j] == 1) {
				ofile << "x" << ",";
			} else if (training_input_data_ptr[i * INPUT_SIZE + j] == 0) {
				ofile << "b" << ",";
			}
		}
		if (training_output_data_ptr[i * OUTPUT_SIZE] == 1) {
			ofile << "positive";
		} else {
			ofile << "negative";
		}
		ofile << std::endl;
	}
	for (uint32_t i = 0; i < TESTING_SIZE; ++i) {
		for (uint32_t j = 0; j < INPUT_SIZE; ++j) {
			if (testing_input_data_ptr[i * INPUT_SIZE + j] == -1) {
				ofile << "o" << ",";
			} else if (testing_input_data_ptr[i * INPUT_SIZE + j] == 1) {
				ofile << "x" << ",";
			} else if (testing_input_data_ptr[i * INPUT_SIZE + j] == 0) {
				ofile << "b" << ",";
			}
		}
		if (testing_output_data_ptr[i * OUTPUT_SIZE] == 1) {
			ofile << "positive";
		} else {
			ofile << "negative";
		}
		ofile << std::endl;
	}
	ofile.close();
}



#endif	// MNIST_HELPER_H
