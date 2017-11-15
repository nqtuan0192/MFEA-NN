#ifndef MNIST_HELPER_H
#define MNIST_HELPER_H

#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <endian.h> 

#include "utils.h"


#define TRAINING_IMAGES_FILE "mnist/train-images-idx3-ubyte"
#define TRAINING_LABELS_FILE "mnist/train-labels-idx1-ubyte"
#define TESTING_IMAGES_FILE "mnist/t10k-images-idx3-ubyte"
#define TESTING_LABELS_FILE "mnist/t10k-labels-idx1-ubyte"

#define TRAINING_DATA	true
#define TESTING_DATA	false

#define IMAGE_WIDTH		28
#define IMAGE_HEIGHT	28
#define IMAGE_SIZE		((IMAGE_WIDTH) * (IMAGE_HEIGHT))

#define INPUT_SIZE		IMAGE_SIZE
#define OUTPUT_SIZE		10

#define TRAINING_SIZE	60000
#define TEST_SIZE		10000


uint8_t* readImagesFile(const char* filename) {
	uint8_t* ret;
	cudaCALL(CUDA_M_MALLOC_MANAGED(ret, uint8_t, TRAINING_SIZE * IMAGE_SIZE));
	
	std::ifstream input(filename, std::ifstream::binary);
	
	int32_t magic_number;
	input.read((char*)&magic_number, sizeof(magic_number));
	magic_number = be32toh(magic_number);
	//BUG(magic_number);

	int32_t numberof_images;
	input.read((char*)&numberof_images, sizeof(numberof_images));
	numberof_images = be32toh(numberof_images);
	//BUG(numberof_images);

	int32_t numberof_rows;
	input.read((char*)&numberof_rows, sizeof(numberof_rows));
	numberof_rows = be32toh(numberof_rows);
	//BUG(numberof_rows);

	int32_t numberof_columns;
	input.read((char*)&numberof_columns, sizeof(numberof_columns));
	numberof_columns = be32toh(numberof_columns);
	//BUG(numberof_columns);
	
	input.read((char*)ret, numberof_images * numberof_rows * numberof_columns);

	input.close();
	
	return ret;
}

uint8_t* readLabelsFile(const char* filename) {
	uint8_t* ret;
	cudaCALL(CUDA_M_MALLOC_MANAGED(ret, uint8_t, TRAINING_SIZE));
	
	std::ifstream input(filename, std::ios::binary);
	
	int32_t magic_number;
	input.read((char*)&magic_number, sizeof(magic_number));
	magic_number = be32toh(magic_number);
	//BUG(magic_number);
	
	int32_t numberof_items;
	input.read((char*)&numberof_items, sizeof(numberof_items));
	numberof_items = be32toh(numberof_items);
	//BUG(numberof_items);
	
	input.read((char*)ret, numberof_items);
	
	input.close();
	
	return ret;
}

template<typename TYPE> void visualizeImage(TYPE* image_data, uint8_t* label_data, int index, TYPE* label_code_data = nullptr, TYPE* predicted_label_code = nullptr) {
	// visualize
	TYPE* p = image_data + index * IMAGE_SIZE;
	for (int i = 0; i < IMAGE_WIDTH; ++i) {
		for (int j = 0; j < IMAGE_HEIGHT; ++j) {
			if (p[i * IMAGE_WIDTH + j] == 0) {
				std::cout << " ";
			} else {
				std::cout << "*";
			}
		}
		std::cout << std::endl;
	}
	
	// print label
	std::cout << "Label = " << label_data[index] + 0 << std::endl;
	
	// print code if provided
	if (label_code_data != nullptr) {
		std::cout << "Code  = [";
		for (int i = 0; i < OUTPUT_SIZE; ++i) {
			std::cout << label_code_data[index * OUTPUT_SIZE + i] << ", ";
		}
		std::cout << "]" << std::endl;
	}
	
	// print predicted label if provided
	if (predicted_label_code != nullptr) {
		int max_index = 0;
		std::cout << "Predicted  = [";
		for (int i = 0; i < OUTPUT_SIZE; ++i) {
			if (predicted_label_code[index * OUTPUT_SIZE + i] > predicted_label_code[index * OUTPUT_SIZE + max_index]) {
				max_index = i;
			}
			std::cout << predicted_label_code[index * OUTPUT_SIZE + i] << ", ";
		}
		std::cout << "] label = " << max_index << std::endl;
	}
}

// training_indicator = true : load training files
// training_indicator = false : load testing files
template<typename TYPE> void loadMnistData(uint8_t*& images_data_ptr, uint8_t*& labels_data_ptr,
					TYPE*& input_data_ptr, TYPE*& output_data_ptr,
					bool training_indicator = true) {
	// read data files
	if (training_indicator) {
		images_data_ptr = readImagesFile(TRAINING_IMAGES_FILE);
		labels_data_ptr = readLabelsFile(TRAINING_LABELS_FILE);
	} else {
		images_data_ptr = readImagesFile(TESTING_IMAGES_FILE);
		labels_data_ptr = readLabelsFile(TESTING_LABELS_FILE);
	}
	
	// allocate memory
	cudaCALL(CUDA_M_MALLOC_MANAGED(input_data_ptr, TYPE, TRAINING_SIZE * INPUT_SIZE));
	cudaCALL(CUDA_M_MALLOC_MANAGED(output_data_ptr, TYPE, TRAINING_SIZE * OUTPUT_SIZE));
	
	// assigning data
	for (int i = 0; i < TRAINING_SIZE * INPUT_SIZE; ++i) {
		input_data_ptr[i] = TYPE(images_data_ptr[i]) / UINT8_MAX;
	}
	for (int i = 0; i < TRAINING_SIZE; ++i) {
		output_data_ptr[i * OUTPUT_SIZE + labels_data_ptr[i]] = 1.0;
	}
}

#endif	// MNIST_HELPER_H
