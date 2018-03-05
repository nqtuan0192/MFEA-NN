#ifndef MFEA_CHROMOSOME_HPP
#define MFEA_CHROMOSOME_HPP

#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <array>
#include <random>

#include <curand.h>

#include "utils.h"
#include "Matrix.hpp"
#include "MFEATask.hpp"


#define WEIGHT_INIT_MIN	(0.0)
#define WEIGHT_INIT_MAX	(1.0)
#define WEIGHT_MIN		(-5.0)
#define WEIGHT_MAX		(5.0)

void printGPUArray(DATATYPE* ptr, uint32_t size) {
	cudaDeviceSynchronize();
	std::cout << "arr size = " << size << ": ";
	for (uint32_t i = 0; i < size; ++i) {
		std::cout << ptr[i] << " ";
	} std::cout << std::endl;
}


std::random_device rd;
std::mt19937 e(rd());
std::uniform_real_distribution<> r(WEIGHT_INIT_MIN, WEIGHT_INIT_MAX);
DATATYPE f_rand() {
	return r(e);
}

// simulated binary crossover algorithm - beg
template<typename TYPE> struct __functor_sbx_beta_transform {
};
template<> struct __functor_sbx_beta_transform<float> {
	__functor_sbx_beta_transform(float distribution_index) : _distribution_index(distribution_index) {
	}

	__host__ __device__ float operator()(const float& x) {
		if (x <= 0.5) {
			return powf(2.0 * x, 1.0 / (_distribution_index + 1.0));
		} else {
			return powf(2.0 * (1.0 - x), -1.0 / (_distribution_index + 1.0));
		}
	}
private:
	float _distribution_index;
};
template<> struct __functor_sbx_beta_transform<double> {
	__functor_sbx_beta_transform(double distribution_index) : _distribution_index(distribution_index) {
	}

	__host__ __device__ double operator()(const double& x) {
		if (x <= 0.5) {
			return  pow(2.0 * x, 1.0 / (_distribution_index + 1.0));
		} else {
			return  pow(2.0 * (1.0 - x), -1.0 / (_distribution_index + 1.0));
		}
	}
private:
	double _distribution_index;
};
template<typename TYPE> void sbx_beta_transform(size_t no_elements, TYPE* dev_ptr, TYPE distribution_index) {
	thrust::transform(thrust::device,
						thrust::device_pointer_cast<TYPE>(dev_ptr),
						thrust::device_pointer_cast<TYPE>(dev_ptr) + no_elements,
						thrust::device_pointer_cast<TYPE>(dev_ptr),
						__functor_sbx_beta_transform<TYPE>(distribution_index));
}
template<typename TYPE> struct __functor_sbx_children_generate {
	template <typename Tuple> __host__ __device__ void operator()(Tuple tup) {
		DATATYPE oneminus = 1 - thrust::get<2>(tup);
		DATATYPE oneplus = 1 + thrust::get<2>(tup);
		thrust::get<3>(tup) = 0.5 * (oneplus * thrust::get<0>(tup) + oneminus * thrust::get<1>(tup));
		thrust::get<4>(tup) = 0.5 * (oneminus * thrust::get<0>(tup) + oneplus * thrust::get<1>(tup));
		
		if (thrust::get<3>(tup) > 1) {
			thrust::get<3>(tup) = 1;
		} else if (thrust::get<3>(tup) < 0) {
			thrust::get<3>(tup) = 0;
		}

		if (thrust::get<4>(tup) > 1) {
			thrust::get<4>(tup) = 1;
		} else if (thrust::get<4>(tup) < 0) {
			thrust::get<4>(tup) = 0;
		}
	}
};

inline void crossover_core(DATATYPE* p1, DATATYPE* p2, DATATYPE* c1, DATATYPE* c2,
					  DATATYPE cf_distributionindex, DATATYPE* ct_beta,
					  size_t genotype_length) {
	sbx_beta_transform<DATATYPE>(genotype_length, ct_beta, cf_distributionindex);

	auto beg = thrust::make_zip_iterator(thrust::make_tuple(p1,
															p2,
															ct_beta,
															c1,
															c2));
	auto end = thrust::make_zip_iterator(thrust::make_tuple(p1 + genotype_length,
															p2 + genotype_length,
															ct_beta + genotype_length,
															c1 + genotype_length,
															c2 + genotype_length));
	thrust::for_each(thrust::device, beg, end, __functor_sbx_children_generate<DATATYPE>());
}	
// simulated binary crossover algorithm - end

// polynomial mutation algorithm - beg
template<typename TYPE> struct __functor_pmu_children_generate {
};
template<> struct __functor_pmu_children_generate<float> {
	template <typename Tuple> __host__ __device__ void operator()(Tuple tup) {
		if (thrust::get<3>(tup) < thrust::get<4>(tup)) {	// rand < mutation_ratio
			if (thrust::get<2>(tup) <= 0.5) {	// random binary option
				float del = powf(2 * thrust::get<2>(tup), 1.0 / (1.0 + thrust::get<5>(tup))) - 1.0;	// transform random array with polynomial mutation index
				thrust::get<1>(tup) = thrust::get<0>(tup) + del * thrust::get<0>(tup);				// update child weight itself
			} else {
				float del = 1.0 - powf(2 * (1.0 - thrust::get<2>(tup)), 1.0 / (1.0 + thrust::get<5>(tup)));	// transform random array with polynomial mutation index
				thrust::get<1>(tup) = thrust::get<0>(tup) + del * (1.0 - thrust::get<0>(tup));				// update child weight itself
			}
		} else {	// just copy from parent
			thrust::get<1>(tup) = thrust::get<0>(tup);
		}
	}
};
template<> struct __functor_pmu_children_generate<double> {
	template <typename Tuple> __host__ __device__ void operator()(Tuple tup) {
		if (thrust::get<3>(tup) < thrust::get<4>(tup)) {	// rand < mutation_ratio
			if (thrust::get<2>(tup) <= 0.5) {	// random binary option
				double del = pow(2 * thrust::get<2>(tup), 1.0 / (1.0 + thrust::get<5>(tup))) - 1.0;	// transform random array with polynomial mutation index
				thrust::get<1>(tup) = thrust::get<0>(tup) + del * thrust::get<0>(tup);				// update child weight itself
			} else {
				double del = 1.0 - pow(2 * (1.0 - thrust::get<2>(tup)), 1.0 / (1.0 + thrust::get<5>(tup)));	// transform random array with polynomial mutation index
				thrust::get<1>(tup) = thrust::get<0>(tup) + del * (1.0 - thrust::get<0>(tup));				// update child weight itself
			}
		} else {	// just copy from parent
			thrust::get<1>(tup) = thrust::get<0>(tup);
		}
	}
};
inline void mutate_core(DATATYPE* p, DATATYPE* c,
						DATATYPE mf_polynomialmutationindex, DATATYPE mf_mutationratio,
						DATATYPE* ct_beta, DATATYPE* rp,
						size_t genotype_length) {
	auto beg = thrust::make_zip_iterator(thrust::make_tuple(p,
															c,
															ct_beta,
															rp,
															thrust::make_constant_iterator(mf_mutationratio),
															thrust::make_constant_iterator(mf_polynomialmutationindex)));
	auto end = thrust::make_zip_iterator(thrust::make_tuple(p + genotype_length,
															c + genotype_length,
															ct_beta + genotype_length,
															rp + genotype_length,
															thrust::make_constant_iterator(mf_mutationratio),
															thrust::make_constant_iterator(mf_polynomialmutationindex)));
	thrust::for_each(thrust::device, beg, end, __functor_pmu_children_generate<DATATYPE>());
}
// polynomial mutation algorithm - end

// uniform crossover like algorithm - beg
template<typename TYPE> struct __functor_ucl_childs_generate {
	template <typename Tuple> __host__ __device__ void operator()(Tuple tup) {
		if (thrust::get<2>(tup) >= 0.5) {	// random binary option
			// do swap
			TYPE temp = thrust::get<0>(tup);
			thrust::get<0>(tup) = thrust::get<1>(tup);
			thrust::get<1>(tup) = temp;
		} else {
			// remain, do nothing
		}
	}
};
inline void uniformcrossoverlike_core(DATATYPE* c1, DATATYPE* c2, DATATYPE* ct_beta, size_t genotype_length) {
	auto beg = thrust::make_zip_iterator(thrust::make_tuple(c1,
															c2,
															ct_beta));
	auto end = thrust::make_zip_iterator(thrust::make_tuple(c1 + genotype_length,
															c2 + genotype_length,
															ct_beta + genotype_length));
	thrust::for_each(thrust::device, beg, end, __functor_ucl_childs_generate<DATATYPE>());
}
// uniform crossover like algorithm - end


template<typename TYPE> struct __functor_weights_transform {
	__functor_weights_transform(TYPE min_val, TYPE max_val) : _min_val(min_val), _max_val(max_val) {
	}
	__host__ __device__ TYPE operator()(const TYPE& x) {
		//return x;
		return x * (_max_val - _min_val) + _min_val;
	}
private:
	TYPE _min_val;
	TYPE _max_val;
};


/*
 * l0_size: number of hidden layers in input layer (layer 0)
 * lo_size: number of hidden layers in output layer
 * task_size: number of tasks
 * 
 * */
struct MFEA_Chromosome {
	DATATYPE scalar_fitness;
	uint32_t skill_factor;
	
	std::array<DATATYPE, TASK_SIZE> factorial_costs;
	std::array<uint32_t, TASK_SIZE> factorial_rank;
	std::array<DATATYPE, TASK_SIZE> accuracy;
	
	// layers
	DATATYPE* rnvec;	// start using from 0, layer i + 1 stored in rnvec[i] (i >= 0)

	
	/** Default constructor */
	MFEA_Chromosome() : scalar_fitness(std::numeric_limits<DATATYPE>::max()), skill_factor(std::numeric_limits<uint32_t>::max()) {
		//std::cout << "Default constructor" <<std::endl;
		for (uint32_t i = 0; i < TASK_SIZE; ++i) {
			factorial_costs[i] = std::numeric_limits<DATATYPE>::max();
			factorial_rank[i] = std::numeric_limits<uint32_t>::max();
			accuracy[i] = 0;
		}
		
		cudaCALL(CUDA_M_MALLOC_MANAGED(rnvec, DATATYPE, getTotalLayerWeightsandBiases()));
	}
	
    /** Copy constructor */
    MFEA_Chromosome(const MFEA_Chromosome& other) : scalar_fitness(other.scalar_fitness), skill_factor(other.skill_factor),
													 factorial_costs(other.factorial_costs), factorial_rank(other.factorial_rank),
													 accuracy(other.accuracy) {
        //std::cout << "Copy constructor" << std::endl;
        cudaCALL(CUDA_M_MALLOC_MANAGED(rnvec, DATATYPE, getTotalLayerWeightsandBiases()));
		cudaCALL(cudaMemcpy(this->rnvec, other.rnvec, getTotalLayerWeightsandBiases() * sizeof(DATATYPE), cudaMemcpyDeviceToDevice));
    }
    
    /** Move constructor */
    MFEA_Chromosome(MFEA_Chromosome&& other) noexcept : scalar_fitness(other.scalar_fitness), skill_factor(other.skill_factor),
														 factorial_costs(other.factorial_costs), factorial_rank(other.factorial_rank),
														 accuracy(other.accuracy) { /* noexcept needed to enable optimizations in containers */
		//std::cout << "Move constructor" << std::endl;
		other.scalar_fitness = std::numeric_limits<DATATYPE>::max();
		other.skill_factor = std::numeric_limits<uint32_t>::max();
		for (uint32_t i = 0; i < TASK_SIZE; ++i) {
			other.factorial_costs[i] = std::numeric_limits<DATATYPE>::max();
			other.factorial_rank[i] = std::numeric_limits<uint32_t>::max();
			other.accuracy[i] = 0;
		}

		this->rnvec = other.rnvec;
		other.rnvec = nullptr;
    }
    
    /** Destructor - use default */
    ~MFEA_Chromosome() noexcept { /* explicitly specified destructors should be annotated noexcept as best-practice */
        //std::cout << "Destructor" << std::endl;
		if (this->rnvec != nullptr) {
			//std::cout << "CUDA free : "; BUG(rnvec[i]);
			cudaCALL(cudaFree(this->rnvec));
			this->rnvec = nullptr;
		}
    }
    
	/** Copy assignment operator */
    MFEA_Chromosome& operator=(const MFEA_Chromosome& other) {
        //std::cout << "Copy assignment operator" << std::endl;
        this->scalar_fitness = other.scalar_fitness;
        this->skill_factor = other.skill_factor;
        this->factorial_costs = other.factorial_costs;
        this->factorial_rank = other.factorial_rank;
        this->accuracy = other.accuracy;
        
		cudaCALL(cudaMemcpy(this->rnvec, other.rnvec, getTotalLayerWeightsandBiases() * sizeof(DATATYPE), cudaMemcpyDeviceToDevice));
        return *this;
    }
    
    /** Move assignment operator */
    MFEA_Chromosome& operator=(MFEA_Chromosome&& other) noexcept {
        //std::cout << "Move assignment operator" << std::endl;
        std::swap(scalar_fitness, other.scalar_fitness);
        std::swap(skill_factor, other.skill_factor);
        std::swap(factorial_costs, other.factorial_costs);
        std::swap(factorial_rank, other.factorial_rank);
        std::swap(accuracy, other.accuracy);
        std::swap(rnvec, other.rnvec);
		
        return *this;
    }
    
    /**	ostream operator */
	friend std::ostream& operator<<(std::ostream& os, MFEA_Chromosome& chromo) {
		os << "Individual info:" << std::endl;
		for (uint32_t i = 0; i < getTotalLayerWeightsandBiases(); ++i) {
			os << chromo.rnvec[i] << " ";
		} os << std::endl;
		/*for (uint32_t i = 0; i < getNumberofLayersbyTask(chromo.skill_factor); ++i) {
			os << "Layer " << i << "(size = " << getMaximumLayerWeightsandBiasesbyLayer(i + 1) << "): ";
			uint32_t loff = getLayerOffset(chromo.skill_factor, i + 1);
			for (uint32_t j = 0; j < getMaximumLayerWeightsandBiasesbyLayer(i + 1); ++j) {
				os << chromo.rnvec[loff + j] << " ";
			}

			auto w_tup = getLayerWeightsbyTaskLayer(0, i + 1);
			auto b_tup = getLayerBiasesbyTaskLayer(0, i + 1);

			BUG_ENDL;
			BUG(std::get<OFFSET_IDX>(w_tup));	BUG(std::get<SIZE_IDX>(w_tup));
			BUG(std::get<OFFSET_IDX>(b_tup));	BUG(std::get<SIZE_IDX>(b_tup));

			os << "Weights: ";
			for (uint32_t j = 0; j < std::get<SIZE_IDX>(w_tup); ++j) {
				os << chromo.rnvec[std::get<OFFSET_IDX>(w_tup) + j] << " ";
			} os << std::endl;
			os << "biases: ";
			for (uint32_t j = 0; j < std::get<SIZE_IDX>(b_tup); ++j) {
				os << chromo.rnvec[std::get<OFFSET_IDX>(b_tup) + j] << " ";
			} os << std::endl;

			os << std::endl;
		}*/
		os << "Factorial cost: ";
		for (uint32_t i = 0; i < TASK_SIZE; ++i) {
			os << chromo.factorial_costs[i] << "\t";
		}	os << std::endl;
		os << "Factorial rank: ";
		for (uint32_t i = 0; i < TASK_SIZE; ++i) {
			os << chromo.factorial_rank[i] << "\t";
		}	os << std::endl;
		os << "Scalar fitness: " << chromo.scalar_fitness << std::endl;
		os << "Skill factor: " << chromo.skill_factor << std::endl;
		os << "Predict accuracy: ";
		for (uint32_t i = 0; i < TASK_SIZE; ++i) {
			os << chromo.accuracy[i] << "\t";
		}	os << std::endl;
		return os;
	}
	
	/**	Compare operator */
	friend bool operator<(const MFEA_Chromosome& chromo1, const MFEA_Chromosome& chromo2) {
		return chromo1.scalar_fitness < chromo2.scalar_fitness;
	}
	friend bool operator<(MFEA_Chromosome& chromo1, MFEA_Chromosome& chromo2) {
		return chromo1.scalar_fitness < chromo2.scalar_fitness;
	}
	
	void exportToFile(std::string filename) {
		std::ofstream ofile(filename, std::ofstream::binary);
		ofile.write((char*)rnvec, getTotalLayerWeightsandBiases() * sizeof(DATATYPE));
		ofile.close();
	}
	void loadFromFile(std::string filename) {
		std::ifstream ifile(filename, std::ifstream::binary);
		ifile.read((char*)rnvec, getTotalLayerWeightsandBiases() * sizeof(DATATYPE));
		ifile.close();
	}

	std::tuple<uint32_t, uint32_t> decode(DATATYPE* inp_rnvec, DATATYPE* W, uint32_t task, uint32_t layer, cublasHandle_t cublas_handle) {
		std::tuple<uint32_t, size_t> tup = getLayerWeightsbyTaskLayer(task, layer);
		uint32_t offset = std::get<0>(tup);
		size_t size = std::get<1>(tup);

		uint32_t unrow;
		uint32_t uncol;
		if (layer == getNumberofLayersbyTask(task)) {
			unrow = getMaximumNumberofUnitsofUnifiedLayer(getUnifiedNumberofLayers());
			uncol = getMaximumNumberofUnitsofUnifiedLayer(getUnifiedNumberofLayers() - 1);
		} else {
			unrow = getMaximumNumberofUnitsofUnifiedLayer(layer);
			uncol = getMaximumNumberofUnitsofUnifiedLayer(layer - 1);
		}

		cublas_transposeMatrix<DATATYPE>(unrow, uncol, inp_rnvec + offset, W, cublas_handle);
		//cudaDeviceSynchronize();
		//std::cout << "checkpoint\n";
		//printMatrix<DATATYPE>(1, getTotalLayerWeightsandBiases(), inp_rnvec);

		uint32_t nrow = getNumberofUnitsbyTaskLayer(task, layer);
		uint32_t ncol = getNumberofUnitsbyTaskLayer(task, layer - 1);

		cuda_copySubMatrix(0, 0, W, uncol, unrow, W, ncol, nrow);

		return std::make_tuple(ncol, nrow);
	}

	void evalObj(size_t training_size, size_t output_size, DATATYPE* X, DATATYPE* Y,
						DATATYPE* mat_temp_rnvec, DATATYPE* mat_temp_w,
						DATATYPE* mat_one, std::array<DATATYPE*, LAYER_SIZE + 1> mat_temp_layer,
						cublasHandle_t& cublas_handle, cudnnHandle_t& cudnn_handle,
						bool is_evalAcc = false, bool is_alltaskeval = false, bool is_reupdateskillfactor = false) {
		DATATYPE factorial_cost_min = std::numeric_limits<DATATYPE>::max();

		// do weights transformation before evaluating
		thrust::transform(thrust::device, this->rnvec, this->rnvec + getTotalLayerWeightsandBiases(), mat_temp_rnvec, __functor_weights_transform<DATATYPE>(-5, 5));

		mat_temp_layer[0] = X;

		for (uint32_t task = 0; task < TASK_SIZE; ++task) {
			if ((task == skill_factor) || is_alltaskeval) {
				uint32_t numberof_layers = getNumberofLayersbyTask(task);
				for (uint32_t layer = 1; layer < numberof_layers; ++layer) {
					// decode w
					std::tuple<uint32_t, uint32_t> w_shape = this->decode(mat_temp_rnvec, mat_temp_w, task, layer, cublas_handle);
					uint32_t w_nrow = std::get<MATRIX_NROW>(w_shape), w_ncol = std::get<MATRIX_NCOL>(w_shape);

					// prepare b
					auto b_tup = getLayerBiasesbyTaskLayer(task, layer);
					uint32_t b_off = std::get<OFFSET_IDX>(b_tup), b_size = std::get<SIZE_IDX>(b_tup);

					//cudaDeviceSynchronize();
					//printMatrix<DATATYPE>(w_nrow, w_ncol, mat_temp_w);
					//printMatrix<DATATYPE>(1, b_size, mat_temp_rnvec + b_off);

					cublas_multiplyMatrices<DATATYPE>(training_size, b_size, 1,
														mat_one,
														mat_temp_rnvec + b_off,	// layer index started at 0
														mat_temp_layer[layer],	// layer index started at 0
														cublas_handle);
					//cudaDeviceSynchronize();
					//printMatrix<DATATYPE>(training_size, b_size, mat_temp_layer[layer]);

					// multiply add z[i] = z[i-1] * w[i-1] + b[i-1]
					cublas_multiplyandaddMatrices<DATATYPE>(training_size, b_size, w_nrow,
															mat_temp_layer[layer - 1],
															mat_temp_w,
															mat_temp_layer[layer],
															cublas_handle);
					//cudaDeviceSynchronize();
					//printMatrix<DATATYPE>(training_size, b_size, mat_temp_layer[layer]);

					// apply activation function default by sigmoid
					cuda_relu<DATATYPE>(training_size, b_size, mat_temp_layer[layer], mat_temp_layer[layer]);
					//cudaDeviceSynchronize();
					//printMatrix<DATATYPE>(training_size, b_size, mat_temp_layer[layer]);
				}

				// decode w
				std::tuple<uint32_t, uint32_t> w_shape = this->decode(mat_temp_rnvec, mat_temp_w, task, numberof_layers, cublas_handle);
				uint32_t w_nrow = std::get<MATRIX_NROW>(w_shape), w_ncol = std::get<MATRIX_NCOL>(w_shape);

				// prepare b
				auto b_tup = getLayerBiasesbyTaskLayer(task, numberof_layers);
				uint32_t b_off = std::get<OFFSET_IDX>(b_tup), b_size = std::get<SIZE_IDX>(b_tup);

				//cudaDeviceSynchronize();
				//printMatrix<DATATYPE>(w_nrow, w_ncol, mat_temp_w);
				//printMatrix<DATATYPE>(1, b_size, mat_temp_rnvec + b_off);

				cublas_multiplyMatrices<DATATYPE>(training_size, b_size, 1,
													mat_one,
													mat_temp_rnvec + b_off,	// layer index started at 0
													mat_temp_layer[numberof_layers],	// layer index started at 0
													cublas_handle);
				//cudaDeviceSynchronize();
				//printMatrix<DATATYPE>(training_size, b_size, mat_temp_layer[numberof_layers]);

				// multiply add z[i] = z[i-1] * w[i-1] + b[i-1]
				cublas_multiplyandaddMatrices<DATATYPE>(training_size, b_size, w_nrow,
														mat_temp_layer[numberof_layers - 1],
														mat_temp_w,
														mat_temp_layer[numberof_layers],
														cublas_handle);
				//cudaDeviceSynchronize();
				//printMatrix<DATATYPE>(training_size, b_size, mat_temp_layer[numberof_layers]);

				// apply activation function softmax to final layer
				cudnn_softmax<DATATYPE>(training_size, b_size,
										mat_temp_layer[numberof_layers], mat_temp_layer[numberof_layers],
										cudnn_handle);
				cuda_eliminatezero(training_size, b_size, mat_temp_layer[numberof_layers], mat_temp_layer[numberof_layers]);
				// cuda_sigmoid<DATATYPE>(training_size, b_size, mat_temp_layer[numberof_layers], mat_temp_layer[numberof_layers]);
				//cudaDeviceSynchronize();
				//printMatrix<DATATYPE>(training_size, b_size, mat_temp_layer[numberof_layers]);

				
				// eval cross entropy over the training_size
				//BUG(getNumberofUnitsofLastLayerbyTask(task));
				//printMatrix<DATATYPE>(training_size, getNumberofUnitsofLastLayerbyTask(task), Y);
				//printMatrix<DATATYPE>(training_size, getNumberofUnitsofLastLayerbyTask(task), mat_temp_layer[numberof_layers]);
				factorial_costs[task] = cuda_evalCrossEntropy<DATATYPE>(training_size, b_size, Y, mat_temp_layer[numberof_layers]);
				// factorial_costs[task] = cuda_evalMSE<DATATYPE>(training_size, getNumberofUnitsofLastLayerbyTask(task),
				// 														Y, mat_temp_layer[numberof_layers]);
				
				// update skill factor
				if ((factorial_costs[task] < factorial_cost_min) && is_reupdateskillfactor) {
					skill_factor = task;
					factorial_cost_min = factorial_costs[task];
				}
				if (is_evalAcc) {
					accuracy[task] = cuda_evalAccuracy<DATATYPE>(training_size, getNumberofUnitsofLastLayerbyTask(task),
															Y, mat_temp_layer[numberof_layers]);
				}
			} else {
				factorial_costs[task] = std::numeric_limits<DATATYPE>::max();
			}
		}
	}


	friend void crossover(MFEA_Chromosome& chromo1, MFEA_Chromosome& chromo2,
						  MFEA_Chromosome& child1, MFEA_Chromosome& child2,
						  DATATYPE cf_distributionindex, DATATYPE* ct_beta,
						  const curandGenerator_t& prng) {
		// beta value must be in range [0, 1]
		curand_randomizeArray<DATATYPE>(prng, ct_beta, 1, getTotalLayerWeightsandBiases(), 0, 1);
		// invoke core crossover
		crossover_core(chromo1.rnvec, chromo2.rnvec, child1.rnvec, child2.rnvec,
						cf_distributionindex, ct_beta, getTotalLayerWeightsandBiases());
	}

	friend void test_crossover(MFEA_Chromosome& chromo1, MFEA_Chromosome& chromo2,
						  MFEA_Chromosome& child1, MFEA_Chromosome& child2,
						  DATATYPE cf_distributionindex, DATATYPE* ct_beta,
						  const curandGenerator_t& prng) {
		cf_distributionindex = 2;
		size_t arr_len = getTotalLayerWeightsandBiases();
		DATATYPE arr_p1[] = {0.40058077,  0.10789012,  0.66173051,  0.09754657,  0.82390663,
					        0.27997668,  0.75598554,  0.16852512,  0.38259621,  0.40241174,
					        0.12989646,  0.55951094,  0.20796159,  0.95924295,  0.17089823,
					        0.32895566,  0.53358916,  0.86704552,  0.17475058,  0.60126442,
					        0.12464698,  0.30750506,  0.9787328 ,  0.70446726,  0.91416106,
					        0.27524185,  0.11129455,  0.41308348,  0.58408755,  0.6584263 ,
					        0.76198835,  0.93218168,  0.28613302,  0.26102955,  0.19515768,
					        0.87862592,  0.15490241,  0.83462384,  0.74058954,  0.02662627,
					        0.35583321,  0.19753112,  0.77318452,  0.9932761 ,  0.3736706 ,
					        0.04807932,  0.41503751,  0.48654105,  0.00918542,  0.96629716,
					        0.3045214 ,  0.97625911,  0.74494477,  0.37351417,  0.70412958,
					        0.42843002,  0.22495722,  0.43897174,  0.75720537,  0.16692599,
					        0.92578559,  0.51164152,  0.02608266,  0.26014288,  0.06493951,
					        0.91603925,  0.92673705,  0.55254473};
		DATATYPE arr_p2[] = {0.25200831,  0.13275223,  0.29404046,  0.79132252,  0.9840467 ,
					        0.53133389,  0.42160015,  0.74310602,  0.86266448,  0.51111094,
					        0.48997705,  0.40472111,  0.4401997 ,  0.43793953,  0.11574458,
					        0.93452713,  0.52037384,  0.68871181,  0.4543441 ,  0.78055969,
					        0.22313558,  0.84663807,  0.95861796,  0.13044873,  0.43735409,
					        0.0409308 ,  0.77568847,  0.5230747 ,  0.03226744,  0.90622697,
					        0.9240827 ,  0.28024557,  0.74029606,  0.07383527,  0.65452409,
					        0.31436297,  0.02454635,  0.30452135,  0.53712043,  0.03446845,
					        0.87904375,  0.41756199,  0.51163042,  0.96695009,  0.92925426,
					        0.71872782,  0.72621547,  0.83758528,  0.63460803,  0.60689358,
					        0.86918629,  0.12683783,  0.98118576,  0.25187347,  0.23354516,
					        0.62378106,  0.55692021,  0.5462752 ,  0.87133431,  0.00575162,
					        0.15201889,  0.19396073,  0.5526941 ,  0.36077303,  0.64714191,
					        0.41893931,  0.02558479,  0.79018024};
		DATATYPE arr_u[] = {0.55401019,  0.85047611,  0.8227469 ,  0.0403004 ,  0.30381855,
					        0.8021805 ,  0.86406417,  0.48714753,  0.79786296,  0.05519576,
					        0.54080356,  0.43601656,  0.82913197,  0.08757968,  0.12134181,
					        0.89070392,  0.00311884,  0.44511744,  0.23563644,  0.45067704,
					        0.72420047,  0.22874243,  0.96837766,  0.7535714 ,  0.81153212,
					        0.68901919,  0.23426194,  0.93525361,  0.2653546 ,  0.9608321 ,
					        0.28923025,  0.75630549,  0.89329305,  0.33717517,  0.28190506,
					        0.45158795,  0.35403079,  0.84567846,  0.45121137,  0.09780998,
					        0.27767   ,  0.58663004,  0.18719402,  0.22522257,  0.21045784,
					        0.62332907,  0.49107774,  0.03337812,  0.08460284,  0.83961386,
					        0.14858183,  0.48390999,  0.16443224,  0.47573758,  0.91038048,
					        0.86998783,  0.36185703,  0.74795191,  0.15545892,  0.18190848,
					        0.22547659,  0.9118953 ,  0.46280127,  0.37432461,  0.86560837,
					        0.04057165,  0.44815559,  0.99477211};

		cudaCALL(cudaMemcpy(chromo1.rnvec, arr_p1, arr_len * sizeof(DATATYPE), cudaMemcpyHostToDevice));
		cudaCALL(cudaMemcpy(chromo2.rnvec, arr_p2, arr_len * sizeof(DATATYPE), cudaMemcpyHostToDevice));
		cudaCALL(cudaMemcpy(ct_beta, arr_u, arr_len * sizeof(DATATYPE), cudaMemcpyHostToDevice));

		printGPUArray(chromo1.rnvec, arr_len);
		printGPUArray(chromo2.rnvec, arr_len);
		printGPUArray(ct_beta, arr_len);

		// invoke core crossover
		crossover_core(chromo1.rnvec, chromo2.rnvec, child1.rnvec, child2.rnvec,
						cf_distributionindex, ct_beta, arr_len);

		printGPUArray(child1.rnvec, arr_len);
		// arr size = 68: 0.403466 0.101732 0.737648 0.294592 0.836158 0.234459 0.846878 0.171008 0.298006 0.430689 0.124714 0.556058 0.157992 0.84443 0.160523 0.129101 0.528198 0.863656 0.205758 0.604315 0.113846 0.369361 0.993919 0.780806 1 0.295335 0.18548 0.359375 0.531562 0.492757 0.775506 1 0.133223 0.24951 0.235094 0.869209 0.147818 0.961776 0.737167 0.0282712 0.402408 0.190328 0.736663 0.990203 0.443277 0.014878 0.415969 0.590861 0.148936 1 0.398449 0.971654 0.781533 0.372514 0.886152 0.373074 0.241917 0.42521 0.775611 0.143869 0.835584 0.636126 0.0327814 0.264771 0 0.775094 0.910592 0.127992 
		printGPUArray(child2.rnvec, arr_len);
		// arr size = 68: 0.249123 0.13891 0.218122 0.594277 0.971796 0.576852 0.330708 0.740623 0.947255 0.482833 0.495159 0.408174 0.490169 0.552753 0.12612 1 0.525765 0.692102 0.423337 0.777509 0.233937 0.784782 0.943432 0.0541105 0.345725 0.0208376 0.701503 0.576783 0.084793 1 0.910565 0.192008 0.893206 0.0853548 0.614587 0.323779 0.0316312 0.17737 0.540543 0.0328236 0.832469 0.424765 0.548152 0.970023 0.859648 0.751929 0.725284 0.733266 0.494858 0.524081 0.775259 0.131443 0.944598 0.252874 0.051523 0.679137 0.53996 0.560037 0.852929 0.0288089 0.242221 0.0694763 0.545995 0.356145 0.80711 0.559885 0.0417296 1 
		printGPUArray(ct_beta, arr_len);
		// arr size = 68: 1.03884 1.49539 1.41295 0.431963 0.846996 1.36218 1.54364 0.991357 1.35241 0.47971 1.02878 0.955383 1.43033 0.559514 0.623754 1.66005 0.18408 0.961985 0.778199 0.965973 1.21934 0.770535 2.50991 1.26598 1.38434 1.17151 0.776683 1.9766 0.809628 2.33711 0.833217 1.2707 1.67337 0.876924 0.826122 0.966624 0.8913 1.47973 0.966355 0.580503 0.821964 1.06548 0.720732 0.766562 0.749431 1.09901 0.994016 0.405662 0.553102 1.46083 0.667317 0.989156 0.690249 0.983556 1.7736 1.56673 0.897819 1.2565 0.677457 0.713884 0.76685 1.78371 0.974559 0.908015 1.54953 0.43293 0.964168 4.57314 
	}

	template<typename TYPE> struct __functor_examineCrossover {
		template <typename Tuple> __host__ __device__ bool operator()(Tuple tup) {
			return (thrust::get<0>(tup) + thrust::get<1>(tup)) - (thrust::get<2>(tup) + thrust::get<3>(tup)) > 0.0000001;
		}
	};
	friend void examineCrossover(MFEA_Chromosome& chromo1, MFEA_Chromosome& chromo2, MFEA_Chromosome& child1, MFEA_Chromosome& child2) {
		std::cout << "------- BEGIN examining crossover operator -------" << std::endl;	
		auto begin = thrust::make_zip_iterator(thrust::make_tuple(chromo1.rnvec,
																chromo2.rnvec,
																child1.rnvec,
																child2.rnvec));
		auto end = thrust::make_zip_iterator(thrust::make_tuple(chromo1.rnvec + getTotalLayerWeightsandBiases(),
																chromo2.rnvec + getTotalLayerWeightsandBiases(),
																child1.rnvec + getTotalLayerWeightsandBiases(),
																child2.rnvec + getTotalLayerWeightsandBiases()));
		uint32_t count_error = thrust::count_if(thrust::device, begin, end, __functor_examineCrossover<DATATYPE>());
		std::cout << "Error count = " << count_error << std::endl;
		std::cout << "------- END examine crossover operator -------" << std::endl;
	}

	friend void mutate(MFEA_Chromosome& chromo, MFEA_Chromosome& child,
						DATATYPE mf_polynomialmutationindex, DATATYPE mf_mutationratio,
						DATATYPE* ct_beta, DATATYPE* rp, const curandGenerator_t& prng) {
		curand_randomizeArray<DATATYPE>(prng, ct_beta, 1, getTotalLayerWeightsandBiases(), 0, 1);

		// invoke core mutation
		mutate_core(chromo.rnvec, child.rnvec,
					mf_polynomialmutationindex, mf_mutationratio,
					ct_beta, rp, getTotalLayerWeightsandBiases());
	}

	friend void test_mutate(MFEA_Chromosome& chromo, MFEA_Chromosome& child,
						DATATYPE mf_polynomialmutationindex, DATATYPE mf_mutationratio,
						DATATYPE* ct_beta, DATATYPE* rp, const curandGenerator_t& prng) {
		mf_polynomialmutationindex = 5;
		mf_mutationratio = DATATYPE(1.0) / getTotalLayerWeightsandBiases();
		size_t arr_len = getTotalLayerWeightsandBiases();
		DATATYPE arr_p[] = {0.40058077,  0.10789012,  0.66173051,  0.09754657,  0.82390663,
					        0.27997668,  0.75598554,  0.16852512,  0.38259621,  0.40241174,
					        0.12989646,  0.55951094,  0.20796159,  0.95924295,  0.17089823,
					        0.32895566,  0.53358916,  0.86704552,  0.17475058,  0.60126442,
					        0.12464698,  0.30750506,  0.9787328 ,  0.70446726,  0.91416106,
					        0.27524185,  0.11129455,  0.41308348,  0.58408755,  0.6584263 ,
					        0.76198835,  0.93218168,  0.28613302,  0.26102955,  0.19515768,
					        0.87862592,  0.15490241,  0.83462384,  0.74058954,  0.02662627,
					        0.35583321,  0.19753112,  0.77318452,  0.9932761 ,  0.3736706 ,
					        0.04807932,  0.41503751,  0.48654105,  0.00918542,  0.96629716,
					        0.3045214 ,  0.97625911,  0.74494477,  0.37351417,  0.70412958,
					        0.42843002,  0.22495722,  0.43897174,  0.75720537,  0.16692599,
					        0.92578559,  0.51164152,  0.02608266,  0.26014288,  0.06493951,
					        0.91603925,  0.92673705,  0.55254473};
		DATATYPE arr_ct_beta[] = {0.25200831,  0.13275223,  0.29404046,  0.79132252,  0.9840467 ,
					        0.53133389,  0.42160015,  0.74310602,  0.86266448,  0.51111094,
					        0.48997705,  0.40472111,  0.4401997 ,  0.43793953,  0.11574458,
					        0.93452713,  0.52037384,  0.68871181,  0.4543441 ,  0.78055969,
					        0.22313558,  0.84663807,  0.95861796,  0.13044873,  0.43735409,
					        0.0409308 ,  0.77568847,  0.5230747 ,  0.03226744,  0.90622697,
					        0.9240827 ,  0.28024557,  0.74029606,  0.07383527,  0.65452409,
					        0.31436297,  0.02454635,  0.30452135,  0.53712043,  0.03446845,
					        0.87904375,  0.41756199,  0.51163042,  0.96695009,  0.92925426,
					        0.71872782,  0.72621547,  0.83758528,  0.63460803,  0.60689358,
					        0.86918629,  0.12683783,  0.98118576,  0.25187347,  0.23354516,
					        0.62378106,  0.55692021,  0.5462752 ,  0.87133431,  0.00575162,
					        0.15201889,  0.19396073,  0.5526941 ,  0.36077303,  0.64714191,
					        0.41893931,  0.02558479,  0.79018024};
		DATATYPE arr_rp[] = {0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
					        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
					        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
					        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
					        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
					        0.,  0.,  0.};

		cudaCALL(cudaMemcpy(chromo.rnvec, arr_p, arr_len * sizeof(DATATYPE), cudaMemcpyHostToDevice));
		cudaCALL(cudaMemcpy(ct_beta, arr_ct_beta, arr_len * sizeof(DATATYPE), cudaMemcpyHostToDevice));
		cudaCALL(cudaMemcpy(rp, arr_rp, arr_len * sizeof(DATATYPE), cudaMemcpyHostToDevice));

		printGPUArray(chromo.rnvec, arr_len);
		printGPUArray(ct_beta, arr_len);
		printGPUArray(rp, arr_len);

		// invoke core mutation
		mutate_core(chromo.rnvec, child.rnvec,
					mf_polynomialmutationindex, mf_mutationratio,
					ct_beta, rp, arr_len);


		printGPUArray(child.rnvec, arr_len);
		// arr size = 68: 0.357353 0.0864955 0.605695 0.219854 0.900828 0.287701 0.734799 0.255874 0.502219 0.404646 0.129459 0.54014 0.203593 0.938288 0.133914 0.521811 0.536812 0.877142 0.171984 0.652403 0.108964 0.431311 0.985961 0.563126 0.893991 0.181368 0.222432 0.417687 0.369922 0.741574 0.826156 0.846438 0.359967 0.189775 0.243251 0.813232 0.0937334 0.76842 0.743903 0.0170498 0.491519 0.191688 0.774072 0.995724 0.547876 0.135111 0.470904 0.574289 0.0596478 0.967622 0.4438 0.776745 0.852354 0.333178 0.62023 0.454894 0.240413 0.44798 0.806363 0.0793099 0.759157 0.436941 0.0439927 0.246371 0.11771 0.889428 0.564666 0.612835 
	}
	
	friend void uniformcrossoverlike(MFEA_Chromosome& child1, MFEA_Chromosome& child2,
										DATATYPE* ct_beta, const curandGenerator_t& prng) {
		curand_randomizeArray<DATATYPE>(prng, ct_beta, 1, getTotalLayerWeightsandBiases(), 0, 1);
		// invoke uniformcrossoverlike_core
		uniformcrossoverlike_core(child1.rnvec, child2.rnvec, ct_beta, getTotalLayerWeightsandBiases());
	}

	friend void test_uniformcrossoverlike(MFEA_Chromosome& child1, MFEA_Chromosome& child2,
										DATATYPE* ct_beta, const curandGenerator_t& prng) {
		size_t arr_len = getTotalLayerWeightsandBiases();
		DATATYPE arr_p1[] = {0.403466  ,  0.10173195,  0.73764847,  0.29459181,  0.83615766,
					        0.23445875,  0.84687778,  0.17100811,  0.29800603,  0.4306893 ,
					        0.12471434,  0.55605784,  0.15799202,  0.84442957,  0.16052257,
					        0.12910119,  0.52819784,  0.86365582,  0.2057576 ,  0.60431483,
					        0.11384564,  0.36936119,  0.99391863,  0.78080553,  1.        ,
					        0.29533503,  0.18547971,  0.35937492,  0.53156201,  0.49275743,
					        0.77550567,  1.        ,  0.13322275,  0.24950997,  0.23509449,
					        0.86920942,  0.14781752,  0.96177568,  0.73716668,  0.02827116,
					        0.40240827,  0.19032774,  0.73666272,  0.99020335,  0.44327659,
					        0.01487803,  0.41596854,  0.59086058,  0.14893558,  1.        ,
					        0.39844873,  0.97165361,  0.78153274,  0.37251405,  0.88615173,
					        0.37307396,  0.2419173 ,  0.42521012,  0.7756111 ,  0.14386871,
					        0.83558378,  0.63612594,  0.0327814 ,  0.26477114,  0.        ,
					        0.77509399,  0.91059223,  0.12799212};
		DATATYPE arr_p2[] = {0.24912308,  0.1389104 ,  0.2181225 ,  0.59427728,  0.97179567,
					        0.57685182,  0.33070791,  0.74062303,  0.94725466,  0.48283337,
					        0.49515917,  0.40817422,  0.49016927,  0.55275291,  0.12612024,
					        1.        ,  0.52576516,  0.69210151,  0.42333707,  0.78277255,
					        0.23393691,  0.78478193,  0.94343213,  0.05411046,  0.34572499,
					        0.02083762,  0.70150331,  0.57678325,  0.08479298,  1.        ,
					        0.91056538,  0.19200755,  0.89320634,  0.08535485,  0.61458728,
					        0.32377947,  0.03163123,  0.17736952,  0.54054329,  0.03282357,
					        0.83246869,  0.46495077,  0.54815223,  0.97002284,  0.85964827,
					        0.75192912,  0.72528444,  0.73326575,  0.49485787,  0.52408077,
					        0.77525896,  0.13144333,  0.94459779,  0.25287359,  0.051523  ,
					        0.67913712,  0.53996013,  0.56003682,  0.85292858,  0.0288089 ,
					        0.2422207 ,  0.06947631,  0.54599536,  0.35614477,  0.80711   ,
					        0.55988457,  0.04172961,  1.        };
		DATATYPE arr_ct_beta[] = {0.86335564,  0.36834361,  0.85389856,  0.02573013,  0.44752639,
					        0.87542514,  0.8682124 ,  0.54806114,  0.88701455,  0.89737247,
					        0.05367614,  0.60358718,  0.72588557,  0.54748066,  0.95682063,
					        0.61976026,  0.13468317,  0.85574888,  0.64665441,  0.29429707,
					        0.92660739,  0.24301152,  0.91197534,  0.36638151,  0.72398239,
					        0.87119611,  0.98097662,  0.47081251,  0.62757963,  0.60200898,
					        0.36201012,  0.71184907,  0.51302716,  0.43023373,  0.43632189,
					        0.81552029,  0.82769663,  0.08061939,  0.57964562,  0.24211439,
					        0.05363023,  0.50664714,  0.0106789 ,  0.36384069,  0.30344904,
					        0.5274006 ,  0.17595282,  0.87409106,  0.8001164 ,  0.2290995 ,
					        0.99254275,  0.30784949,  0.03791029,  0.4611907 ,  0.0178598 ,
					        0.83138265,  0.48918022,  0.66021231,  0.58113956,  0.22394684,
					        0.54692974,  0.97539042,  0.02725502,  0.09440559,  0.29156004,
					        0.32864267,  0.36880155,  0.9205902 };
		cudaCALL(cudaMemcpy(child1.rnvec, arr_p1, arr_len * sizeof(DATATYPE), cudaMemcpyHostToDevice));
		cudaCALL(cudaMemcpy(child2.rnvec, arr_p2, arr_len * sizeof(DATATYPE), cudaMemcpyHostToDevice));
		cudaCALL(cudaMemcpy(ct_beta, arr_ct_beta, arr_len * sizeof(DATATYPE), cudaMemcpyHostToDevice));

		printGPUArray(child1.rnvec, arr_len);
		printGPUArray(child2.rnvec, arr_len);

		// invoke uniformcrossoverlike_core
		uniformcrossoverlike_core(child1.rnvec, child2.rnvec, ct_beta, arr_len);

		printGPUArray(child1.rnvec, arr_len);
		// arr size = 68: 0.249123 0.101732 0.218122 0.294592 0.836158 0.576852 0.330708 0.740623 0.947255 0.482833 0.124714 0.408174 0.490169 0.552753 0.12612 1 0.528198 0.692102 0.423337 0.604315 0.233937 0.369361 0.943432 0.780806 0.345725 0.0208376 0.701503 0.359375 0.084793 1 0.775506 0.192008 0.893206 0.24951 0.235094 0.323779 0.0316312 0.961776 0.540543 0.0282712 0.402408 0.464951 0.736663 0.990203 0.443277 0.751929 0.415969 0.733266 0.494858 1 0.775259 0.971654 0.781533 0.372514 0.886152 0.679137 0.241917 0.560037 0.852929 0.143869 0.242221 0.0694763 0.0327814 0.264771 0 0.775094 0.910592 1 
		printGPUArray(child2.rnvec, arr_len);
		// arr size = 68: 0.403466 0.13891 0.737648 0.594277 0.971796 0.234459 0.846878 0.171008 0.298006 0.430689 0.495159 0.556058 0.157992 0.84443 0.160523 0.129101 0.525765 0.863656 0.205758 0.782773 0.113846 0.784782 0.993919 0.0541105 1 0.295335 0.18548 0.576783 0.531562 0.492757 0.910565 1 0.133223 0.0853548 0.614587 0.869209 0.147818 0.17737 0.737167 0.0328236 0.832469 0.190328 0.548152 0.970023 0.859648 0.014878 0.725284 0.590861 0.148936 0.524081 0.398449 0.131443 0.944598 0.252874 0.051523 0.373074 0.53996 0.42521 0.775611 0.0288089 0.835584 0.636126 0.545995 0.356145 0.80711 0.559885 0.0417296 0.127992 
	}




	friend void test_reproduce(MFEA_Chromosome& chromo1, MFEA_Chromosome& chromo2,
						  MFEA_Chromosome& child1, MFEA_Chromosome& child2,
						  DATATYPE cf_distributionindex, DATATYPE mf_polynomialmutationindex, DATATYPE mf_mutationratio,
						  DATATYPE* ct_beta, DATATYPE* rp,
						  const curandGenerator_t& prng) {
		cf_distributionindex = 2;
		mf_polynomialmutationindex = 5;
		mf_mutationratio = DATATYPE(1.0) / getTotalLayerWeightsandBiases();

		size_t arr_len = getTotalLayerWeightsandBiases();


		DATATYPE arr_p1[] = {0.40058077,  0.10789012,  0.66173051,  0.09754657,  0.82390663,
					         0.27997668,  0.75598554,  0.16852512,  0.38259621,  0.40241174,
					         0.12989646,  0.55951094,  0.20796159,  0.95924295,  0.17089823,
					         0.32895566,  0.53358916,  0.86704552,  0.17475058,  0.60126442,
					         0.12464698,  0.30750506,  0.9787328 ,  0.70446726,  0.91416106,
					         0.27524185,  0.11129455,  0.41308348,  0.58408755,  0.6584263 ,
					         0.76198835,  0.93218168,  0.28613302,  0.26102955,  0.19515768,
					         0.87862592,  0.15490241,  0.83462384,  0.74058954,  0.02662627,
					         0.35583321,  0.19753112,  0.77318452,  0.9932761 ,  0.3736706 ,
					         0.04807932,  0.41503751,  0.48654105,  0.00918542,  0.96629716,
					         0.3045214 ,  0.97625911,  0.74494477,  0.37351417,  0.70412958,
					         0.42843002,  0.22495722,  0.43897174,  0.75720537,  0.16692599,
					         0.92578559,  0.51164152,  0.02608266,  0.26014288,  0.06493951,
					         0.91603925,  0.92673705,  0.55254473};
		cudaCALL(cudaMemcpy(chromo1.rnvec, arr_p1, arr_len * sizeof(DATATYPE), cudaMemcpyHostToDevice));
		chromo1.skill_factor = 1;

		DATATYPE arr_p2[] = {0.25200831,  0.13275223,  0.29404046,  0.79132252,  0.9840467 ,
					         0.53133389,  0.42160015,  0.74310602,  0.86266448,  0.51111094,
					         0.48997705,  0.40472111,  0.4401997 ,  0.43793953,  0.11574458,
					         0.93452713,  0.52037384,  0.68871181,  0.4543441 ,  0.78055969,
					         0.22313558,  0.84663807,  0.95861796,  0.13044873,  0.43735409,
					         0.0409308 ,  0.77568847,  0.5230747 ,  0.03226744,  0.90622697,
					         0.9240827 ,  0.28024557,  0.74029606,  0.07383527,  0.65452409,
					         0.31436297,  0.02454635,  0.30452135,  0.53712043,  0.03446845,
					         0.87904375,  0.41756199,  0.51163042,  0.96695009,  0.92925426,
					         0.71872782,  0.72621547,  0.83758528,  0.63460803,  0.60689358,
					         0.86918629,  0.12683783,  0.98118576,  0.25187347,  0.23354516,
					         0.62378106,  0.55692021,  0.5462752 ,  0.87133431,  0.00575162,
					         0.15201889,  0.19396073,  0.5526941 ,  0.36077303,  0.64714191,
					         0.41893931,  0.02558479,  0.79018024};
		cudaCALL(cudaMemcpy(chromo2.rnvec, arr_p2, arr_len * sizeof(DATATYPE), cudaMemcpyHostToDevice));
		chromo2.skill_factor = 0;

		DATATYPE arr_u[] = {0.55401019,  0.85047611,  0.8227469 ,  0.0403004 ,  0.30381855,
					        0.8021805 ,  0.86406417,  0.48714753,  0.79786296,  0.05519576,
					        0.54080356,  0.43601656,  0.82913197,  0.08757968,  0.12134181,
					        0.89070392,  0.00311884,  0.44511744,  0.23563644,  0.45067704,
					        0.72420047,  0.22874243,  0.96837766,  0.7535714 ,  0.81153212,
					        0.68901919,  0.23426194,  0.93525361,  0.2653546 ,  0.9608321 ,
					        0.28923025,  0.75630549,  0.89329305,  0.33717517,  0.28190506,
					        0.45158795,  0.35403079,  0.84567846,  0.45121137,  0.09780998,
					        0.27767   ,  0.58663004,  0.18719402,  0.22522257,  0.21045784,
					        0.62332907,  0.49107774,  0.03337812,  0.08460284,  0.83961386,
					        0.14858183,  0.48390999,  0.16443224,  0.47573758,  0.91038048,
					        0.86998783,  0.36185703,  0.74795191,  0.15545892,  0.18190848,
					        0.22547659,  0.9118953 ,  0.46280127,  0.37432461,  0.86560837,
					        0.04057165,  0.44815559,  0.99477211};
		cudaCALL(cudaMemcpy(ct_beta, arr_u, arr_len * sizeof(DATATYPE), cudaMemcpyHostToDevice));

		// invoke core crossover
		crossover_core(chromo1.rnvec, chromo2.rnvec, child1.rnvec, child2.rnvec,
						cf_distributionindex, ct_beta, arr_len);

		printGPUArray(child1.rnvec, arr_len);
		// arr size = 68: 0.403466 0.101732 0.737648 0.294592 0.836158 0.234459 0.846878 0.171008 0.298006 0.430689 0.124714 0.556058 0.157992 0.84443 0.160523 0.129101 0.528198 0.863656 0.205758 0.604315 0.113846 0.369361 0.993919 0.780806 1 0.295335 0.18548 0.359375 0.531562 0.492757 0.775506 1 0.133223 0.24951 0.235094 0.869209 0.147818 0.961776 0.737167 0.0282712 0.402408 0.190328 0.736663 0.990203 0.443277 0.014878 0.415969 0.590861 0.148936 1 0.398449 0.971654 0.781533 0.372514 0.886152 0.373074 0.241917 0.42521 0.775611 0.143869 0.835584 0.636126 0.0327814 0.264771 0 0.775094 0.910592 0.127992 
		printGPUArray(child2.rnvec, arr_len);
		// arr size = 68: 0.249123 0.13891 0.218122 0.594277 0.971796 0.576852 0.330708 0.740623 0.947255 0.482833 0.495159 0.408174 0.490169 0.552753 0.12612 1 0.525765 0.692102 0.423337 0.777509 0.233937 0.784782 0.943432 0.0541105 0.345725 0.0208376 0.701503 0.576783 0.084793 1 0.910565 0.192008 0.893206 0.0853548 0.614587 0.323779 0.0316312 0.17737 0.540543 0.0328236 0.832469 0.424765 0.548152 0.970023 0.859648 0.751929 0.725284 0.733266 0.494858 0.524081 0.775259 0.131443 0.944598 0.252874 0.051523 0.679137 0.53996 0.560037 0.852929 0.0288089 0.242221 0.0694763 0.545995 0.356145 0.80711 0.559885 0.0417296 1 
		printGPUArray(ct_beta, arr_len);
		// arr size = 68: 1.03884 1.49539 1.41295 0.431963 0.846996 1.36218 1.54364 0.991357 1.35241 0.47971 1.02878 0.955383 1.43033 0.559514 0.623754 1.66005 0.18408 0.961985 0.778199 0.965973 1.21934 0.770535 2.50991 1.26598 1.38434 1.17151 0.776683 1.9766 0.809628 2.33711 0.833217 1.2707 1.67337 0.876924 0.826122 0.966624 0.8913 1.47973 0.966355 0.580503 0.821964 1.06548 0.720732 0.766562 0.749431 1.09901 0.994016 0.405662 0.553102 1.46083 0.667317 0.989156 0.690249 0.983556 1.7736 1.56673 0.897819 1.2565 0.677457 0.713884 0.76685 1.78371 0.974559 0.908015 1.54953 0.43293 0.964168 4.57314 

		DATATYPE arr_rp[] = {0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
					        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
					        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
					        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
					        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
					        0.,  0.,  0.};
		cudaCALL(cudaMemcpy(rp, arr_rp, arr_len * sizeof(DATATYPE), cudaMemcpyHostToDevice));


		DATATYPE arr_u1[] = {0.61786588,  0.18873209,  0.82444858,  0.43792892,  0.23503938,
					         0.34822923,  0.263273  ,  0.14429133,  0.86678443,  0.56915517,
					         0.68617289,  0.02485217,  0.45544666,  0.72947819,  0.5768658 ,
					         0.34466833,  0.66388974,  0.42324134,  0.9217575 ,  0.56642512,
					         0.41470426,  0.60396907,  0.31263607,  0.83592878,  0.80653034,
					         0.6015999 ,  0.83106568,  0.83878437,  0.42235698,  0.73491831,
					         0.56392522,  0.14842095,  0.777232  ,  0.76445644,  0.43735819,
					         0.99534689,  0.52294566,  0.41651997,  0.66853754,  0.18547802,
					         0.57565362,  0.38573392,  0.90474716,  0.04674225,  0.11421379,
					         0.14436235,  0.42884963,  0.99504094,  0.9225646 ,  0.38500616,
					         0.80799155,  0.82864291,  0.20877516,  0.16569346,  0.88851709,
					         0.71090505,  0.56433767,  0.22336365,  0.95631976,  0.66641385,
					         0.35743709,  0.21469466,  0.55999368,  0.77508087,  0.3687653 ,
					         0.34517545,  0.13152812,  0.82776727};
		DATATYPE arr_r1[] = {0.1288757 ,  0.42444283,  0.32355118,  0.30899515,  0.76489928,
					         0.92304936,  0.95280478,  0.865248  ,  0.26529308,  0.52732468,
					         0.54688376,  0.02500307,  0.66367177,  0.32209382,  0.68983125,
					         0.27758723,  0.20079709,  0.12391166,  0.6376313 ,  0.75692633,
					         0.53629465,  0.55567406,  0.16234477,  0.91411046,  0.90449526,
					         0.20395669,  0.88282045,  0.59334981,  0.58700337,  0.31420989,
					         0.97107258,  0.38972054,  0.61331866,  0.19356732,  0.23627168,
					         0.95294961,  0.1737039 ,  0.5295848 ,  0.56773753,  0.19968627,
					         0.03258503,  0.47487036,  0.40930447,  0.51655992,  0.37803025,
					         0.81344729,  0.62561008,  0.05917355,  0.84484483,  0.52153103,
					         0.23962302,  0.93397556,  0.712624  ,  0.72820855,  0.08916658,
					         0.41105724,  0.05965412,  0.2250655 ,  0.68074111,  0.97084284,
					         0.45810756,  0.63911411,  0.77195825,  0.75309318,  0.93274807,
					         0.71994885,  0.37953216,  0.18143309};
		cudaCALL(cudaMemcpy(ct_beta, arr_u1, arr_len * sizeof(DATATYPE), cudaMemcpyHostToDevice));
		cudaCALL(cudaMemcpy(rp, arr_r1, arr_len * sizeof(DATATYPE), cudaMemcpyHostToDevice));

		// invoke core mutation
		mutate_core(child1.rnvec, child1.rnvec,
					mf_polynomialmutationindex, mf_mutationratio,
					ct_beta, rp, arr_len);

		printGPUArray(child1.rnvec, arr_len);
		// arr size = 68: 0.403466 0.101732 0.737648 0.294592 0.836158 0.234459 0.846878 0.171008 0.298006 0.430689 0.124714 0.556058 0.157992 0.84443 0.160523 0.129101 0.528198 0.863656 0.205758 0.604315 0.113846 0.369361 0.993919 0.780806 1 0.295335 0.18548 0.359375 0.531562 0.492757 0.775506 1 0.133223 0.24951 0.235094 0.869209 0.147818 0.961776 0.737167 0.0282712 0.402408 0.190328 0.736663 0.990203 0.443277 0.014878 0.415969 0.590861 0.148936 1 0.398449 0.971654 0.781533 0.372514 0.886152 0.373074 0.241917 0.42521 0.775611 0.143869 0.835584 0.636126 0.0327814 0.264771 0 0.775094 0.910592 0.127992 

		DATATYPE arr_u2[] = {0.02743752,  0.92723818,  0.34322354,  0.38793878,  0.45929728,
					         0.35494349,  0.29030592,  0.2077282 ,  0.0541821 ,  0.58753638,
					         0.58639038,  0.65927492,  0.77495742,  0.27016617,  0.56120907,
					         0.37179425,  0.75851729,  0.06467139,  0.71915049,  0.56690145,
					         0.65263028,  0.08730726,  0.73527886,  0.89233323,  0.62176899,
					         0.32027767,  0.3338375 ,  0.07013805,  0.69691006,  0.7251765 ,
					         0.29388861,  0.83777984,  0.54441692,  0.65642671,  0.61519253,
					         0.39915471,  0.59954461,  0.96440818,  0.34598173,  0.53287674,
					         0.81925966,  0.67621083,  0.54126379,  0.11702312,  0.83308993,
					         0.45976202,  0.92623169,  0.50449371,  0.41719599,  0.76937555,
					         0.7060828 ,  0.48168007,  0.05559654,  0.42459473,  0.04754664,
					         0.38851353,  0.82043645,  0.25942843,  0.85367487,  0.88367703,
					         0.60546339,  0.11277538,  0.34366063,  0.53895307,  0.19681922,
					         0.40853766,  0.2220928 ,  0.34073846};
		DATATYPE arr_r2[] = {0.40425712,  0.43869907,  0.71519338,  0.80298167,  0.48524693,
					         0.15157334,  0.32812411,  0.4343965 ,  0.0613061 ,  0.50573045,
					         0.59330276,  0.29510184,  0.06459244,  0.98540428,  0.84000914,
					         0.93623823,  0.99545353,  0.42704143,  0.59696053,  0.01428885,
					         0.39971292,  0.04746098,  0.37412239,  0.89970774,  0.15445334,
					         0.7159094 ,  0.69644072,  0.22031592,  0.74959002,  0.18572985,
					         0.31146056,  0.31517588,  0.72712095,  0.82913654,  0.1319611 ,
					         0.64081144,  0.53341464,  0.66069798,  0.06327807,  0.78349475,
					         0.773835  ,  0.00433484,  0.28293504,  0.14088591,  0.94066189,
					         0.60434295,  0.70605332,  0.95963021,  0.27341911,  0.27958768,
					         0.67859218,  0.61503153,  0.61365892,  0.05828667,  0.57755107,
					         0.59673562,  0.40686599,  0.39004087,  0.37963249,  0.04764278,
					         0.04780249,  0.39329155,  0.76440291,  0.12330675,  0.58748336,
					         0.06963835,  0.46054632,  0.25368624};
		cudaCALL(cudaMemcpy(ct_beta, arr_u2, arr_len * sizeof(DATATYPE), cudaMemcpyHostToDevice));
		cudaCALL(cudaMemcpy(rp, arr_r2, arr_len * sizeof(DATATYPE), cudaMemcpyHostToDevice));

		mutate_core(child2.rnvec, child2.rnvec,
					mf_polynomialmutationindex, mf_mutationratio,
					ct_beta, rp, arr_len);

		printGPUArray(child2.rnvec, arr_len);
		// arr size = 68: 0.249123 0.13891 0.218122 0.594277 0.971796 0.576852 0.330708 0.740623 0.947255 0.482833 0.495159 0.408174 0.490169 0.552753 0.12612 1 0.525765 0.692102 0.423337 0.782773 0.233937 0.784782 0.943432 0.0541105 0.345725 0.0208376 0.701503 0.576783 0.084793 1 0.910565 0.192008 0.893206 0.0853548 0.614587 0.323779 0.0316312 0.17737 0.540543 0.0328236 0.832469 0.464951 0.548152 0.970023 0.859648 0.751929 0.725284 0.733266 0.494858 0.524081 0.775259 0.131443 0.944598 0.252874 0.051523 0.679137 0.53996 0.560037 0.852929 0.0288089 0.242221 0.0694763 0.545995 0.356145 0.80711 0.559885 0.0417296 1 

		child1.skill_factor = chromo1.skill_factor;
		child2.skill_factor = chromo2.skill_factor;

		DATATYPE arr_swap[] = {0.86335564,  0.36834361,  0.85389856,  0.02573013,  0.44752639,
					        0.87542514,  0.8682124 ,  0.54806114,  0.88701455,  0.89737247,
					        0.05367614,  0.60358718,  0.72588557,  0.54748066,  0.95682063,
					        0.61976026,  0.13468317,  0.85574888,  0.64665441,  0.29429707,
					        0.92660739,  0.24301152,  0.91197534,  0.36638151,  0.72398239,
					        0.87119611,  0.98097662,  0.47081251,  0.62757963,  0.60200898,
					        0.36201012,  0.71184907,  0.51302716,  0.43023373,  0.43632189,
					        0.81552029,  0.82769663,  0.08061939,  0.57964562,  0.24211439,
					        0.05363023,  0.50664714,  0.0106789 ,  0.36384069,  0.30344904,
					        0.5274006 ,  0.17595282,  0.87409106,  0.8001164 ,  0.2290995 ,
					        0.99254275,  0.30784949,  0.03791029,  0.4611907 ,  0.0178598 ,
					        0.83138265,  0.48918022,  0.66021231,  0.58113956,  0.22394684,
					        0.54692974,  0.97539042,  0.02725502,  0.09440559,  0.29156004,
					        0.32864267,  0.36880155,  0.9205902};
		cudaCALL(cudaMemcpy(ct_beta, arr_swap, arr_len * sizeof(DATATYPE), cudaMemcpyHostToDevice));

		// invoke uniformcrossoverlike_core
		uniformcrossoverlike_core(child1.rnvec, child2.rnvec, ct_beta, arr_len);

		printGPUArray(child1.rnvec, arr_len);
		// arr size = 68: 0.249123 0.101732 0.218122 0.294592 0.836158 0.576852 0.330708 0.740623 0.947255 0.482833 0.124714 0.408174 0.490169 0.552753 0.12612 1 0.528198 0.692102 0.423337 0.604315 0.233937 0.369361 0.943432 0.780806 0.345725 0.0208376 0.701503 0.359375 0.084793 1 0.775506 0.192008 0.893206 0.24951 0.235094 0.323779 0.0316312 0.961776 0.540543 0.0282712 0.402408 0.464951 0.736663 0.990203 0.443277 0.751929 0.415969 0.733266 0.494858 1 0.775259 0.971654 0.781533 0.372514 0.886152 0.679137 0.241917 0.560037 0.852929 0.143869 0.242221 0.0694763 0.0327814 0.264771 0 0.775094 0.910592 1 
		printGPUArray(child2.rnvec, arr_len);
		// arr size = 68: 0.403466 0.13891 0.737648 0.594277 0.971796 0.234459 0.846878 0.171008 0.298006 0.430689 0.495159 0.556058 0.157992 0.84443 0.160523 0.129101 0.525765 0.863656 0.205758 0.782773 0.113846 0.784782 0.993919 0.0541105 1 0.295335 0.18548 0.576783 0.531562 0.492757 0.910565 1 0.133223 0.0853548 0.614587 0.869209 0.147818 0.17737 0.737167 0.0328236 0.832469 0.190328 0.548152 0.970023 0.859648 0.014878 0.725284 0.590861 0.148936 0.524081 0.398449 0.131443 0.944598 0.252874 0.051523 0.373074 0.53996 0.42521 0.775611 0.0288089 0.835584 0.636126 0.545995 0.356145 0.80711 0.559885 0.0417296 0.127992 
	}
};

struct MFEA_Chromosome_Print {
    void operator()(MFEA_Chromosome& chromo) {
		std::cout << chromo;
	}
};

struct MFEA_Chromosome_Randomize {
    MFEA_Chromosome_Randomize(const curandGenerator_t& curand_prng) : _curand_prng(curand_prng)  { }
    void operator()(MFEA_Chromosome& chromo) {
		curand_randomizeArray<DATATYPE>(_curand_prng, chromo.rnvec, getTotalLayerWeightsandBiases(), 1, WEIGHT_INIT_MIN, WEIGHT_INIT_MAX);
	}
private:
	const curandGenerator_t& _curand_prng;
};

#endif	// MFEA_CHROMOSOME_HPP
