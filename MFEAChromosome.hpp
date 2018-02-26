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

		uint32_t unrow = getMaximumNumberofUnitsofUnifiedLayer(layer);
		uint32_t uncol = getMaximumNumberofUnitsofUnifiedLayer(layer - 1);

		cublas_transposeMatrix<DATATYPE>(unrow, uncol, inp_rnvec + offset, W, cublas_handle);

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

					cudaDeviceSynchronize();
					printMatrix<DATATYPE>(w_nrow, w_ncol, mat_temp_w);
					printMatrix<DATATYPE>(1, b_size, mat_temp_rnvec + b_off);

					cublas_multiplyMatrices<DATATYPE>(training_size, b_size, 1,
														mat_one,
														mat_temp_rnvec + b_off,	// layer index started at 0
														mat_temp_layer[layer],	// layer index started at 0
														cublas_handle);
					cudaDeviceSynchronize();
					printMatrix<DATATYPE>(training_size, b_size, mat_temp_layer[layer]);

					// multiply add z[i] = z[i-1] * w[i-1] + b[i-1]
					cublas_multiplyandaddMatrices<DATATYPE>(training_size, b_size, w_nrow,
															mat_temp_layer[layer - 1],
															mat_temp_w,
															mat_temp_layer[layer],
															cublas_handle);
					cudaDeviceSynchronize();
					printMatrix<DATATYPE>(training_size, b_size, mat_temp_layer[layer]);

					// apply activation function default by sigmoid
					cuda_sigmoid<DATATYPE>(training_size, b_size, mat_temp_layer[layer], mat_temp_layer[layer]);
					cudaDeviceSynchronize();
					printMatrix<DATATYPE>(training_size, b_size, mat_temp_layer[layer]);
				}

				// decode w
				std::tuple<uint32_t, uint32_t> w_shape = this->decode(mat_temp_rnvec, mat_temp_w, task, numberof_layers, cublas_handle);
				uint32_t w_nrow = std::get<MATRIX_NROW>(w_shape), w_ncol = std::get<MATRIX_NCOL>(w_shape);

				// prepare b
				auto b_tup = getLayerBiasesbyTaskLayer(task, numberof_layers);
				uint32_t b_off = std::get<OFFSET_IDX>(b_tup), b_size = std::get<SIZE_IDX>(b_tup);

				cudaDeviceSynchronize();
				printMatrix<DATATYPE>(w_nrow, w_ncol, mat_temp_w);
				printMatrix<DATATYPE>(1, b_size, mat_temp_rnvec + b_off);

				cublas_multiplyMatrices<DATATYPE>(training_size, b_size, 1,
													mat_one,
													mat_temp_rnvec + b_off,	// layer index started at 0
													mat_temp_layer[numberof_layers],	// layer index started at 0
													cublas_handle);
				cudaDeviceSynchronize();
				printMatrix<DATATYPE>(training_size, b_size, mat_temp_layer[numberof_layers]);

				// multiply add z[i] = z[i-1] * w[i-1] + b[i-1]
				cublas_multiplyandaddMatrices<DATATYPE>(training_size, b_size, w_nrow,
														mat_temp_layer[numberof_layers - 1],
														mat_temp_w,
														mat_temp_layer[numberof_layers],
														cublas_handle);
				cudaDeviceSynchronize();
				printMatrix<DATATYPE>(training_size, b_size, mat_temp_layer[numberof_layers]);

				// apply activation function softmax to final layer
				cuda_sigmoid<DATATYPE>(training_size, b_size, mat_temp_layer[numberof_layers], mat_temp_layer[numberof_layers]);
				cudaDeviceSynchronize();
				printMatrix<DATATYPE>(training_size, b_size, mat_temp_layer[numberof_layers]);

				
				// eval cross entropy over the training_size
				BUG(getNumberofUnitsofLastLayerbyTask(task));
				printMatrix<DATATYPE>(training_size, getNumberofUnitsofLastLayerbyTask(task), Y);
				printMatrix<DATATYPE>(training_size, getNumberofUnitsofLastLayerbyTask(task), mat_temp_layer[numberof_layers]);
				factorial_costs[task] = cuda_evalMSE<DATATYPE>(training_size, getNumberofUnitsofLastLayerbyTask(task),
																		Y, mat_temp_layer[numberof_layers]);
				
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
		sbx_beta_transform<DATATYPE>(getTotalLayerWeightsandBiases(), ct_beta, cf_distributionindex);

		auto begin = thrust::make_zip_iterator(thrust::make_tuple(chromo1.rnvec,
																chromo2.rnvec,
																ct_beta,
																child1.rnvec,
																child2.rnvec));
		auto end = thrust::make_zip_iterator(thrust::make_tuple(chromo1.rnvec + getTotalLayerWeightsandBiases(),
																chromo2.rnvec + getTotalLayerWeightsandBiases(),
																ct_beta + getTotalLayerWeightsandBiases(),
																child1.rnvec + getTotalLayerWeightsandBiases(),
																child2.rnvec + getTotalLayerWeightsandBiases()));
		thrust::for_each(thrust::device, begin, end, __functor_sbx_children_generate<DATATYPE>());
	}

	friend void test_crossover(MFEA_Chromosome& chromo1, MFEA_Chromosome& chromo2,
						  MFEA_Chromosome& child1, MFEA_Chromosome& child2,
						  DATATYPE cf_distributionindex, DATATYPE* ct_beta,
						  const curandGenerator_t& prng) {
		DATATYPE arr[] = {0.46909909,  0.80761482,  0.15104835,  0.31183161,  0.77106882,
					        0.57776087,  0.30824062,  0.72808436,  0.8847482 ,  0.64604334,
					        0.18724646,  0.30891031,  0.18923413,  0.03636244,  0.55573386,
					        0.98752743,  0.97783515,  0.14333174,  0.62897381,  0.48999703,
					        0.15184541,  0.65403992,  0.43548032,  0.07369697,  0.51259634,
					        0.87620964,  0.42982082,  0.3332016 ,  0.95607483,  0.53540217,
					        0.18304035,  0.19920316,  0.87395232,  0.78342059,  0.53547685,
					        0.81776975,  0.64013451,  0.58255344,  0.92857015,  0.06491889,
					        0.25458728};
		// beta value must be in range [0, 1]
		curand_randomizeArray<DATATYPE>(prng, ct_beta, 1, getTotalLayerWeightsandBiases(), 0, 1);
		cudaCALL(cudaMemcpy(ct_beta, arr, getTotalLayerWeightsandBiases() * sizeof(DATATYPE), cudaMemcpyHostToDevice));

		sbx_beta_transform<DATATYPE>(getTotalLayerWeightsandBiases(), ct_beta, cf_distributionindex);



		auto begin = thrust::make_zip_iterator(thrust::make_tuple(chromo1.rnvec,
																chromo2.rnvec,
																ct_beta,
																child1.rnvec,
																child2.rnvec));
		auto end = thrust::make_zip_iterator(thrust::make_tuple(chromo1.rnvec + getTotalLayerWeightsandBiases(),
																chromo2.rnvec + getTotalLayerWeightsandBiases(),
																ct_beta + getTotalLayerWeightsandBiases(),
																child1.rnvec + getTotalLayerWeightsandBiases(),
																child2.rnvec + getTotalLayerWeightsandBiases()));
		thrust::for_each(thrust::device, begin, end, __functor_sbx_children_generate<DATATYPE>());

		cudaDeviceSynchronize();
		std::cout << "check beg" << std::endl;
		for (uint32_t i = 0; i < getTotalLayerWeightsandBiases(); ++i) {
			std::cout << child1.rnvec[i] << " ";
		} std::cout << std::endl;
		std::cout << "check end" << std::endl;
	}

	friend void mutate(MFEA_Chromosome& chromo, MFEA_Chromosome& child,
						DATATYPE mf_polynomialmutationindex, DATATYPE mf_mutationratio,
						DATATYPE* ct_beta, DATATYPE* rp, const curandGenerator_t& prng) {
		curand_randomizeArray<DATATYPE>(prng, ct_beta, 1, getTotalLayerWeightsandBiases(), 0, 1);

		auto begin = thrust::make_zip_iterator(thrust::make_tuple(chromo.rnvec,
																child.rnvec,
																ct_beta,
																rp,
																thrust::make_constant_iterator(mf_mutationratio),
																thrust::make_constant_iterator(mf_polynomialmutationindex)));
		auto end = thrust::make_zip_iterator(thrust::make_tuple(chromo.rnvec + getTotalLayerWeightsandBiases(),
																child.rnvec + getTotalLayerWeightsandBiases(),
																ct_beta + getTotalLayerWeightsandBiases(),
																rp + getTotalLayerWeightsandBiases(),
																thrust::make_constant_iterator(mf_mutationratio),
																thrust::make_constant_iterator(mf_polynomialmutationindex)));
		thrust::for_each(thrust::device, begin, end, __functor_pmu_children_generate<DATATYPE>());
	}

	friend void test_mutate(MFEA_Chromosome& chromo, MFEA_Chromosome& child,
						DATATYPE mf_polynomialmutationindex, DATATYPE mf_mutationratio,
						DATATYPE* ct_beta, DATATYPE* rp, const curandGenerator_t& prng) {
		DATATYPE arr[] = {0.46909909,  0.80761482,  0.15104835,  0.31183161,  0.77106882,
					        0.57776087,  0.30824062,  0.72808436,  0.8847482 ,  0.64604334,
					        0.18724646,  0.30891031,  0.18923413,  0.03636244,  0.55573386,
					        0.98752743,  0.97783515,  0.14333174,  0.62897381,  0.48999703,
					        0.15184541,  0.65403992,  0.43548032,  0.07369697,  0.51259634,
					        0.87620964,  0.42982082,  0.3332016 ,  0.95607483,  0.53540217,
					        0.18304035,  0.19920316,  0.87395232,  0.78342059,  0.53547685,
					        0.81776975,  0.64013451,  0.58255344,  0.92857015,  0.06491889,
					        0.25458728};
		DATATYPE ran[] = {0.85334729, 0.83041455, 0.91900871, 0.30592925, 0.14355824, 0.8070453, 
						0.31289032, 0.12356785, 0.55340493, 0.77444661, 0.62025491, 0.61190521, 
						0.58074571, 0.63782585, 0.41218446, 0.57232291, 0.09909049, 0.31150111, 
						0.33743135, 0.65770299, 0.26862436, 0.35846936, 0.90454419, 0.21924989, 
						0.50293893, 0.44367589, 0.84753395, 0.75304561, 0.84204609, 0.44031161, 
						0.36395839, 0.68240606, 0.29549063, 0.27371912, 0.71515266, 0.67363564, 
						0.91808274, 0.40140346, 0.94547614, 0.87794311, 0.05007344};

		curand_randomizeArray<DATATYPE>(prng, ct_beta, 1, getTotalLayerWeightsandBiases(), 0, 1);
		
		cudaCALL(cudaMemcpy(ct_beta, arr, getTotalLayerWeightsandBiases() * sizeof(DATATYPE), cudaMemcpyHostToDevice));
		cudaCALL(cudaMemcpy(rp, ran, getTotalLayerWeightsandBiases() * sizeof(DATATYPE), cudaMemcpyHostToDevice));

		auto begin = thrust::make_zip_iterator(thrust::make_tuple(chromo.rnvec,
																child.rnvec,
																ct_beta,
																rp,
																thrust::make_constant_iterator(mf_mutationratio),
																thrust::make_constant_iterator(mf_polynomialmutationindex)));
		auto end = thrust::make_zip_iterator(thrust::make_tuple(chromo.rnvec + getTotalLayerWeightsandBiases(),
																child.rnvec + getTotalLayerWeightsandBiases(),
																ct_beta + getTotalLayerWeightsandBiases(),
																rp + getTotalLayerWeightsandBiases(),
																thrust::make_constant_iterator(mf_mutationratio),
																thrust::make_constant_iterator(mf_polynomialmutationindex)));
		thrust::for_each(thrust::device, begin, end, __functor_pmu_children_generate<DATATYPE>());

		cudaDeviceSynchronize();
		std::cout << "check beg" << std::endl;
		for (uint32_t i = 0; i < getTotalLayerWeightsandBiases(); ++i) {
			std::cout << chromo.rnvec[i] << " ";
		} std::cout << std::endl;
		std::cout << "check end" << std::endl;
	}
	
	friend void uniformcrossoverlike(MFEA_Chromosome& child1, MFEA_Chromosome& child2,
										DATATYPE* ct_beta, const curandGenerator_t& prng) {
		curand_randomizeArray<DATATYPE>(prng, ct_beta, 1, getTotalLayerWeightsandBiases(), 0, 1);
		auto begin = thrust::make_zip_iterator(thrust::make_tuple(child1.rnvec,
																child2.rnvec,
																ct_beta));
		auto end = thrust::make_zip_iterator(thrust::make_tuple(child1.rnvec + getTotalLayerWeightsandBiases(),
																child2.rnvec + getTotalLayerWeightsandBiases(),
																ct_beta + getTotalLayerWeightsandBiases()));
		thrust::for_each(thrust::device, begin, end, __functor_ucl_childs_generate<DATATYPE>());
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
