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
#define WEIGHT_MIN		(1 * WEIGHT_INIT_MIN)
#define WEIGHT_MAX		(1 * WEIGHT_INIT_MAX)


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
		thrust::get<3>(tup) = 0.5 * ((1.0 + thrust::get<2>(tup)) * thrust::get<0>(tup) + (1.0 - thrust::get<2>(tup)) * thrust::get<1>(tup));
		thrust::get<4>(tup) = 0.5 * ((1.0 - thrust::get<2>(tup)) * thrust::get<0>(tup) + (1.0 + thrust::get<2>(tup)) * thrust::get<1>(tup));
	}
};
// simulated binary crossover algorithm - end

// polynomial mutation algorithm - beg
template<typename TYPE> struct __functor_pmu_children_generate {
};
template<> struct __functor_pmu_children_generate<float> {
	template <typename Tuple> __host__ __device__ void operator()(Tuple tup) {
		if (thrust::get<2>(tup) < thrust::get<3>(tup)) {	// rand < mutation_ratio
			if (thrust::get<2>(tup) / thrust::get<3>(tup) <= 0.5) {	// random binary option
				float del = powf(2 * thrust::get<2>(tup), 1.0 / (1.0 + thrust::get<4>(tup))) - 1.0;	// transform random array with polynomial mutation index
				thrust::get<1>(tup) = thrust::get<0>(tup) + del * thrust::get<0>(tup);				// update child weight itself
			} else {
				float del = 1.0 - powf(2 * (1.0 - thrust::get<2>(tup)), 1.0 / (1.0 + thrust::get<4>(tup)));	// transform random array with polynomial mutation index
				thrust::get<1>(tup) = thrust::get<0>(tup) + del * (1.0 - thrust::get<0>(tup));				// update child weight itself
			}
		} else {	// just copy from parent
			thrust::get<1>(tup) = thrust::get<0>(tup);
		}
	}
};
template<> struct __functor_pmu_children_generate<double> {
	template <typename Tuple> __host__ __device__ void operator()(Tuple tup) {
		if (thrust::get<2>(tup) < thrust::get<3>(tup)) {	// rand < mutation_ratio
			if (thrust::get<2>(tup) / thrust::get<3>(tup) <= 0.5) {	// random binary option
				double del = pow(2 * thrust::get<2>(tup), 1.0 / (1.0 + thrust::get<4>(tup))) - 1.0;	// transform random array with polynomial mutation index
				thrust::get<1>(tup) = thrust::get<0>(tup) + del * thrust::get<0>(tup);				// update child weight itself
			} else {
				double del = 1.0 - pow(2 * (1.0 - thrust::get<2>(tup)), 1.0 / (1.0 + thrust::get<4>(tup)));	// transform random array with polynomial mutation index
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
		if (thrust::get<2>(tup) <= 0.5) {	// random binary option
			// do swap
			TYPE temp = thrust::get<0>(tup);
			thrust::get<0>(tup) = thrust::get<1>(tup);
			thrust::get<1>(tup) = temp;
		} else {
			// remain, do nothing
		}
		
		// do transform weights i they are larger than WEIGHT_MAX of lower than WEIGHT_MIN
		if (thrust::get<0>(tup) > WEIGHT_MAX) {						// overDomain
			thrust::get<0>(tup) = thrust::get<2>(tup);				// replace by generated random number
		} else if (thrust::get<0>(tup) < WEIGHT_MIN) {				// underDomain
			 thrust::get<0>(tup) = thrust::get<2>(tup);				// replace by generated random number
		} else if (thrust::get<0>(tup) != thrust::get<0>(tup)) {	// is NAN
			thrust::get<0>(tup) = thrust::get<2>(tup);				// replace by generated random number
		} else {													// if valid, do nothing
		}
		if (thrust::get<1>(tup) > WEIGHT_MAX) {						// overDomain
			thrust::get<1>(tup) = WEIGHT_MAX-thrust::get<2>(tup);				// replace by generated random number
		} else if (thrust::get<1>(tup) < WEIGHT_MIN) {				// underDomain
			 thrust::get<1>(tup) = WEIGHT_MAX-thrust::get<2>(tup);				// replace by generated random number
		} else if (thrust::get<1>(tup) != thrust::get<1>(tup)) {	// is NAN
			thrust::get<1>(tup) = WEIGHT_MAX-thrust::get<2>(tup);				// replace by generated random number
		} else {													// if valid, do nothing
		}
	}
};
// uniform crossover like algorithm - end

template<typename TYPE> struct __functor_transformWeights {
	template <typename Tuple> __host__ __device__ float operator()(Tuple tup) {
		// thrust::get<0>(tup) is weight
		// thrust::get<1>(tup) is radomized vector for replacement
		if (thrust::get<0>(tup) > WEIGHT_MAX) {			// overDomain
			return thrust::get<1>(tup);					// return by replacement
		} else if (thrust::get<0>(tup) < WEIGHT_MIN) {	// underDomain
			return thrust::get<1>(tup);					// return by replacement
		} else if (thrust::get<0>(tup) != thrust::get<0>(tup)) {			// is NAN, then return 0
			return thrust::get<1>(tup);					// return by replacement
		} else {
			return thrust::get<0>(tup);					// is NOT NAN, then return x
		}
	}
};

template<typename TYPE> struct __functor_doSelfTransform {
	template <typename Tuple> __host__ __device__ void operator()(Tuple tup) {
		/*if (thrust::get<1>(tup) <= 0.05) {
			thrust::get<0>(tup) = (thrust::get<1>(tup) - 0.025)  / 0.025;
		}*/
		thrust::get<0>(tup) = thrust::get<0>(tup) / 5 + thrust::get<1>(tup);
	}
};

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
	DATATYPE* W[LAYER_SIZE];	// start using from 0, layer i + 1 stored in W[i] (i >= 0)

	
	/** Default constructor */
	MFEA_Chromosome() : scalar_fitness(std::numeric_limits<DATATYPE>::max()), skill_factor(std::numeric_limits<uint32_t>::max()) {
		//std::cout << "Default constructor" <<std::endl;
		for (uint32_t i = 0; i < TASK_SIZE; ++i) {
			factorial_costs[i] = std::numeric_limits<DATATYPE>::max();
			factorial_rank[i] = std::numeric_limits<uint32_t>::max();
			accuracy[i] = 0;
		}
		for (uint32_t i = 0; i < LAYER_SIZE; ++i) {
			cudaCALL(CUDA_M_MALLOC_MANAGED(W[i], DATATYPE, getMaximumLayerWeightsandBiasesbyLayer(i + 1)));
		}
	}
	
    /** Copy constructor */
    MFEA_Chromosome(const MFEA_Chromosome& other) : scalar_fitness(other.scalar_fitness), skill_factor(other.skill_factor),
													 factorial_costs(other.factorial_costs), factorial_rank(other.factorial_rank),
													 accuracy(other.accuracy) {
        //std::cout << "Copy constructor" << std::endl;
        for (uint32_t i = 0; i < LAYER_SIZE; ++i) {
			cudaCALL(CUDA_M_MALLOC_MANAGED(W[i], DATATYPE, getMaximumLayerWeightsandBiasesbyLayer(i + 1)));
			cudaCALL(cudaMemcpy(this->W[i], other.W[i], getMaximumLayerWeightsandBiasesbyLayer(i + 1) * sizeof(DATATYPE), cudaMemcpyDeviceToDevice));
		}
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

		for (uint32_t i = 0; i < LAYER_SIZE; ++i) {
			this->W[i] = other.W[i];
			other.W[i] = nullptr;
		}
    }
    
    /** Destructor - use default */
    ~MFEA_Chromosome() noexcept { /* explicitly specified destructors should be annotated noexcept as best-practice */
        //std::cout << "Destructor" << std::endl;
        for (uint32_t i = 0; i < LAYER_SIZE; ++i) {
			if (this->W[i] != nullptr) {
				//std::cout << "CUDA free : "; BUG(W[i]);
				cudaCALL(cudaFree(this->W[i]));
				this->W[i] = nullptr;
			}
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
        for (uint32_t i = 0; i < LAYER_SIZE; ++i) {
			cudaCALL(cudaMemcpy(this->W[i], other.W[i], getMaximumLayerWeightsandBiasesbyLayer(i + 1) * sizeof(DATATYPE), cudaMemcpyDeviceToDevice));
		}
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
        std::swap(W, other.W);
		
        return *this;
    }
    
    /**	ostream operator */
	friend std::ostream& operator<<(std::ostream& os, MFEA_Chromosome& chromo) {
		os << "Individual info:" << std::endl;
		for (uint32_t i = 0; i < LAYER_SIZE; ++i) {
			os << "Layer " << i << "(size = " << getMaximumLayerWeightsandBiasesbyLayer(i + 1) << "): ";
			for (uint32_t j = 0; j < getMaximumLayerWeightsandBiasesbyLayer(i + 1); ++j) {
				//os << chromo.W[i][j] << " ";
			}
			os << std::endl;
		}
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

	void evalObj(size_t training_size, size_t output_size, DATATYPE* X, DATATYPE* Y,
					DATATYPE* mat_temp_w, DATATYPE* mat_one, std::array<DATATYPE*, LAYER_SIZE> mat_temp_layer,
					cublasHandle_t& cublas_handle, cudnnHandle_t& cudnn_handle,
					bool is_evalAcc = false, bool is_alltaskeval = false, bool is_reupdateskillfactor = false) {
		DATATYPE factorial_cost_min = std::numeric_limits<DATATYPE>::max();
		for (uint32_t task = 0; task < TASK_SIZE; ++task) {
			if ((task == skill_factor) || is_alltaskeval) {
				uint32_t numberof_layers = getNumberofLayersbyTask(task);
				for (uint32_t layer = 1; layer < numberof_layers; ++layer) {
					// do weights transformation before evaluating
					thrust::transform(thrust::device, this->W[layer - 1], this->W[layer - 1] + getMaximumLayerWeightsandBiasesbyLayer(layer), mat_temp_w, __functor_weights_transform<float>(-5, 5));
					
					// prepare bias matrix from the last row (the last row of the largest matrix)
					auto biases = getLayerBiasesbyTaskLayer(task, layer);
					cublas_multiplyMatrices<DATATYPE>(training_size, getNumberofUnitsbyTaskLayer(task, layer), 1,
														mat_one,
														/*W[layer - 1]*/ mat_temp_w + std::get<OFFSET_IDX>(biases),	// layer index started at 0
														mat_temp_layer[layer - 1],						// layer index started at 0
														cublas_handle);
					// multiply add z[i] = z[i-1] * w[i-1] + b[i-1]
					cublas_multiplyandaddMatrices<DATATYPE>(training_size, getNumberofUnitsbyTaskLayer(task, layer), getNumberofUnitsbyTaskLayer(task, layer - 1),
															X,
															/*W[layer - 1]*/ mat_temp_w,
															mat_temp_layer[layer - 1],
															cublas_handle);
					// apply activation function default by sigmoid
					cuda_tanh<DATATYPE>(training_size, getNumberofUnitsbyTaskLayer(task, layer),
											mat_temp_layer[layer - 1], mat_temp_layer[layer - 1]);
				}
				// final layer applies softmax
				// prepare bias matrix from the last row (the last row of the largest matrix)
				auto biases = getLayerBiasesbyTaskLayer(task, numberof_layers);
				cublas_multiplyMatrices<DATATYPE>(training_size, getNumberofUnitsofLastLayerbyTask(task), 1,
													mat_one,
													/*W[numberof_layers - 1]*/ mat_temp_w + std::get<OFFSET_IDX>(biases),
													mat_temp_layer[numberof_layers - 1],
													cublas_handle);
				// multiply add z[i] = z[i-1] * w[i-1] + b[i-1]
				cublas_multiplyandaddMatrices<DATATYPE>(training_size, getNumberofUnitsofLastLayerbyTask(task), getNumberofUnitsbyTaskLayer(task, numberof_layers - 1),
														numberof_layers > 1 ? mat_temp_layer[numberof_layers - 2] : X,
														/*W[numberof_layers - 1]*/ mat_temp_w,
														mat_temp_layer[numberof_layers - 1],
														cublas_handle);
				// apply activation function softmax to final layer
				cuda_sigmoid<DATATYPE>(training_size, getNumberofUnitsofLastLayerbyTask(task),
										mat_temp_layer[numberof_layers - 1], mat_temp_layer[numberof_layers - 1]);
				
				// eval cross entropy over the training_size
				factorial_costs[task] = cuda_evalMSE<DATATYPE>(training_size, getNumberofUnitsofLastLayerbyTask(task),
																		Y, mat_temp_layer[numberof_layers - 1]);
				
				// update skill factor
				if ((factorial_costs[task] < factorial_cost_min) && is_reupdateskillfactor) {
					skill_factor = task;
					factorial_cost_min = factorial_costs[task];
				}
				if (is_evalAcc) {
					accuracy[task] = cuda_evalAccuracy<DATATYPE>(training_size, getNumberofUnitsofLastLayerbyTask(task),
															Y, mat_temp_layer[numberof_layers - 1]);
				}
			} else {
				factorial_costs[task] = std::numeric_limits<DATATYPE>::max();
			}
		}
	}
	
/*	void evalObj_cpu(size_t training_size, size_t output_size, DATATYPE* X, DATATYPE* Y, uint8_t* Ylabel,
					DATATYPE* mat_one, std::array<DATATYPE*, LAYER_SIZE> mat_temp_layer,
					cublasHandle_t& cublas_handle, cudnnHandle_t& cudnn_handle) {
		DATATYPE factorial_cost_min = std::numeric_limits<DATATYPE>::max();
		for (uint32_t task = 0; task < TASK_SIZE; ++task) {
			for (uint32_t layer = 0; layer < LAYER_SIZE - 1; ++layer) {
				// prepare bias matrix from the last row (the last row of the largest matrix)
				multiplyMatrices<DATATYPE>(training_size, getNumberofUnitssbyTaskLayer(task, layer), 1, mat_one, W[layer] + TASK_LAYERSIZES[TASKINDEX_LARGEST][layer] * TASK_LAYERSIZES[TASKINDEX_LARGEST][layer + 1], mat_temp_layer[layer]);
				// multiply add z[i] = z[i-1] * w[i-1] + b[i-1]
				multiplyandaddMatrices<DATATYPE>(training_size, getNumberofUnitssbyTaskLayer(task, layer), TASK_LAYERSIZES[task][layer], X, W[layer], mat_temp_layer[layer]);
				
				// apply activation function default by sigmoid
				sigmoid<DATATYPE>(training_size, getNumberofUnitssbyTaskLayer(task, layer), mat_temp_layer[layer], mat_temp_layer[layer]);
			}
			// final layer applies softmax
			// prepare bias matrix from the last row (the last row of the largest matrix)
			multiplyMatrices<DATATYPE>(training_size, TASK_LAYERSIZES[task][LAYER_SIZE], 1, mat_one, W[LAYER_SIZE - 1] + TASK_LAYERSIZES[TASKINDEX_LARGEST][LAYER_SIZE - 1] * TASK_LAYERSIZES[TASKINDEX_LARGEST][LAYER_SIZE], mat_temp_layer[LAYER_SIZE - 1]);
			// multiply add z[i] = z[i-1] * w[i-1] + b[i-1]
			multiplyandaddMatrices<DATATYPE>(training_size, TASK_LAYERSIZES[task][LAYER_SIZE], TASK_LAYERSIZES[task][LAYER_SIZE - 1], LAYER_SIZE > 1 ? mat_temp_layer[LAYER_SIZE - 2] : X, W[LAYER_SIZE - 1], mat_temp_layer[LAYER_SIZE - 1]);
			
			// apply activation function softmax to final layer
			softmax<DATATYPE>(training_size, TASK_LAYERSIZES[task][LAYER_SIZE], mat_temp_layer[LAYER_SIZE - 1], mat_temp_layer[LAYER_SIZE - 1]);
			
			// eval cross entropy over the training_size
			factorial_costs[task] = evalCrossEntropy<DATATYPE>(training_size, TASK_LAYERSIZES[task][LAYER_SIZE], Y, mat_temp_layer[LAYER_SIZE - 1]);
			
			// update skill factor
			if (factorial_costs[task] < factorial_cost_min) {
				skill_factor = task;
				factorial_cost_min = factorial_costs[task];
			}
		}
	}*/
	
	friend void crossover(MFEA_Chromosome& chromo1, MFEA_Chromosome& chromo2, MFEA_Chromosome& child1, MFEA_Chromosome& child2, DATATYPE cf_distributionindex, DATATYPE* ct_beta, const curandGenerator_t& prng) {
		for (uint32_t i = 0; i < LAYER_SIZE; ++i) {
			// beta value must be in range [0, 1]
			curand_randomizeArray<DATATYPE>(prng, ct_beta, 1, getMaximumLayerWeightsandBiasesbyLayer(i + 1), 0, 1);
			sbx_beta_transform<DATATYPE>(getMaximumLayerWeightsandBiasesbyLayer(i + 1), ct_beta, cf_distributionindex);
			auto begin = thrust::make_zip_iterator(thrust::make_tuple(chromo1.W[i],
																	chromo2.W[i],
																	ct_beta,
																	child1.W[i],
																	child2.W[i]));
			auto end = thrust::make_zip_iterator(thrust::make_tuple(chromo1.W[i] + getMaximumLayerWeightsandBiasesbyLayer(i + 1),
																	chromo2.W[i] + getMaximumLayerWeightsandBiasesbyLayer(i + 1),
																	ct_beta + getMaximumLayerWeightsandBiasesbyLayer(i + 1),
																	child1.W[i] + getMaximumLayerWeightsandBiasesbyLayer(i + 1),
																	child2.W[i] + getMaximumLayerWeightsandBiasesbyLayer(i + 1)));
			thrust::for_each(thrust::device, begin, end, __functor_sbx_children_generate<DATATYPE>());
			
		}
	}

	friend void mutate(MFEA_Chromosome& chromo, MFEA_Chromosome& child, DATATYPE mf_polynomialmutationindex, DATATYPE mf_mutationratio, DATATYPE* ct_beta, const curandGenerator_t& prng) {
		for (uint32_t i = 0; i < LAYER_SIZE; ++i) {
			curand_randomizeArray<DATATYPE>(prng, ct_beta, 1, getMaximumLayerWeightsandBiasesbyLayer(i + 1), 0, 1);
			auto begin = thrust::make_zip_iterator(thrust::make_tuple(chromo.W[i],
																	child.W[i],
																	ct_beta,
																	thrust::make_constant_iterator(mf_mutationratio),
																	thrust::make_constant_iterator(mf_polynomialmutationindex)));
			auto end = thrust::make_zip_iterator(thrust::make_tuple(chromo.W[i] + getMaximumLayerWeightsandBiasesbyLayer(i + 1),
																	child.W[i] + getMaximumLayerWeightsandBiasesbyLayer(i + 1),
																	ct_beta + getMaximumLayerWeightsandBiasesbyLayer(i + 1),
																	thrust::make_constant_iterator(mf_mutationratio),
																	thrust::make_constant_iterator(mf_polynomialmutationindex)));
			thrust::for_each(thrust::device, begin, end, __functor_pmu_children_generate<DATATYPE>());
		}
	}
	
	friend void uniformcrossoverlike(MFEA_Chromosome& child1, MFEA_Chromosome& child2, DATATYPE* ct_beta, const curandGenerator_t& prng) {
		for (uint32_t i = 0; i < LAYER_SIZE; ++i) {
			curand_randomizeArray<DATATYPE>(prng, ct_beta, 1, getMaximumLayerWeightsandBiasesbyLayer(i + 1), 0, 1);
			auto begin = thrust::make_zip_iterator(thrust::make_tuple(child1.W[i],
																	child2.W[i],
																	ct_beta));
			auto end = thrust::make_zip_iterator(thrust::make_tuple(child1.W[i] + getMaximumLayerWeightsandBiasesbyLayer(i + 1),
																	child2.W[i] + getMaximumLayerWeightsandBiasesbyLayer(i + 1),
																	ct_beta + getMaximumLayerWeightsandBiasesbyLayer(i + 1)));
			thrust::for_each(thrust::device, begin, end, __functor_ucl_childs_generate<DATATYPE>());
		}
	}

	friend void transformWeights(MFEA_Chromosome& chromo, DATATYPE* ct_beta, const curandGenerator_t& prng) {
		for (uint32_t i = 0; i < LAYER_SIZE; ++i) {
			curand_randomizeArray<DATATYPE>(prng, ct_beta, 1, getMaximumLayerWeightsandBiasesbyLayer(i + 1), WEIGHT_MIN, WEIGHT_MAX);
			auto begin = thrust::make_zip_iterator(thrust::make_tuple(chromo.W[i],
																	ct_beta));
			auto end = thrust::make_zip_iterator(thrust::make_tuple(chromo.W[i] + getMaximumLayerWeightsandBiasesbyLayer(i + 1),
																	ct_beta + getMaximumLayerWeightsandBiasesbyLayer(i + 1)));
			thrust::for_each(thrust::device, begin, end, __functor_transformWeights<DATATYPE>());
		}
	}
	
	
	template<typename TYPE> struct __functor_examineCrossover {
		template <typename Tuple> __host__ __device__ bool operator()(Tuple tup) {
			return (thrust::get<0>(tup) + thrust::get<1>(tup)) - (thrust::get<2>(tup) + thrust::get<3>(tup)) > 0.0000001;
		}
	};
	friend void examineCrossover(MFEA_Chromosome& chromo1, MFEA_Chromosome& chromo2, MFEA_Chromosome& child1, MFEA_Chromosome& child2) {
		std::cout << "------- BEGIN examining crossover operator -------" << std::endl;
		for (uint32_t i = 0; i < LAYER_SIZE; ++i) {
			std::cout << "--- Layer " << i <<  " (size = " <<  getMaximumLayerWeightsandBiasesbyLayer(i + 1) << "):" << std::endl;
			auto begin = thrust::make_zip_iterator(thrust::make_tuple(chromo1.W[i],
																	chromo2.W[i],
																	child1.W[i],
																	child2.W[i]));
			auto end = thrust::make_zip_iterator(thrust::make_tuple(chromo1.W[i] + getMaximumLayerWeightsandBiasesbyLayer(i + 1),
																	chromo2.W[i] + getMaximumLayerWeightsandBiasesbyLayer(i + 1),
																	child1.W[i] + getMaximumLayerWeightsandBiasesbyLayer(i + 1),
																	child2.W[i] + getMaximumLayerWeightsandBiasesbyLayer(i + 1)));
			uint32_t count_error = thrust::count_if(thrust::device, begin, end, __functor_examineCrossover<DATATYPE>());
			std::cout << "Error count = " << count_error << std::endl;
		}
		std::cout << "------- END examine crossover operator -------" << std::endl;
	}
	friend void examineIndividual(MFEA_Chromosome& chromo) {
		std::cout << "------- BEGIN examining individual weights -------" << std::endl;
		for (uint32_t i = 0; i < LAYER_SIZE; ++i) {
			std::cout << "--- Layer " << i <<  " (size = " <<  getMaximumLayerWeightsandBiasesbyLayer(i + 1) << "):" << std::endl;
			std::cout << "--- Layer " << i <<  " (size = " <<  (TASK_LAYERSIZES[TASKINDEX_LARGEST][i] + 1) * TASK_LAYERSIZES[TASKINDEX_LARGEST][i + 1] << "):" << std::endl;
			examineMatrix<DATATYPE>(TASK_LAYERSIZES[TASKINDEX_LARGEST][i] + 1, TASK_LAYERSIZES[TASKINDEX_LARGEST][i + 1], chromo.W[i]);
		}
		std::cout << "------- END examine individual weights -------" << std::endl;
	}
	
	void exportToFile(std::string filename) {
		std::ofstream ofile(filename, std::ofstream::binary);
		for (uint32_t i = 0; i < LAYER_SIZE; ++i) {
			ofile.write((char*)W[i], getMaximumLayerWeightsandBiasesbyLayer(i + 1) * sizeof(DATATYPE));
		}
		ofile.close();
	}
	void loadFromFile(std::string filename) {
		std::ifstream ifile(filename, std::ifstream::binary);
		for (uint32_t i = 0; i < LAYER_SIZE; ++i) {
			ifile.read((char*)W[i], getMaximumLayerWeightsandBiasesbyLayer(i + 1) * sizeof(DATATYPE));
		}
		ifile.close();
	}
	
	void doSelfTransform(DATATYPE* ct_beta, const curandGenerator_t& prng) {
		for (uint32_t i = 0; i < LAYER_SIZE; ++i) {
			curand_randomizeArray<DATATYPE>(prng, ct_beta, 1, getMaximumLayerWeightsandBiasesbyLayer(i + 1), 0, 1);
			auto begin = thrust::make_zip_iterator(thrust::make_tuple(W[i],
																	ct_beta));
			auto end = thrust::make_zip_iterator(thrust::make_tuple(W[i] + getMaximumLayerWeightsandBiasesbyLayer(i + 1),
																	ct_beta + getMaximumLayerWeightsandBiasesbyLayer(i + 1)));
			thrust::for_each(thrust::device, begin, end, __functor_doSelfTransform<DATATYPE>());
		}
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
		for (uint32_t i = 0; i < LAYER_SIZE; ++i) {
			curand_randomizeArray<DATATYPE>(_curand_prng, chromo.W[i], getMaximumLayerWeightsandBiasesbyLayer(i + 1), 1, WEIGHT_INIT_MIN, WEIGHT_INIT_MAX);
		}
	}
private:
	const curandGenerator_t& _curand_prng;
};

#endif	// MFEA_CHROMOSOME_HPP
