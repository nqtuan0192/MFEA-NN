#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <random>

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#define MATRIX_NROW	0
#define MATRIX_NCOL	1

#define DOMAIN_MAX_VALUE 1.0
#define DOMAIN_MIN_VALUE -1.0

template<typename TYPE> void printMatrix(size_t nrow, size_t ncol, TYPE* mat_in) {
	std::cout << "matrix size = " << nrow << " x " << ncol << ":" << std::endl;
	for (uint32_t i = 0; i < nrow; ++i) {
		for (uint32_t j = 0; j < ncol; ++j) {
			std::cout << mat_in[i * ncol + j] << "\t";
		}
		std::cout << std::endl;
	}
}

template<typename TYPE> struct __functor_isNAN {
	__host__ __device__ bool operator()(const TYPE& x) {
		return (x != x);
	}
};
template<typename TYPE> struct __functor_isOverDomain {
	__host__ __device__ bool operator()(const TYPE& x) {
		return (x > DOMAIN_MAX_VALUE);
	}
};
template<typename TYPE> struct __functor_isUnderDomain {
	__host__ __device__ bool operator()(const TYPE& x) {
		return (x < DOMAIN_MIN_VALUE);
	}
};
template<typename TYPE> void examineMatrix(size_t nrow, size_t ncol, TYPE* mat_in) {
	uint32_t count_zero = thrust::count(thrust::device , mat_in, mat_in + nrow * ncol, 0.0);
	uint32_t count_max_domain = thrust::count(thrust::device , mat_in, mat_in + nrow * ncol, DOMAIN_MAX_VALUE);
	uint32_t count_min_domain = thrust::count(thrust::device , mat_in, mat_in + nrow * ncol, DOMAIN_MIN_VALUE);
	uint32_t count_nan = thrust::count_if(thrust::device , mat_in, mat_in + nrow * ncol, __functor_isNAN<TYPE>());
	uint32_t count_overdomain = thrust::count_if(thrust::device , mat_in, mat_in + nrow * ncol, __functor_isOverDomain<TYPE>());
	uint32_t count_underdomain = thrust::count_if(thrust::device , mat_in, mat_in + nrow * ncol, __functor_isUnderDomain<TYPE>());
	std::cout << "Matrix size = " << nrow << " x " << ncol << std::endl;
	std::cout << "Memory pointer begin = " << mat_in << std::endl;
	std::cout << "Memory size = " << nrow * ncol * sizeof(TYPE) << std::endl;
	BUG(count_zero);
	BUG(count_max_domain);
	BUG(count_min_domain);
	BUG(count_nan);
	BUG(count_overdomain);
	BUG(count_underdomain);
	if (count_nan > 0) std::cout << "FUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCK";
}

template<typename TYPE> struct __functor_randomize {
	__functor_randomize(TYPE min_val, TYPE max_val) : _min_val(min_val), _max_val(max_val) {
	}
	__host__ __device__ TYPE operator()(const TYPE& x) {
		return x * (_max_val - _min_val) + _min_val;
	}
private:
	TYPE _min_val;
	TYPE _max_val;
};
template<typename TYPE> void curand_randomizeArray(const curandGenerator_t& prng, TYPE* dev_ptr, size_t nrow, size_t ncol, TYPE min_val, TYPE max_val) {
	std::cerr << "Error! Only support uint32, uint64, float or double." << std::endl;
}
template<> void curand_randomizeArray(const curandGenerator_t& prng, uint32_t* dev_ptr, size_t nrow, size_t ncol, uint32_t min_val, uint32_t max_val) {
    size_t size = nrow * ncol;
    curandCALL(curandGenerate(prng, dev_ptr, size));
    thrust::transform(thrust::device, dev_ptr, dev_ptr + size, dev_ptr, __functor_randomize<uint32_t>(min_val, max_val));
}
template<> void curand_randomizeArray(const curandGenerator_t& prng, uint64_t* dev_ptr, size_t nrow, size_t ncol, uint64_t min_val, uint64_t max_val) {
    size_t size = nrow * ncol;
    curandCALL(curandGenerateLongLong(prng, (unsigned long long*)dev_ptr, size));
    thrust::transform(thrust::device, dev_ptr, dev_ptr + size, dev_ptr, __functor_randomize<uint64_t>(min_val, max_val));
}
template<> void curand_randomizeArray(const curandGenerator_t& prng, float* dev_ptr, size_t nrow, size_t ncol, float min_val, float max_val) {
    size_t size = nrow * ncol;
    curandCALL(curandGenerateUniform(prng, dev_ptr, size));
    thrust::transform(thrust::device, dev_ptr, dev_ptr + size, dev_ptr, __functor_randomize<float>(min_val, max_val));
}
template<> void curand_randomizeArray(const curandGenerator_t& prng, double* dev_ptr, size_t nrow, size_t ncol, double min_val, double max_val) {
    size_t size = nrow * ncol;
    curandCALL(curandGenerateUniformDouble(prng, dev_ptr, size));
    thrust::transform(thrust::device, dev_ptr, dev_ptr + size, dev_ptr, __functor_randomize<double>(min_val, max_val));
}


template<typename TYPE> void cuda_fillMatrix(size_t nrow, size_t ncol, TYPE* dev_ptr, TYPE value) {
	thrust::fill(thrust::device, thrust::device_pointer_cast<TYPE>(dev_ptr), thrust::device_pointer_cast<TYPE>(dev_ptr) + nrow * ncol, value);
}

// in out pointers could NOT be same
template<typename TYPE> void multiplyMatrices(size_t nrow, size_t ncol, size_t commonsize, TYPE* mat_a, TYPE* mat_b, TYPE* mat_out) {
	for (uint32_t i = 0; i < nrow; ++i) {
		for (uint32_t j = 0; j < ncol; ++j) {
			mat_out[i * ncol + j] = 0;
			for (uint32_t k = 0; k < commonsize; ++k) {
				mat_out[i * ncol + j] += mat_a[i * commonsize + k] * mat_b[k * ncol + j];
			}
		}
	}
}
// in out pointers could NOT be same
template<typename TYPE> void multiplyandaddMatrices(size_t nrow, size_t ncol, size_t commonsize, TYPE* mat_a, TYPE* mat_b, TYPE* mat_out) {
	for (uint32_t i = 0; i < nrow; ++i) {
		for (uint32_t j = 0; j < ncol; ++j) {
			for (uint32_t k = 0; k < commonsize; ++k) {
				mat_out[i * ncol + j] += mat_a[i * commonsize + k] * mat_b[k * ncol + j];
			}
		}
	}
}

// not efficient implementation but easy to debugging
template<typename TYPE> __global__ void cuda_multiplyMatrices(size_t nrow, size_t ncol, size_t commonsize, TYPE* mat_a, TYPE* mat_b, TYPE* mat_out) {
	int i = blockIdx.x, j = threadIdx.x;
	mat_out[i * ncol + j] = 0;
	for (int k = 0; k < commonsize; ++k) {
		mat_out[i * ncol + j] =  mat_out[i * ncol + j] + mat_a[i * commonsize + k] * mat_b[k * ncol + j];
	}
}

// using high efficient implementation from CUBLAS library
template<typename TYPE> void cublas_multiplyMatrices(size_t nrow, size_t ncol, size_t commonsize, TYPE* mat_a, TYPE* mat_b, TYPE* mat_out, cublasHandle_t& handle) {
	std::cerr << "Error! Only support float or double." << std::endl;
}
template<> void cublas_multiplyMatrices<float>(size_t nrow, size_t ncol, size_t commonsize, float* mat_a, float* mat_b, float* mat_out, cublasHandle_t& handle) {
	float alpha = 1.0;
	float beta = 0.0;
	cublasCALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ncol, nrow, commonsize, &alpha, mat_b, ncol, mat_a, commonsize, &beta, mat_out, ncol));
}
template<> void cublas_multiplyMatrices<double>(size_t nrow, size_t ncol, size_t commonsize, double* mat_a, double* mat_b, double* mat_out, cublasHandle_t& handle) {
	double alpha = 1.0;
	double beta = 0.0;
	cublasCALL(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ncol, nrow, commonsize, &alpha, mat_b, ncol, mat_a, commonsize, &beta, mat_out, ncol));
}

template<typename TYPE> void cublas_multiplyandaddMatrices(size_t nrow, size_t ncol, size_t commonsize, TYPE* mat_a, TYPE* mat_b, TYPE* mat_out, cublasHandle_t& handle) {
	std::cerr << "Error! Only support float or double." << std::endl;
}
template<> void cublas_multiplyandaddMatrices<float>(size_t nrow, size_t ncol, size_t commonsize, float* mat_a, float* mat_b, float* mat_out, cublasHandle_t& handle) {
	float alpha = 1.0;
	float beta = 1.0;
	cublasCALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ncol, nrow, commonsize, &alpha, mat_b, ncol, mat_a, commonsize, &beta, mat_out, ncol));
}
template<> void cublas_multiplyandaddMatrices<double>(size_t nrow, size_t ncol, size_t commonsize, double* mat_a, double* mat_b, double* mat_out, cublasHandle_t& handle) {
	double alpha = 1.0;
	double beta = 1.0;
	cublasCALL(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ncol, nrow, commonsize, &alpha, mat_b, ncol, mat_a, commonsize, &beta, mat_out, ncol));
}

template<typename TYPE> void cublas_addMatrices(size_t nrow, size_t ncol, TYPE* mat_a, TYPE* mat_b, TYPE* mat_out, cublasHandle_t& handle) {
	std::cerr << "Error! Only support float or double." << std::endl;
}
template<> void cublas_addMatrices<float>(size_t nrow, size_t ncol, float* mat_a, float* mat_b, float* mat_out, cublasHandle_t& handle) {
	float alpha = 1.0;
	float beta = 1.0;
	cublasCALL(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, nrow, ncol, &alpha, mat_a, nrow, &beta, mat_b, nrow, mat_out, nrow));
}
template<> void cublas_addMatrices<double>(size_t nrow, size_t ncol, double* mat_a, double* mat_b, double* mat_out, cublasHandle_t& handle) {
	double alpha = 1.0;
	double beta = 1.0;
	cublasCALL(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, nrow, ncol, &alpha, mat_a, nrow, &beta, mat_b, nrow, mat_out, nrow));
}

template<typename TYPE> void cublas_transposeMatrix(size_t nrow, size_t ncol, TYPE* mat_in, TYPE* mat_out, cublasHandle_t& handle) {
	std::cerr << "Error! Only support float or double." << std::endl;
}
template<> void cublas_transposeMatrix<float>(size_t nrow, size_t ncol, float* mat_in, float* mat_out, cublasHandle_t& handle) {
	float alpha = 1.0;
	float beta = 0.0;
	cublasCALL(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, nrow, ncol, &alpha, mat_in, ncol, &beta, NULL, nrow, mat_out, nrow));
}
template<> void cublas_transposeMatrix<double>(size_t nrow, size_t ncol, double* mat_in, double* mat_out, cublasHandle_t& handle) {
	double alpha = 1.0;
	double beta = 0.0;
	cublasCALL(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, nrow, ncol, &alpha, mat_in, ncol, &beta, NULL, nrow, mat_out, nrow));
}

template<typename TYPE> void cublas_copyMatrix(size_t nrow, size_t ncol, TYPE* mat_in, TYPE* mat_out, cublasHandle_t& handle) {
	std::cerr << "Error! Only support float or double." << std::endl;
}
template<> void cublas_copyMatrix<float>(size_t nrow, size_t ncol, float* mat_in, float* mat_out, cublasHandle_t& handle) {
	cublasCALL(cublasScopy(handle, nrow * ncol, mat_in, 1, mat_out, 1));
}
template<> void cublas_copyMatrix<double>(size_t nrow, size_t ncol, double* mat_in, double* mat_out, cublasHandle_t& handle) {
	cublasCALL(cublasDcopy(handle, nrow * ncol, mat_in, 1, mat_out, 1));
}

// Copy matrix mat_in with shape = in_nrow x in_ncol from starting point at in_srow, in_scol to matrix mat_out with shape out_nrow x out_nrow
// inplace possible
template<typename TYPE> void cuda_copySubMatrix(uint32_t in_srow, uint32_t in_scol, TYPE* mat_in, uint32_t in_nrow, uint32_t in_ncol, TYPE* mat_out, uint32_t out_nrow, uint32_t out_ncol) {
	cudaMemcpy2D(mat_out, out_ncol * sizeof(TYPE), mat_in + in_scol + in_srow * in_ncol, in_ncol * sizeof(TYPE), out_ncol * sizeof(TYPE), out_nrow, cudaMemcpyDeviceToDevice);
}

// in out pointers could NOT be same
template<typename TYPE> void addColumn(size_t nrow, size_t ncol, TYPE* mat_in, TYPE* mat_out, TYPE value) {
	//#pragma unroll
	for (uint32_t i = 0; i < nrow; ++i) {
		//#pragma unroll
		for (uint32_t j = 0; j < ncol; ++j) {
			mat_out[i * (ncol + 1) + j] =  mat_in[i * ncol + j];
		}
		mat_out[i * (ncol + 1) + ncol] = value;
	}
}

// in out pointers could NOT be same
// transform matrix a by adding a new column of 1 before multiplication
template<typename TYPE> void multiplyMatricesAndAddedBiasColumn(size_t nrow, size_t ncol, size_t commonsize, TYPE* mat_a, TYPE* mat_b, TYPE* mat_out) {
	//#pragma unroll
	for (uint32_t i = 0; i < nrow; ++i) {
		//#pragma unroll
		for (uint32_t j = 0; j < ncol; ++j) {
			mat_out[i * ncol + j] = 0;
			//#pragma unroll
			for (uint32_t k = 0; k < commonsize - 1; ++k) {
				mat_out[i * ncol + j] += mat_a[i * (commonsize - 1) + k] * mat_b[k * ncol + j];
			}
			mat_out[i * ncol + j] += mat_b[(commonsize - 1) * ncol + j];
		}
	}
}

// in out pointer COULD be same
template<typename TYPE> void softmax(size_t nrow, size_t ncol, TYPE* mat_in, TYPE* mat_out) {
	//#pragma unroll
	for (uint32_t i = 0; i < nrow; ++i) {
		// find max
		TYPE max = mat_in[i * ncol + 0];
		//#pragma unroll
		for (uint32_t j = 1; j < ncol; ++j) {
			if (mat_in[i * ncol + j] > max) {
				max = mat_in[i * ncol + j];
			}
		}
		
		// calculate sum exp
		TYPE sum = 0;
		//#pragma unroll
		for (uint32_t j = 0; j < ncol; ++j) {
			mat_out[i * ncol + j] = expf(mat_in[i * ncol + j] - max);
			sum += mat_out[i * ncol + j];
		}
		
		// calculate each value
		//#pragma unroll
		for (uint32_t j = 0; j < ncol; ++j) {
			mat_out[i * ncol + j] = mat_out[i * ncol + j] * 1.0 / sum;
			// prevent numeric error
			if (mat_out[i * ncol + j] == 0) {
				mat_out[i * ncol + j] = std::numeric_limits<float>::min();
			}
		}
	}
}

// in out pointer COULD be same
template<typename TYPE> void sigmoid(size_t nrow, size_t ncol, TYPE* mat_in, TYPE* mat_out) {
	//#pragma unroll
	for (uint32_t i = 0; i < nrow; ++i) {
		//#pragma unroll
		for (uint32_t j = 0; j < ncol; ++j) {
			mat_out[i * ncol + j] = 1.0 / (1.0 + std::exp(-mat_in[i * ncol + j]));
		}
	}
}

// in out pointer COULD be same
template<typename TYPE> void relu(size_t nrow, size_t ncol, TYPE* mat_in, TYPE* mat_out) {
	//#pragma unroll
	for (uint32_t i = 0; i < nrow; ++i) {
		//#pragma unroll
		for (uint32_t j = 0; j < ncol; ++j) {
			mat_out[i * ncol + j] = MAX2(0.0, mat_in[i * ncol + j]);
		}
	}
}

// in out pointer COULD be same
template<typename TYPE> void cudnn_softmax(size_t nrow, size_t ncol, TYPE* mat_in, TYPE* mat_out, cudnnHandle_t& handle) {
	std::cerr << "Error! Only support float or double." << std::endl;
}
template<> void cudnn_softmax<float>(size_t nrow, size_t ncol, float* mat_in, float* mat_out, cudnnHandle_t& handle) {
    cudnnTensorDescriptor_t srcTensorDesc, sftTensorDesc;
    cudnnCALL(cudnnCreateTensorDescriptor(&srcTensorDesc));
    cudnnCALL(cudnnCreateTensorDescriptor(&sftTensorDesc));
    
    cudnnCALL(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, nrow, ncol, 1, 1));
    cudnnCALL(cudnnSetTensor4dDescriptor(sftTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, nrow, ncol, 1, 1));
    
    float alpha = 1.0, beta = 0.0;
    cudnnCALL(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha, srcTensorDesc, mat_in, &beta, sftTensorDesc, mat_out));

	cudnnCALL(cudnnDestroyTensorDescriptor(srcTensorDesc));
	cudnnCALL(cudnnDestroyTensorDescriptor(sftTensorDesc));
}
template<> void cudnn_softmax<double>(size_t nrow, size_t ncol, double* mat_in, double* mat_out, cudnnHandle_t& handle) {
    cudnnTensorDescriptor_t srcTensorDesc, sftTensorDesc;
    cudnnCALL(cudnnCreateTensorDescriptor(&srcTensorDesc));
    cudnnCALL(cudnnCreateTensorDescriptor(&sftTensorDesc));
    
    cudnnCALL(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, nrow, ncol, 1, 1));
    cudnnCALL(cudnnSetTensor4dDescriptor(sftTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, nrow, ncol, 1, 1));
    
    double alpha = 1.0, beta = 0.0;
    cudnnCALL(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha, srcTensorDesc, mat_in, &beta, sftTensorDesc, mat_out));
	
	cudnnCALL(cudnnDestroyTensorDescriptor(srcTensorDesc));
	cudnnCALL(cudnnDestroyTensorDescriptor(sftTensorDesc));
}


template<typename TYPE> struct __functor_sigmoid {
	__host__ __device__ TYPE operator()(const TYPE& x) {
		// by default return float
		return 1.0 / (1.0 + expf(-x));
	}
};
template<> struct __functor_sigmoid<float> {
	__host__ __device__ float operator()(const float& x) {
		return 1.0 / (1.0 + expf(-x));
	}
};
template<> struct __functor_sigmoid<double> {
	__host__ __device__ float operator()(const double& x) {
		return 1.0 / (1.0 + exp(-x));
	}
};
// in out pointer COULD be same
template<typename TYPE> void cuda_sigmoid(size_t nrow, size_t ncol, TYPE* mat_in, TYPE* mat_out) {
	thrust::transform(thrust::device,
						thrust::device_pointer_cast<TYPE>(mat_in),
						thrust::device_pointer_cast<TYPE>(mat_in) + nrow * ncol,
						thrust::device_pointer_cast<TYPE>(mat_out),
						__functor_sigmoid<TYPE>());
}

template<typename TYPE> struct __functor_relu {
	__host__ __device__ TYPE operator()(const TYPE& x) {
		return x > 0.0 ? x : 0.0;
	}
};
// in out pointer COULD be same
template<typename TYPE> void cuda_relu(size_t nrow, size_t ncol, TYPE* mat_in, TYPE* mat_out) {
	thrust::transform(thrust::device,
						thrust::device_pointer_cast<TYPE>(mat_in),
						thrust::device_pointer_cast<TYPE>(mat_in) + nrow * ncol,
						thrust::device_pointer_cast<TYPE>(mat_out),
						__functor_relu<TYPE>());
}

template<typename TYPE> struct __functor_tanh {
	__host__ __device__ TYPE operator()(const TYPE& x) {
		// by default return float
		float exp_plus = expf(x);
		float exp_minus = expf(-x);
		return (exp_plus - exp_minus) / (exp_plus + exp_minus);
	}
};
template<> struct __functor_tanh<float> {
	__host__ __device__ float operator()(const float& x) {
		float exp_plus = expf(x);
		float exp_minus = expf(-x);
		return (exp_plus - exp_minus) / (exp_plus + exp_minus);
	}
};
template<> struct __functor_tanh<double> {
	__host__ __device__ double operator()(const double& x) {
		double exp_plus = exp(x);
		double exp_minus = exp(-x);
		return (exp_plus - exp_minus) / (exp_plus + exp_minus);
	}
};
// in out pointer COULD be same
template<typename TYPE> void cuda_tanh(size_t nrow, size_t ncol, TYPE* mat_in, TYPE* mat_out) {
	thrust::transform(thrust::device,
						thrust::device_pointer_cast<TYPE>(mat_in),
						thrust::device_pointer_cast<TYPE>(mat_in) + nrow * ncol,
						thrust::device_pointer_cast<TYPE>(mat_out),
						__functor_relu<TYPE>());
}

float const _FLT_MIN = std::numeric_limits<float>::min();
double const _DBL_MIN = std::numeric_limits<double>::min();
template<typename TYPE> struct __functor_eliminatezero {
	__host__ __device__ TYPE operator()(const TYPE& x) {
		// by default return float
		return x != 0 ? x : _FLT_MIN;
	}
};
template<> struct __functor_eliminatezero<float> {
	__host__ __device__ float operator()(const float& x) {
		return x != 0 ? x : _FLT_MIN;
	}
};
template<> struct __functor_eliminatezero<double> {
	__host__ __device__ double operator()(const double& x) {
		return x != 0 ? x : _DBL_MIN;
	}
};
// in out pointer COULD be same
template<typename TYPE> void cuda_eliminatezero(size_t nrow, size_t ncol, TYPE* mat_in, TYPE* mat_out) {
	thrust::transform(thrust::device,
						thrust::device_pointer_cast<TYPE>(mat_in),
						thrust::device_pointer_cast<TYPE>(mat_in) + nrow * ncol,
						thrust::device_pointer_cast<TYPE>(mat_out),
						__functor_eliminatezero<TYPE>());
}

// in out pointers could NOT be same
template<typename TYPE> void multiplyMatricesAndSoftmax(size_t nrow, size_t ncol, size_t commonsize, TYPE* mat_a, TYPE* mat_b, TYPE* mat_out) {
	//#pragma unroll
	for (uint32_t i = 0; i < nrow; ++i) {
		//#pragma unroll
		for (uint32_t j = 0; j < ncol; ++j) {
			mat_out[i * ncol + j] = 0;
			//#pragma unroll
			for (uint32_t k = 0; k < commonsize; ++k) {
				mat_out[i * ncol + j] += mat_a[i * commonsize + k] * mat_b[k * ncol + j];
			}
		}
		
		// find max
		TYPE max = mat_out[i * ncol + 0];
		//#pragma unroll
		for (uint32_t j = 1; j < ncol; ++j) {
			if (mat_out[i * ncol + j] > max) {
				max = mat_out[i * ncol + j];
			}
		}
		
		// calculate sum exp
		TYPE sum = 0;
		//#pragma unroll
		for (uint32_t j = 0; j < ncol; ++j) {
			mat_out[i * ncol + j] = std::exp(mat_out[i * ncol + j] - max);
			sum += mat_out[i * ncol + j];
		}
		
		// calculate each value
		//#pragma unroll
		for (uint32_t j = 0; j < ncol; ++j) {
			mat_out[i * ncol + j] = mat_out[i * ncol + j] * 1.0 / sum;
			// prevent numeric error
			if (mat_out[i * ncol + j] == 0) {
				mat_out[i * ncol + j] = std::numeric_limits<float>::min();
			}
		}
	}
}

// in out pointers could NOT be same
// transform matrix a by adding a new column of 1 before multiplication
template<typename TYPE> void multiplyMatricesAndAddedBiasColumnAndSoftmax(size_t nrow, size_t ncol, size_t commonsize, TYPE* mat_a, TYPE* mat_b, TYPE* mat_out) {
	//#pragma unroll
	for (uint32_t i = 0; i < nrow; ++i) {
		//#pragma unroll
		for (uint32_t j = 0; j < ncol; ++j) {
			mat_out[i * ncol + j] = 0;
			//#pragma unroll
			for (uint32_t k = 0; k < commonsize - 1; ++k) {
				mat_out[i * ncol + j] += mat_a[i * (commonsize - 1) + k] * mat_b[k * ncol + j];
			}
			mat_out[i * ncol + j] += mat_b[(commonsize - 1) * ncol + j];
		}
		
		// find max
		TYPE max = mat_out[i * ncol + 0];
		//#pragma unroll
		for (uint32_t j = 1; j < ncol; ++j) {
			if (mat_out[i * ncol + j] > max) {
				max = mat_out[i * ncol + j];
			}
		}
		
		// calculate sum exp
		TYPE sum = 0;
		//#pragma unroll
		for (uint32_t j = 0; j < ncol; ++j) {
			mat_out[i * ncol + j] = std::exp(mat_out[i * ncol + j] - max);
			sum += mat_out[i * ncol + j];
		}
		
		// calculate each value
		//#pragma unroll
		for (uint32_t j = 0; j < ncol; ++j) {
			mat_out[i * ncol + j] = mat_out[i * ncol + j] * 1.0 / sum;
			// prevent numeric error
			if (mat_out[i * ncol + j] == 0) {
				mat_out[i * ncol + j] = std::numeric_limits<float>::min();
			}
		}
	}
}

template<typename TYPE> TYPE evalCrossEntropy_quick(size_t nrow, size_t ncol, uint8_t* Ylabel, TYPE* Ybar) {
	TYPE enp = 0;
	//#pragma unroll
	for (uint32_t i = 0; i < nrow; ++i) {
		enp += std::log(Ybar[i * ncol + Ylabel[i]]);
	}
	return -enp / nrow;
}

template<typename TYPE> TYPE evalCrossEntropy(size_t nrow, size_t ncol, TYPE* Y, TYPE* Ybar) {
	TYPE enp = 0;
	for (uint32_t i = 0; i < nrow * ncol; ++i) {
		enp += Y[i] * std::log(Ybar[i]);
	}
	return -enp / nrow;
}


template<typename TYPE> struct __functor_evalCrossEntropy {
	// by default return float
	template <typename Tuple> __device__ TYPE operator()(Tuple tup) {
		return thrust::get<0>(tup) * logf(thrust::get<1>(tup));
	}
};
template<> struct __functor_evalCrossEntropy<float> {
	template <typename Tuple> __device__ float operator()(Tuple tup) {
		return thrust::get<0>(tup) * logf(thrust::get<1>(tup));
	}
};
template<> struct __functor_evalCrossEntropy<double> {
	template <typename Tuple> __device__ double operator()(Tuple tup) {
		return thrust::get<0>(tup) * log(thrust::get<1>(tup));
	}
};
template<typename TYPE> TYPE cuda_evalCrossEntropy(size_t nrow, size_t ncol, TYPE* Y, TYPE* Ybar) {
	auto begin = thrust::make_zip_iterator(thrust::make_tuple(Y, Ybar));
	auto end = thrust::make_zip_iterator(thrust::make_tuple(Y + nrow * ncol, Ybar + nrow * ncol));
	// bug code: TYPE enp = thrust::transform_reduce(thrust::device, begin, end, __functor_evalCrossEntropy<TYPE>(), 0, thrust::plus<TYPE>());
	TYPE enp = thrust::transform_reduce(thrust::device, begin, end, __functor_evalCrossEntropy<TYPE>(), (TYPE)0.0, thrust::plus<TYPE>());
	return -enp / nrow;
}

template<typename TYPE> struct __functor_evalMSE {
	// by default return float
	template <typename Tuple> __device__ TYPE operator()(Tuple tup) {
		return powf(thrust::get<0>(tup) - thrust::get<1>(tup), 2);
	}
};
template<> struct __functor_evalMSE<float> {
	template <typename Tuple> __device__ float operator()(Tuple tup) {
		return powf(thrust::get<0>(tup) - thrust::get<1>(tup), 2);
	}
};
template<> struct __functor_evalMSE<double> {
	template <typename Tuple> __device__ double operator()(Tuple tup) {
		return pow(thrust::get<0>(tup) - thrust::get<1>(tup), 2);
	}
};
template<typename TYPE> TYPE cuda_evalMSE(size_t nrow, size_t ncol, TYPE* Y, TYPE* Ybar) {
	auto begin = thrust::make_zip_iterator(thrust::make_tuple(Y, Ybar));
	auto end = thrust::make_zip_iterator(thrust::make_tuple(Y + nrow * ncol, Ybar + nrow * ncol));
	TYPE mse = thrust::transform_reduce(thrust::device, begin, end, __functor_evalMSE<TYPE>(), (TYPE)0.0, thrust::plus<TYPE>());
	return mse / nrow;
}

// for binary classification
template<typename TYPE> struct __functor_evalAccuracy {
	template <typename Tuple> __device__ TYPE operator()(Tuple tup) {
		return (thrust::get<0>(tup) - 0.5) * (thrust::get<1>(tup) - 0.5) > 0;
	}
};
template<typename TYPE> TYPE cuda_evalAccuracy(size_t nrow, size_t ncol, TYPE* Y, TYPE* Ybar) {
	auto begin = thrust::make_zip_iterator(thrust::make_tuple(Y, Ybar));
	auto end = thrust::make_zip_iterator(thrust::make_tuple(Y + nrow * ncol, Ybar + nrow * ncol));
	TYPE count = thrust::transform_reduce(thrust::device, begin, end, __functor_evalAccuracy<TYPE>(), (TYPE)0.0, thrust::plus<TYPE>());
	return count / nrow;
}

#endif	// MATRIX_HPP
