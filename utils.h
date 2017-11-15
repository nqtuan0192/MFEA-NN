// macro functions
#ifndef HELPER_MACROS
#define HELPER_MACROS

#define BUG_ENDL			std::cout << std::endl;
#define BUG_TAB				std::cout << "\t";
#define BUG_SPACE			std::cout << " ";
#define BUG(v)				std::cout << typeid(v).name() << ": " << #v << " = " << v  << std::endl;
#define BUG_I(v, i)			std::cout << #v << "[" << i << "]" << " = " << v[i];
#define BUG_II(v, i, j)		std::cout << #v << "[" << i << "]"  << "[" << j << "]" << " = " << v[i][j];
#define BUG_A(a, n)			{ for(unsigned int i = 0; i < n; ++i) { BUG_I(a, i) BUG_TAB } }
#define BUG_M(a, m, n)		{ for(unsigned int i = 0; i < m; ++i) { for(unsigned int j = 0; j < n; ++j) { BUG_II(a, i, j) BUG_TAB } BUG_ENDL } }

#define SWAP(type, A, B)	{ type __temp__ = A; A = B; B = __temp__; }
#define ABS(num)	((num) < 0 ? (-(num)) : (num))
#define MAX2(A, B)	((A) > (B) ? (A) : (B))			// Max in 2 numbers
#define MAX3(A, B, C)	((A) > (B) ? MAX2(A, C) : MAX2(B, C))	// Max in 3 numbers
#define MIN2(A, B)	((A) < (B) ? (A) : (B))			// Min in 2 numbers
#define MIN3(A, B, C)	((A) < (B) ? MIN2(A, C) : MIN2(B, C))	// Min in 3 numbers
#define RANDBETWEEN(min, max) ((rand() % (int)(((max) + 1) - (min))) + (min))	// Macro to get a random integer with a specified range
#define FORI_NEQ(i, a, b)	for (i = a; i != b; ++i)		// increasing for loop from a if NOT EQUAL b
#define FORD_NEQ(i, a, b)	for (i = a; i != b; --i)		// decreasing for loop from a if NOT EQUAL b
#define FORI_L(i, a, b)		for (i = a; i < b; ++i)			// increasing for loop from a if LESS than b
#define FORD_G(i, a, b)		for (i = a; i > b; --i)			// decreasing for loop from a if GREATER than b
#define FORI_LEQ(i, a, b)	for (i = a; i <= b; ++i)		// increasing for loop from a if LESS or EQUAL b
#define FORD_GEQ(i, a, b)	for (i = a; i >= b; --i)		// decreasing for loop from a if GREATER or EQUAL b
#define FORIS_NEQ(i, a, b, s)	for (i = a; i != b; i += s)		// increasing for loop from a if NOT EQUAL b with STEP s
#define FORDS_NEQ(i, a, b, s)	for (i = a; i != b; i -= s)		// decreasing for loop from a if NOT EQUAL b with STEP s
#define FORIS_L(i, a, b, s)	for (i = a; i < b; i += s)		// increasing for loop from a if LESS than b with STEP s
#define FORDS_G(i, a, b, s)	for (i = a; i > b; i -= s)		// decreasing for loop from a if GREATER than b with STEP s
#define FORIS_LEQ(i, a, b, s)	for (i = a; i <= b; i += s)		// increasing for loop from a if LESS or EQUAL b with STEP s
#define FORDS_GEQ(i, a, b, s)	for (i = a; i >= b; i -= s)		// decreasing for loop from a if GREATER or EQUAL b with STEP s

#define M_MALLOC(type, number)	(type*)malloc(number * sizeof(type))
#define M_CALLOC(type, number)	(type*)calloc(number, sizeof(type))
#define M_ALLOC(type, number)	(type*)alloc(number * sizeof(type))

#define MK_PTR32(type, base, offset)	(type)((uint32_t)base + (uint32_t)offset)
#define MK_PTR64(type, base, offset)	(type)((uint64_t)base + (uint64_t)offset)
#define DIF_PTR32(type, ptr1, ptr2)	(type)((int32_t)ptr1 - (int32_t)ptr2)
#define DIF_PTR64(type, ptr1, ptr2)	(type)((int64_t)ptr1 - (int64_t)ptr2)

// strings
#define linespace "-------------------------------------------------\n"

// constants
#define MATHCONST_PI	3.14159265358979323846264338327950288419716939937510
#define MATHCONST_E	2.71828182845904523536028747135266249775724709369995

#endif	// HELPER_MACROS

#ifndef FAST_RANDOM
#define FAST_RANDOM
uint32_t ___g_seed___;

// Used to seed the generator.           
inline void fast_srand(uint32_t seed) {
	___g_seed___ = seed;
}

// Compute a pseudorandom integer.
// Output value in range [0, 32767]
inline uint32_t fast_rand() {
	___g_seed___ = (214013 * ___g_seed___ + 2531011);
	return (___g_seed___ >> 16) & 0x7FFF;	// 32767
}
#endif

#ifndef FPRINT_SUPPORT
#define FPRINT_SUPPORT
template <typename DATA> inline void fprintFormatedUDec(FILE* fp, DATA data) {
	switch (sizeof(DATA))
	{
	case 1:
		fprintf(fp, "%d", data);
		break;
	case 2:
		fprintf(fp, "%d", data);
		break;
	case 4:
		fprintf(fp, "%d", data);
		break;
	case 8:
		fprintf(fp, "%llu", data);
		break;
	default:
		break;
	}
}
template <typename DATA> inline void fprintFormatedSDec(FILE* fp, DATA data) {
	switch (sizeof(DATA))
	{
	case 1:
		fprintf(fp, "%d", data);
		break;
	case 2:
		fprintf(fp, "%d", data);
		break;
	case 4:
		fprintf(fp, "%d", data);
		break;
	case 8:
		fprintf(fp, "%ld", data);
		break;
	default:
		break;
	}
}
template <typename DATA> inline void fprintFormatedHex(FILE* fp, DATA data) {
	switch (sizeof(DATA))
	{
	case 1:
		fprintf(fp, "%02X", data);
		break;
	case 2:
		fprintf(fp, "%04X", data);
		break;
	case 4:
		fprintf(fp, "%08X", data);
		break;
	case 8:
		fprintf(fp, "%016X", data);
		break;
	default:
		break;
	}
}
inline void fprintCharArray(FILE* fp, void* data, int size) {
	for (int i = 0; i < size; ++i)
		fprintf(fp, "%c", *((char*)data + i));
}
inline void fprintTrueByteOrder(FILE* fp, void* data, int size) {
	for (int i = 0; i < size; ++i)
		fprintf(fp, "%02X", *((unsigned char*)data + i));
}
inline void fprintReverseByteOrder(FILE* fp, void* data, int size) {
	for (int i = size - 1; i >= 0; --i)
		fprintf(fp, "%02X", *((unsigned char*)data + i));
}
inline void fprintTruebitOrder(FILE* fp, void* data, int size) {
	const char* S = "01";
	for (int i = 0; i < size; ++i)
		for (int j = 7; j >= 0; --j)
			fprintf(fp, "%c", S[(*((unsigned char*)data + i) >> j) & 0x1]);
}
#endif


#ifndef CUDA_SUPPORT	// for CUDA only
#define CUDA_SUPPORT

#include <cassert>

#define cudaCALL(ans) cudaAssert((ans), __FILE__, __LINE__)
inline cudaError_t cudaAssert(cudaError_t code, const char *file, int line, bool abort = true) {
#if defined(DEBUG) || defined(_DEBUG)
	if (code != cudaSuccess) {
		fprintf(stderr,"GPU CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
#endif
	return code;
}

#define curandCALL(ans) curandAssert((ans), __FILE__, __LINE__)
inline curandStatus_t curandAssert(curandStatus_t code, const char *file, int line, bool abort = true) {
#if defined(DEBUG) || defined(_DEBUG)
	if (code != CURAND_STATUS_SUCCESS) {
		fprintf(stderr,"CURAND assert: %d %s %d\n", code, file, line);
		if (abort) exit(code);
	}
#endif
	return code;
}

#define cublasCALL(ans) cublasAssert((ans), __FILE__, __LINE__)
inline cublasStatus_t cublasAssert(cublasStatus_t code, const char *file, int line, bool abort = true) {
#if defined(DEBUG) || defined(_DEBUG)
	if (code != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr,"CUBLAS assert: %d %s %d\n", code, file, line);
		if (abort) exit(code);
	}
#endif
	return code;
}

#define cudnnCALL(ans) cudnnAssert((ans), __FILE__, __LINE__)
inline cudnnStatus_t cudnnAssert(cudnnStatus_t code, const char *file, int line, bool abort = true) {
#if defined(DEBUG) || defined(_DEBUG)
	if (code != CUDNN_STATUS_SUCCESS) {
		fprintf(stderr,"CUDNN assert: %d %s %d\n", code, file, line);
		if (abort) exit(code);
	}
#endif
	return code;
}

template<typename TYPE> cudaError_t cudaStreamSynchronize(TYPE handle) {
	std::cerr << "Unknown handle type" << std::endl;
}
template<> cudaError_t cudaStreamSynchronize<cublasHandle_t>(cublasHandle_t handle) {
	cudaStream_t cuda_stream;
	cublasCALL(cublasGetStream(handle, &cuda_stream));
	return cudaStreamSynchronize(cuda_stream);
}
template<> cudaError_t cudaStreamSynchronize<cudnnHandle_t>(cudnnHandle_t handle) {
	cudaStream_t cuda_stream;
	cudnnCALL(cudnnGetStream(handle, &cuda_stream));
	return cudaStreamSynchronize(cuda_stream);
}

#define CUDA_M_MALLOC(ptr, type, number)					cudaMalloc((void**)&(ptr), (number) * sizeof(type))
#define CUDA_M_MALLOC_MANAGED(ptr, type, number)			cudaMallocManaged((void**)&(ptr), (number) * sizeof(type))
#define CUDA_M_COPY_HOSTTODEVICE(from, to, type, number)	cudaMemcpy((type*)to, (type*)from, (number) * sizeof(type), cudaMemcpyHostToDevice)
#define CUDA_M_COPY_DEVICETOHOST(from, to, type, number)	cudaMemcpy((type*)to, (type*)from, (number) * sizeof(type), cudaMemcpyDeviceToHost)


#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

namespace thrust {
	template <typename Iterator>
	class strided_range
	{
		public:

		typedef typename thrust::iterator_difference<Iterator>::type difference_type;

		struct stride_functor : public thrust::unary_function<difference_type,difference_type>
		{
			difference_type stride;

			stride_functor(difference_type stride)
				: stride(stride) {}

			__host__ __device__
			difference_type operator()(const difference_type& i) const
			{ 
				return stride * i;
			}
		};

		typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
		typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
		typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

		// type of the strided_range iterator
		typedef PermutationIterator iterator;

		// construct strided_range for the range [first,last)
		strided_range(Iterator first, Iterator last, difference_type stride)
			: first(first), last(last), stride(stride) {}
	   
		iterator begin(void) const
		{
			return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
		}

		iterator end(void) const
		{
			return begin() + ((last - first) + (stride - 1)) / stride;
		}
		
		protected:
		Iterator first;
		Iterator last;
		difference_type stride;
	};
};

#endif 	// CUDA_SUPPORT

#ifndef MEMORY_SUPPORT	// for C++11 only
#define MEMORY_SUPPORT
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace std {
    template<class T> struct _Unique_if {
        typedef unique_ptr<T> _Single_object;
    };

    template<class T> struct _Unique_if<T[]> {
        typedef unique_ptr<T[]> _Unknown_bound;
    };

    template<class T, size_t N> struct _Unique_if<T[N]> {
        typedef void _Known_bound;
    };

    template<class T, class... Args>
        typename _Unique_if<T>::_Single_object
        make_unique(Args&&... args) {
            return unique_ptr<T>(new T(std::forward<Args>(args)...));
        }

    template<class T>
        typename _Unique_if<T>::_Unknown_bound
        make_unique(size_t n) {
            typedef typename remove_extent<T>::type U;
            return unique_ptr<T>(new U[n]());
        }

    template<class T, class... Args>
        typename _Unique_if<T>::_Known_bound
        make_unique(Args&&...) = delete;
}
#endif	// MEMORY_SUPPORT

#define DATATYPE		float
