#ifndef MFEA_HPP
#define MFEA_HPP

#include <thread>
#include <mutex>
#include <array>
#include <queue>
#include <unordered_set>
#include <chrono>
#include <omp.h>

#include <curand.h>
#include <thrust/fill.h>

#include "MnistHelper.hpp"
#include "MFEAChromosome.hpp"
#include "IndexValue.hpp"

// prng
std::random_device mfea_random_device;
std::mt19937 mfea_mt_engine(mfea_random_device());


// thread eval
std::mutex thread_mutex;
template<size_t population_size, size_t generation_size, size_t thread_size>
void handle_thread(int _device_id, uint32_t thread_id, std::queue<uint32_t>& data,
					std::array<MFEA_Chromosome, 2 * population_size>& _population, uint32_t _beg, uint32_t _num,
					size_t _training_size,
					DATATYPE* _working_training_input_data_ptr,
					DATATYPE* _working_training_output_data_ptr,
					std::array<DATATYPE*, thread_size>& _dev_mat_temp_rnvec,
					std::array<DATATYPE*, thread_size>& _dev_mat_temp_w,
					std::array<DATATYPE*, thread_size>& _dev_mat_ones,
					std::array<std::array<DATATYPE*, LAYER_SIZE>, thread_size>& _dev_mat_temp_layers) {
//	std::cout << "Thread " << thread_id << " has started..." << std::endl;
	
	cudaCALL(cudaSetDevice(_device_id));
	
	cudaStream_t cuda_stream;
	cudaCALL(cudaStreamCreate(&cuda_stream));
	
	/* Initialize CUBLAS */
	cublasHandle_t cublas_handle;
	cublasCALL(cublasCreate(&cublas_handle));
	cublasCALL(cublasSetStream(cublas_handle, cuda_stream));
	
	/* Initialize CUDNN */
	cudnnHandle_t cudnn_handle;
	cudnnCALL(cudnnCreate(&cudnn_handle));
	cudnnCALL(cudnnSetStream(cudnn_handle, cuda_stream));
	
	/* do stuffs */
	uint32_t it;
	while (true) {
		thread_mutex.lock();
		if (data.size() > 0) {
			it = data.front();
//			std::cout << "Thread " << thread_id << " is evaluating " << it << " ... " << std::endl;
			data.pop();
			thread_mutex.unlock();
		} else {
//			std::cout << "the end" << std::endl;
			thread_mutex.unlock();
			break;
		}

		_population[it].evalObj(_training_size, OUTPUT_SIZE,
								_working_training_input_data_ptr,
								_working_training_output_data_ptr,
								_dev_mat_temp_rnvec,
								_dev_mat_temp_w,
								_dev_mat_ones[thread_id],
								_dev_mat_temp_layers[thread_id],
								cublas_handle, cudnn_handle);
	}
	
	/* Finalize */
	cublasCALL(cublasDestroy(cublas_handle));
	cudnnCALL(cudnnDestroy(cudnn_handle));
	cudaCALL(cudaStreamDestroy(cuda_stream));
	
//	std::cout << "Thread " << thread_id << " stopped..." << std::endl;
}

template<size_t population_size, size_t generation_size, size_t thread_size> class MFEA {
private:
	std::array<MFEA_Chromosome, 2 * population_size> population;
	std::array<std::array<IndexValue<uint32_t, DATATYPE>, 2 * population_size>, TASK_SIZE> tasks_indexvalue;
	
	std::array<std::thread, thread_size> threads;
	std::queue<uint32_t> _eval_indexes_queue;
	
	std::array<DATATYPE, TASK_SIZE> bestobj;
	std::array<std::array<DATATYPE, TASK_SIZE>, generation_size + 1> EvBestFitness;
	std::array<MFEA_Chromosome, TASK_SIZE> bestInd_data;
	
	// variables for training data_ptr
	DATATYPE* training_input_data_ptr;
	DATATYPE* training_output_data_ptr;

	// variables for testing data_ptr
	DATATYPE* testing_input_data_ptr;
	DATATYPE* testing_output_data_ptr;
	
	// CUDA things
	curandGenerator_t curand_prng;
	cublasHandle_t cublas_handle;
	cudnnHandle_t cudnn_handle;
	int _device_id;
	
	// feedforward preallocated memory
	std::array<DATATYPE*, thread_size> dev_mat_temp_rnvec;
	std::array<DATATYPE*, thread_size> dev_mat_temp_w;
	std::array<DATATYPE*, thread_size> dev_mat_ones;	// by thread index
	std::array<std::array<DATATYPE*, LAYER_SIZE + 1>, thread_size> dev_mat_temp_layers;	// by thread index and layer index

	
	// crossover and mutation preallocated memory
	std::array<DATATYPE*, thread_size> dev_ct_beta;	// by thread index
	std::array<DATATYPE*, thread_size> dev_rand;	// by thread index

	const uint32_t THREAD_IDX_CURRENT = 0;
	
	// tunable fators
	DATATYPE cf_distributionindex = 2; 			// crossover factor, index of Simulated Binary Crossover
	DATATYPE mf_randommatingprobability = 1;		// mutation factor, random mating probability
	DATATYPE mf_polynomialmutationindex = 5;		// mutation factor, index of Polynomial Mutation Operator
	DATATYPE mf_mutationratio = 1 / getTotalLayerWeightsandBiases();				// mutation factor, 
	
	
	// logging
	DATATYPE results[TASK_SIZE][2][2];		// task, train-test, factorialcost-accuracy
	std::ofstream log_file;
	std::ofstream training_results_loss;
	std::ofstream training_results_acc;
	std::ofstream testing_results_loss;
	std::ofstream testing_results_acc;
	std::ofstream loss_value_log;
	

public:
	MFEA(DATATYPE* training_input_data, DATATYPE* training_output_data,
					DATATYPE* testing_input_data, DATATYPE* testing_output_data, int device_id) :
					training_input_data_ptr(training_input_data), training_output_data_ptr(training_output_data),
					testing_input_data_ptr(testing_input_data), testing_output_data_ptr(testing_output_data),
					_device_id(device_id) {
	}
	
	
	int init_libraries() {
		log_file.open("mfea.log");
		training_results_loss.open("training_results_loss.log");
		training_results_acc.open("training_results_acc.log");
		testing_results_loss.open("testing_results_loss.log");
		testing_results_acc.open("testing_results_acc.log");
		loss_value_log.open("loss_value_log.txt");
		
		// Create a pseudo-random number generator
		curandCreateGenerator(&curand_prng, CURAND_RNG_PSEUDO_MTGP32);//CURAND_RNG_PSEUDO_MTGP32	CURAND_RNG_QUASI_SCRAMBLED_SOBOL32
		// Set the seed for the random number generator using the system clock
		curandSetPseudoRandomGeneratorSeed(curand_prng, std::chrono::high_resolution_clock::now().time_since_epoch().count());
		
		/* Initialize CUBLAS */
		cublasCALL(cublasCreate(&cublas_handle));
		
		/* Initialize CUDNN */
		cudnnCALL(cudnnCreate(&cudnn_handle));
		
		/* Initialize temporary space */
		for (uint32_t i = 0; i < thread_size; ++i) {
			cudaCALL(CUDA_M_MALLOC_MANAGED(dev_mat_temp_rnvec[i], DATATYPE, getTotalLayerWeightsandBiases()));
			cudaCALL(CUDA_M_MALLOC_MANAGED(dev_mat_temp_w[i], DATATYPE, getTotalLayerWeightsandBiases()));
			// pre-allocate mat_one by TRAINING_SIZE
			// mat_one is used for populating biases
			cudaCALL(CUDA_M_MALLOC_MANAGED(dev_mat_ones[i], DATATYPE, TRAINING_SIZE));
			cuda_fillMatrix<DATATYPE>(TRAINING_SIZE, 1, dev_mat_ones[i], 1.0f);
			
			for (uint32_t j = 0; j < LAYER_SIZE + 1; ++j) {
				// pre-allocate by TRAINING_SIZE multiple by the largest number of units in each layer
				// dev_mat_temp_layers is used for storing temp matrix during forward propagation
				cudaCALL(CUDA_M_MALLOC_MANAGED(dev_mat_temp_layers[i][j], DATATYPE, TRAINING_SIZE * getMaximumNumberofUnitsofUnifiedLayer(j)));
			}
			
			// pre-allocate dev_ct_beta by logest weights and biases vector
			// dev_ct_beta is used for genetic operators
			// layer 1 (first hidden layer) should be largest number of neurons
			cudaCALL(CUDA_M_MALLOC_MANAGED(dev_ct_beta[i], DATATYPE, getMaximumLayerWeightsandBiasesatAll()));
			cudaCALL(CUDA_M_MALLOC_MANAGED(dev_rand[i], DATATYPE, getMaximumLayerWeightsandBiasesatAll()));
		}
		
		cudaStream_t cublas_stream, cudnn_stream;
		cublasCALL(cublasGetStream(cublas_handle, &cublas_stream));
		cudnnCALL(cudnnGetStream(cudnn_handle, &cudnn_stream));
		
		cudaCALL(cudaStreamSynchronize(cublas_handle));
		cudaCALL(cudaStreamSynchronize(cudnn_handle));
		
		std::cout << "CUBLAS stream is " << cublas_stream << " while CUDNN stream is " << cudnn_stream << std::endl;
		
		
		
		return 0;
	}
	
	int finalize_libraries() {
		log_file.close();
		training_results_loss.close();
		training_results_acc.close();
		testing_results_loss.close();
		testing_results_acc.close();
		loss_value_log.close();
		
		/* Destroy CUDNN */
		cudnnDestroy(cudnn_handle);
		/* Destroy CUBLAS */
		return cublasDestroy(cublas_handle);
	}
	
	void initialize() {
		// reseed prng
		curandSetPseudoRandomGeneratorSeed(curand_prng, std::chrono::high_resolution_clock::now().time_since_epoch().count());
		
		// reset records
		for (uint32_t task = 0; task < TASK_SIZE; ++task) {
			bestobj[task] = std::numeric_limits<DATATYPE>::max();
		}
		for (uint32_t i = 0; i < generation_size + 1; ++i) {
			for (uint32_t task = 0; task < TASK_SIZE; ++task) {
				EvBestFitness[i][task] = std::numeric_limits<DATATYPE>::max();
			}
		}
		
		// randomize data for initialization
		thrust::for_each(population.begin(), population.begin() + population_size, MFEA_Chromosome_Randomize(curand_prng));

		// evaluate factorial costs
		//#pragma omp parallel for	// parallel tasking
		for (uint32_t i = 0; i < population_size; ++i) {
			population[i].skill_factor = i % TASK_SIZE;
			population[i].evalObj(TRAINING_SIZE, OUTPUT_SIZE, training_input_data_ptr, training_output_data_ptr,
						dev_mat_temp_rnvec[THREAD_IDX_CURRENT], dev_mat_temp_w[THREAD_IDX_CURRENT],
						dev_mat_ones[THREAD_IDX_CURRENT], dev_mat_temp_layers[THREAD_IDX_CURRENT],
						cublas_handle, cudnn_handle);
		}
	}
	

		
	void evolution() {
		uint32_t generation = 0;
		uint32_t count = 0;
		
		while (++generation <= generation_size) {
			std::cout << "generation " << generation << std::endl;
			
			// random shuffle before crossing over
			std::random_shuffle(population.begin(), population.begin() + population_size);

			count = 0;
			for (uint32_t i = 0; i < population_size / 2; ++i) {
				uint32_t p1 = i, p2 = population_size / 2 + i, c1 = population_size + count, c2 = population_size + count + 1;
				if ((population[p1].skill_factor == population[p2].skill_factor) || f_rand() <= mf_randommatingprobability) {
					crossover(population[p1], population[p2], population[c1], population[c2], cf_distributionindex, dev_ct_beta[THREAD_IDX_CURRENT], curand_prng);

					
					mutate(population[c1], population[c1], mf_polynomialmutationindex, mf_mutationratio, dev_ct_beta[THREAD_IDX_CURRENT], dev_rand[THREAD_IDX_CURRENT], curand_prng);
					mutate(population[c2], population[c2], mf_polynomialmutationindex, mf_mutationratio, dev_ct_beta[THREAD_IDX_CURRENT], dev_rand[THREAD_IDX_CURRENT], curand_prng);
					
					// probabilistic assign the skill factor for children from their parents 
					if (f_rand() <= 0.5) {
						population[c1].skill_factor = population[p1].skill_factor;
					} else {
						population[c1].skill_factor = population[p2].skill_factor;
					}
					if (f_rand() <= 0.5) {
						population[c2].skill_factor = population[p1].skill_factor;
					} else {
						population[c2].skill_factor = population[p2].skill_factor;
					}
					
					// Uniform crossover-like variable swap between two new born children
					uniformcrossoverlike(population[c1], population[c2], dev_ct_beta[THREAD_IDX_CURRENT], curand_prng);

					//examineIndividual(population[population_size + count]);
					//transformWeights(population[population_size + count], dev_ct_beta[THREAD_IDX_CURRENT], curand_prng);
					//transformWeights(population[population_size + count + 1], dev_ct_beta[THREAD_IDX_CURRENT], curand_prng);
				} else {
					std::cout << "never get here" << std::endl;
				}
				count += 2;
			}


			// evaluate factorial costs for children
			for (uint32_t i = 0; i < population_size; ++i) {
				population[population_size + i].evalObj(TRAINING_SIZE, OUTPUT_SIZE,
														training_input_data_ptr,
														training_output_data_ptr,
														dev_mat_temp_rnvec[THREAD_IDX_CURRENT],
														dev_mat_temp_w[THREAD_IDX_CURRENT],
														dev_mat_ones[THREAD_IDX_CURRENT],
														dev_mat_temp_layers[THREAD_IDX_CURRENT],
														cublas_handle, cudnn_handle);
			}
			/*parallelEval(population, population_size, population_size,
							TRAINING_SIZE,
							training_input_data_ptr,
							training_output_data_ptr,
							dev_mat_temp_layers,
							dev_mat_ones,
							dev_mat_temp_layers);*/
			
			//#pragma omp parallel for	// parallel tasking
			for (uint32_t i = 0; i < population_size * 2; ++i) {
				// assign to array to be sorted
				for (uint32_t task = 0; task < TASK_SIZE; ++task) {
					tasks_indexvalue[task][i].index = i;
					tasks_indexvalue[task][i].value = population[i].factorial_costs[task];
				}
			}
			
			//#pragma omp parallel for	// parallel tasking
			for (uint32_t task = 0; task < TASK_SIZE; ++task) {
				// sort factorial cost of all inviduals for each task
				std::sort(tasks_indexvalue[task].begin(), tasks_indexvalue[task].end());
				// ranking all inviduals for each task
				for (uint32_t i = 0; i < 2 * population_size; ++i) {
					population[tasks_indexvalue[task][i].index].factorial_rank[task] = i + 1;
				}
			}
			
			// update best individuals for each generation
			for (uint32_t task = 0; task < TASK_SIZE; ++task) {
				if (population[tasks_indexvalue[task][0].index].factorial_costs[task] < bestobj[task]) {
					bestobj[task] = population[tasks_indexvalue[task][0].index].factorial_costs[task];
					bestInd_data[task] = population[tasks_indexvalue[task][0].index];
				}
				EvBestFitness[generation][task] = bestobj[task];
			}

			
			// evaluate scalar fitness for selection
			for (uint32_t i = 0; i < 2 * population_size; ++i) {
				DATATYPE temp = 1.0 / *std::min_element(population[i].factorial_rank.begin(), population[i].factorial_rank.end());
				population[i].scalar_fitness = temp;
				//std::cout << population[i].factorial_rank[0] << "\t" << population[i].factorial_rank[1] << "\t" << population[i].factorial_rank[2] << std::endl;
				//std::cout << temp << std::endl;
			}
			
			std::sort(population.rbegin(), population.rend());

			
			std::cout << "Current best loss value: ";
			for (uint32_t task = 0; task < TASK_SIZE; ++task) {
				std::cout << EvBestFitness[generation][task] << "\t";
				//loss_value_log << EvBestFitness[generation][task] << ",\t";
			} std::cout << std::endl;	//loss_value_log << std::endl;

		}
		
		for (uint32_t task = 0; task < TASK_SIZE; ++task) {
			bestInd_data[task].evalObj(TRAINING_SIZE, OUTPUT_SIZE, training_input_data_ptr, training_output_data_ptr,
							dev_mat_temp_rnvec[THREAD_IDX_CURRENT], dev_mat_temp_w[THREAD_IDX_CURRENT],
							dev_mat_ones[THREAD_IDX_CURRENT], dev_mat_temp_layers[THREAD_IDX_CURRENT],
							cublas_handle, cudnn_handle, true);
			std::cout << "-------Best found individuals for task " << task << std::endl;
			std::cout << bestInd_data[task];
		}
	}
	
	/*
	 * Re-evaluating for whole population and all tasks
	 * Sumarise the results
	 * */
	void reEvaluateTheFinalPopulation() {
		for (uint32_t i = 0; i < population_size; ++i) {
			// eval all tasks on training data
			population[i].evalObj(TRAINING_SIZE, OUTPUT_SIZE,
							training_input_data_ptr, training_output_data_ptr,
							dev_mat_temp_rnvec[THREAD_IDX_CURRENT], dev_mat_temp_w[THREAD_IDX_CURRENT],
							dev_mat_ones[THREAD_IDX_CURRENT], dev_mat_temp_layers[THREAD_IDX_CURRENT],
							cublas_handle, cudnn_handle,
							true, true, true);
		}
		/*
		std::cout << "Final population cross entropy:" << std::endl;
		for (uint32_t i = 0; i < population_size; ++i) {
			std::cout << "\t" << std::endl;
			for (uint32_t task = 0; task < TASK_SIZE; ++task) {
				std::cout << population[i].factorial_costs[task] << ",\t";
			}
			std::cout << std::endl;
		}
		std::cout << "Final population accuracy:" << std::endl;
		for (uint32_t i = 0; i < population_size; ++i) {
			std::cout << "\t" << std::endl;
			for (uint32_t task = 0; task < TASK_SIZE; ++task) {
				std::cout << population[i].accuracy[task] << ",\t";
			}
			std::cout << std::endl;
		}
		
		for (uint32_t i = 0; i < population_size; ++i) {
			// eval all tasks on testing data
			population[i].evalObj(TEST_SIZE, OUTPUT_SIZE,
							testing_input_data_ptr, testing_output_data_ptr, testing_labels_data_ptr,
							dev_mat_ones[THREAD_IDX_CURRENT], dev_mat_temp_layers[THREAD_IDX_CURRENT],
							cublas_handle, cudnn_handle,
							true, true, true);
		}
		
		std::cout << "Final population cross entropy:" << std::endl;
		for (uint32_t i = 0; i < population_size; ++i) {
			std::cout << "\t" << std::endl;
			for (uint32_t task = 0; task < TASK_SIZE; ++task) {
				std::cout << population[i].factorial_costs[task] << ",\t";
			}
			std::cout << std::endl;
		}
		std::cout << "Final population accuracy:" << std::endl;
		for (uint32_t i = 0; i < population_size; ++i) {
			std::cout << "\t" << std::endl;
			for (uint32_t task = 0; task < TASK_SIZE; ++task) {
				std::cout << population[i].accuracy[task] << ",\t";
			}
			std::cout << std::endl;
		}*/
	}
	
	void sumariseResults() {
		for (uint32_t task = 0; task < TASK_SIZE; ++task) {
			std::cout << "-------Best final individuals for task " << task << std::endl;
			
			// re-evaluating for the whole training dataset
			bestInd_data[task].evalObj(TRAINING_SIZE, OUTPUT_SIZE, training_input_data_ptr, training_output_data_ptr,
							dev_mat_temp_rnvec[THREAD_IDX_CURRENT], dev_mat_temp_w[THREAD_IDX_CURRENT],
							dev_mat_ones[THREAD_IDX_CURRENT], dev_mat_temp_layers[THREAD_IDX_CURRENT],
							cublas_handle, cudnn_handle, true);
			
			results[task][0][0] = bestInd_data[task].factorial_costs[task];
			results[task][0][1] = bestInd_data[task].accuracy[task];
			
			std::cout << bestInd_data[task];
			
			// evaluating for the whole test dataset
			bestInd_data[task].evalObj(TESTING_SIZE, OUTPUT_SIZE, testing_input_data_ptr, testing_output_data_ptr,
							dev_mat_temp_rnvec[THREAD_IDX_CURRENT], dev_mat_temp_w[THREAD_IDX_CURRENT],
							dev_mat_ones[THREAD_IDX_CURRENT], dev_mat_temp_layers[THREAD_IDX_CURRENT],
							cublas_handle, cudnn_handle, true);
			
			results[task][1][0] = bestInd_data[task].factorial_costs[task];
			results[task][1][1] = bestInd_data[task].accuracy[task];
			
			std::cout << bestInd_data[task];
		}
		bestInd_data[0].exportToFile("idv_0.txt");
		bestInd_data[1].exportToFile("idv_1.txt");
		bestInd_data[2].exportToFile("idv_2.txt");
	}
	
	
	void countLabels(uint8_t* ldata_ptr, size_t ldata_size) {	// size is number of data samples
		uint32_t lcount[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		float lper[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		for (uint32_t i = 0; i < ldata_size; ++i) {
			++lcount[ldata_ptr[i]];
		}
		for (uint32_t i = 0; i < 10; ++i) {
			lper[i] = lcount[i] * 1.0 / ldata_size;
		}
		BUG_A(lcount, 10); BUG_ENDL;
		BUG_A(lper, 10); BUG_ENDL;
	}
	
	
	void setTunableFactors(DATATYPE __cf_distributionindex,
							DATATYPE __mf_randommatingprobability,
							DATATYPE __mf_polynomialmutationindex,
							DATATYPE __mf_mutationratio	) {
		cf_distributionindex = __cf_distributionindex; 				// crossover factor, index of Simulated Binary Crossover
		mf_randommatingprobability = __mf_randommatingprobability;	// mutation factor, random mating probability
		mf_polynomialmutationindex = __mf_polynomialmutationindex;	// mutation factor, index of Polynomial Mutation Operator
		mf_mutationratio = __mf_mutationratio;						// mutation factor, 
	}
	void writeSumaryResults() {	// write log for each evolution
		log_file << "Tunable factors setting:" << std::endl;
		log_file << "cf_distributionindex       = " << cf_distributionindex << std::endl;
		log_file << "mf_randommatingprobability = " << mf_randommatingprobability << std::endl;
		log_file << "mf_polynomialmutationindex = " << mf_polynomialmutationindex << std::endl;
		log_file << "mf_mutationratio           = " << mf_mutationratio << std::endl;
		log_file << "Best object function:" << std::endl;
		for (uint32_t task = 0; task < TASK_SIZE; ++task) {
			log_file << bestobj[task] << "\t";
		}	log_file << std::endl;
		
		log_file << "-------Best value for training set (crossentropy, accuracy): " << std::endl;
		for (uint32_t task = 0; task < TASK_SIZE; ++task) {
			log_file << results[task][0][0] << ",\t" << results[task][0][1] << std::endl;
		}	log_file << std::endl;
		
		log_file << "-------Best value for testing set (crossentropy, accuracy): " << std::endl;
		for (uint32_t task = 0; task < TASK_SIZE; ++task) {
			log_file << results[task][1][0] << ",\t" << results[task][1][1] << std::endl;
		}	log_file << std::endl;
		
		log_file << linespace;
		
		for (uint32_t task = 0; task < TASK_SIZE; ++task) {
			training_results_loss << results[task][0][0] << ",\t";
			training_results_acc << results[task][0][1] << ",\t";
			testing_results_loss << results[task][1][0] << ",\t";
			testing_results_acc << results[task][1][1] << ",\t";
		}
		training_results_loss << std::endl;
		training_results_acc << std::endl;
		testing_results_loss << std::endl;
		testing_results_acc << std::endl;
	}
	
private:
	void parallelEval(std::array<MFEA_Chromosome, 2 * population_size>& _population, uint32_t _beg, uint32_t _num,
							size_t _training_size,
							DATATYPE* _working_training_input_data_ptr,
							DATATYPE* _working_training_output_data_ptr,
							std::array<DATATYPE*, thread_size>& _dev_mat_temp_rnvec,
							std::array<DATATYPE*, thread_size>& _dev_mat_temp_w,
							std::array<DATATYPE*, thread_size>& _dev_mat_ones,
							std::array<std::array<DATATYPE*, LAYER_SIZE>, thread_size>& _dev_mat_temp_layers) {
		// build job queue
		for (uint32_t i = 0; i < _num; ++i) {
			_eval_indexes_queue.emplace(_beg + i);
		}
		
		for (uint32_t i = 0; i < thread_size; ++i) {
			threads[i] = std::thread(handle_thread<population_size, generation_size, thread_size>,
										_device_id, i, std::ref(_eval_indexes_queue),
										std::ref(_population), _beg, _num,
										_training_size,
										_working_training_input_data_ptr,
										_working_training_output_data_ptr,
										std::ref(_dev_mat_temp_rnvec),
										std::ref(_dev_mat_temp_w),
										std::ref(_dev_mat_ones),
										std::ref(_dev_mat_temp_layers));
		}
		std::for_each(threads.begin(), threads.end(), [](std::thread& t){ t.join(); });
	}
};
#endif	// MFEA_HPP
