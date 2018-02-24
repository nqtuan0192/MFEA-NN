#ifndef MFEA_TASK_HPP
#define MFEA_TASK_HPP

#include <type_traits>

#define NONE_LAYER 0
#define OFFSET_IDX	0
#define SIZE_IDX	1

/*
 * layer 0 : input layer, cannot query for weights and biases size
 * layer l (0 < l < LAYER_SIZE) : hidden layers
 * layer LAYER_SIZE : output layer
 * 
 * */


/*
#define TASK_SIZE	1
#define LAYER_SIZE	1	// number of layers = number of hidden layers + 1

#define TASKINDEX_1		0
#define TASKINDEX_LARGEST	TASKINDEX_1
#define TASKINDEX_SMALLEST	TASKINDEX_1
const uint32_t TASK_NUMBEROF_LAYERS[TASK_SIZE] = {1};
const uint32_t TASK_LAYERSIZES[TASK_SIZE][LAYER_SIZE + 1] = { {INPUT_SIZE, OUTPUT_SIZE} // always the largest layer size
															 };
*/


/*
#define TASK_SIZE	5
#define LAYER_SIZE	2	// number of layers = number of hidden layers + 1

#define TASKINDEX_1		0
#define TASKINDEX_2		1
#define TASKINDEX_3		2
#define TASKINDEX_4		3
#define TASKINDEX_5		4
#define TASKINDEX_LARGEST	TASKINDEX_1
#define TASKINDEX_SMALLEST	TASKINDEX_4

const uint32_t TASK_NUMBEROF_LAYERS[TASK_SIZE] = {2, 2, 2, 2, 1};
const uint32_t TASK_LAYERSIZES[TASK_SIZE][LAYER_SIZE + 1] = { {INPUT_SIZE, 256, OUTPUT_SIZE}, // always the largest layer size
															{INPUT_SIZE, 200, OUTPUT_SIZE}, 
															{INPUT_SIZE, 128, OUTPUT_SIZE},
															{INPUT_SIZE, 64, OUTPUT_SIZE},
															{INPUT_SIZE, OUTPUT_SIZE, NONE_LAYER}
															 };
*/


/* good tasks
 * 3 tasks: 64-32, 64, 0
 * 3 tasks: 64-32-16, 64-32, 64
 * */

#define TASK_SIZE	3
#define LAYER_SIZE	2	// number of layers = number of hidden layers + 1
#define TASKINDEX_1		0
#define TASKINDEX_2		1
#define TASKINDEX_3		2
#define TASKINDEX_LARGEST	TASKINDEX_1
#define TASKINDEX_SMALLEST	TASKINDEX_3
const uint32_t TASK_NUMBEROF_LAYERS[TASK_SIZE] = {2, 2, 2};
const uint32_t TASK_LAYERSIZES[TASK_SIZE][LAYER_SIZE + 1] = { {INPUT_SIZE, 7, OUTPUT_SIZE}, // always the largest layer size
															{INPUT_SIZE, 6, OUTPUT_SIZE}, 
															{INPUT_SIZE, 5, OUTPUT_SIZE} 
														  };


inline static uint32_t getNumberofLayersbyTask(uint32_t task) {
	return TASK_NUMBEROF_LAYERS[task];
}

inline static uint32_t getNumberofUnitsbyTaskLayer(uint32_t task, uint32_t layer) {
	return TASK_LAYERSIZES[task][layer];
}

inline static uint32_t getNumberofUnitsofLastLayerbyTask(uint32_t task) {
	return getNumberofUnitsbyTaskLayer(task, getNumberofLayersbyTask(task));
}

inline static uint32_t getMaximumLayerWeightsandBiasesbyLayer(uint32_t layer) {
	assert(layer > 0);
	return TASK_LAYERSIZES[TASKINDEX_LARGEST][layer] * (TASK_LAYERSIZES[TASKINDEX_LARGEST][layer - 1] + 1);
}

inline static uint32_t getMaximumLayerWeightsandBiasesatAll() {
	return TASK_LAYERSIZES[TASKINDEX_LARGEST][1] * (TASK_LAYERSIZES[TASKINDEX_LARGEST][1 - 1] + 1);
}

inline static uint32_t getTotalLayerWeightsandBiases() {
	uint32_t sum = 0;
	for (uint32_t layer = 1; layer <= LAYER_SIZE; ++layer) {
		sum += TASK_LAYERSIZES[TASKINDEX_LARGEST][layer] * (TASK_LAYERSIZES[TASKINDEX_LARGEST][layer - 1] + 1);
	}
	return sum;
}

inline static uint32_t getLayerOffset(uint32_t layer) {
	assert(layer > 0);
	if (layer <= 1) {
		return 0;
	} else {
		return getLayerOffset(layer - 1) + getMaximumLayerWeightsandBiasesbyLayer(layer - 1);
	}
}

inline static uint32_t getBiasOffset(uint32_t layer) {
	assert(layer > 0);
	if (layer <= 1) {
		return TASK_LAYERSIZES[TASKINDEX_LARGEST][1] * (TASK_LAYERSIZES[TASKINDEX_LARGEST][1 - 1]);
	} else {
		return getLayerOffset(layer) + TASK_LAYERSIZES[TASKINDEX_LARGEST][layer] * (TASK_LAYERSIZES[TASKINDEX_LARGEST][layer - 1]);
	}
}

inline static std::tuple<uint32_t, size_t> getLayerWeightsbyTaskLayer(uint32_t task, uint32_t layer) {	// return offset and size
	return std::make_tuple<uint32_t, size_t>(getLayerOffset(layer), TASK_LAYERSIZES[task][layer] * TASK_LAYERSIZES[task][layer - 1]);
}

inline static std::tuple<uint32_t, size_t> getLayerBiasesbyTaskLayer(uint32_t task, uint32_t layer) {	// return offset and size
	return std::make_tuple<uint32_t, size_t>(getBiasOffset(layer),
													TASK_LAYERSIZES[task][layer]);
}

inline static std::tuple<uint32_t, size_t> getLayerWeightsandBiasesbyTaskLayer(uint32_t task, uint32_t layer) {	// return offset and size
	return std::make_tuple<uint32_t, size_t>(getLayerOffset(layer), TASK_LAYERSIZES[task][layer] * (TASK_LAYERSIZES[task][layer - 1] + 1));
}







/*
 * template concept
// define neural network size for each task
#define TASK_0 
#define TASK_1 char[200]
#define TASK_2 char[200], char[50]
#define TASK_3 char[200], char[50], char[10]

template<class... hiddenlayers> struct MFEA_Task {
	static constexpr uint32_t numberof_layers = sizeof...(hiddenlayers);
	static constexpr uint32_t layers[sizeof...(hiddenlayers)] = {sizeof(hiddenlayers)...};
	
    MFEA_Task() = delete;
    MFEA_Task(const MFEA_Task&) = delete;
    MFEA_Task(MFEA_Task&&) = delete;
};

#define MFEA_Task<TASK_0> MFEA_Task_NONE_HIDDENLAYER
#define MFEA_Task<TASK_1> MFEA_Task_ONE_HIDDENLAYER
#define MFEA_Task<TASK_2> MFEA_Task_TWO_HIDDENLAYERs
#define MFEA_Task<TASK_3> MFEA_Task_THREE_HIDDENLAYERs
*/
#endif	// MFEA_TASK_HPP
