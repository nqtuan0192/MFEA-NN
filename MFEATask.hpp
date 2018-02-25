#ifndef MFEA_TASK_HPP
#define MFEA_TASK_HPP

#include <type_traits>
#include <cassert>
#include <algorithm>

#define INPUT_SIZE	8
#define OUTPUT_SIZE	1

#define NONE_LAYER 	0
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



#define TASK_SIZE	3
#define LAYER_SIZE	4	// number of layers = number of hidden layers + 1
#define TASKINDEX_1		0
#define TASKINDEX_2		1
#define TASKINDEX_3		2
#define TASKINDEX_LARGEST	TASKINDEX_1
#define TASKINDEX_SMALLEST	TASKINDEX_3
const uint32_t TASK_NUMBEROF_LAYERS[TASK_SIZE] = {3, 4, 3};
const uint32_t TASK_LAYERSIZES[TASK_SIZE][LAYER_SIZE + 1] =
						{ {INPUT_SIZE, 3, 3, OUTPUT_SIZE, NONE_LAYER},
						  {INPUT_SIZE, 3, 2, 2, OUTPUT_SIZE}, 
						  {INPUT_SIZE, 3, 4, OUTPUT_SIZE, NONE_LAYER}
						};


/** Return the number of layers for corresponding task
 */
inline static uint32_t getNumberofLayersbyTask(uint32_t task) {
	return TASK_NUMBEROF_LAYERS[task];
}

/** Return the number of layers for the unified task
 */
inline static uint32_t getUnifiedNumberofLayers() {
	return *std::max_element(std::begin(TASK_NUMBEROF_LAYERS), std::end(TASK_NUMBEROF_LAYERS));
}

/** Return the number of units for each layer of corresponding task
 */
inline static uint32_t getNumberofUnitsbyTaskLayer(uint32_t task, uint32_t layer) {
	return TASK_LAYERSIZES[task][layer];
}

inline static uint32_t getNumberofUnitsofLastLayerbyTask(uint32_t task) {
	return getNumberofUnitsbyTaskLayer(task, getNumberofLayersbyTask(task));
}

inline static uint32_t getNumberofUnitsofLastHiddenLayerbyTask(uint32_t task) {
	return getNumberofUnitsbyTaskLayer(task, getNumberofLayersbyTask(task) - 1);
}

inline static uint32_t getMaximumNumberofUnitsofUnifiedLayer(uint32_t layer) {
	uint32_t nunits = 0;
	if (layer == LAYER_SIZE - 1) {
		for (uint32_t task = 0; task < TASK_SIZE; ++task) {
			if (getNumberofUnitsofLastHiddenLayerbyTask(task) > nunits) {
				nunits = getNumberofUnitsofLastHiddenLayerbyTask(task);
			}
		}
	} else {
		for (uint32_t task = 0; task < TASK_SIZE; ++task) {
			if (getNumberofUnitsbyTaskLayer(task, layer) > nunits) {
				nunits = getNumberofUnitsbyTaskLayer(task, layer);
			}
		}
	}
	return nunits;
}

inline static uint32_t getMaximumLayerWeightsandBiasesbyLayer(uint32_t layer) {
	assert(layer > 0);
	return getMaximumNumberofUnitsofUnifiedLayer(layer) * (getMaximumNumberofUnitsofUnifiedLayer(layer - 1) + 1);
}

inline static uint32_t getMaximumLayerWeightsandBiasesatAll() {
	uint32_t n = 0;
	for (uint32_t layer = 1; layer <= LAYER_SIZE; ++layer) {
		if (n < getMaximumNumberofUnitsofUnifiedLayer(layer) * (getMaximumNumberofUnitsofUnifiedLayer(layer - 1) + 1)) {
			n = getMaximumNumberofUnitsofUnifiedLayer(layer) * (getMaximumNumberofUnitsofUnifiedLayer(layer - 1) + 1);
		}
	}
	return n;
}

inline static uint32_t getTotalLayerWeightsandBiases() {
	uint32_t sum = 0;
	for (uint32_t layer = 1; layer <= LAYER_SIZE; ++layer) {
		sum += getMaximumNumberofUnitsofUnifiedLayer(layer) * (getMaximumNumberofUnitsofUnifiedLayer(layer - 1) + 1);
	}
	return sum;
}



inline static uint32_t getUnifiedLayerOffset(uint32_t layer) {
	assert(layer > 0);
	if (layer <= 1) {
		return 0;
	} else {
		return getUnifiedLayerOffset(layer - 1) + getMaximumLayerWeightsandBiasesbyLayer(layer - 1);
	}
}

inline static uint32_t getUnifiedBiasOffset(uint32_t layer) {
	assert(layer > 0);
	if (layer < 1) {
		return getMaximumNumberofUnitsofUnifiedLayer(1) * getMaximumNumberofUnitsofUnifiedLayer(1 - 1);
	} else {
		return getUnifiedLayerOffset(layer) + getMaximumNumberofUnitsofUnifiedLayer(layer) * getMaximumNumberofUnitsofUnifiedLayer(layer - 1);
	}
}

inline static uint32_t getLayerOffset(uint32_t task, uint32_t layer) {
	assert(layer <= getNumberofLayersbyTask(task));
	if ((getNumberofLayersbyTask(task) < getUnifiedNumberofLayers()) && (getNumberofLayersbyTask(task) == layer)) {
		// if this task has less number of layers than unified topo and layer is the last layer of this task
		// then return the last unified layer
		return getUnifiedLayerOffset(getUnifiedNumberofLayers());
	} else {
		return getUnifiedLayerOffset(layer);
	}
}

inline static uint32_t getBiasOffset(uint32_t task, uint32_t layer) {
	assert(layer <= getNumberofLayersbyTask(task));
	if ((getNumberofLayersbyTask(task) < getUnifiedNumberofLayers()) && (getNumberofLayersbyTask(task) == layer)) {
		// if this task has less number of layers than unified topo and layer is the last layer of this task
		// then return the last unified layer
		return getUnifiedBiasOffset(getUnifiedNumberofLayers());
	} else {
		return getUnifiedBiasOffset(layer);
	}
}

inline static std::tuple<uint32_t, size_t> getLayerWeightsbyTaskLayer(uint32_t task, uint32_t layer) {	// return offset and size
	return std::make_tuple<uint32_t, size_t>(getLayerOffset(task, layer), getNumberofUnitsbyTaskLayer(task, layer) * getNumberofUnitsbyTaskLayer(task, layer - 1));
}

inline static std::tuple<uint32_t, size_t> getLayerBiasesbyTaskLayer(uint32_t task, uint32_t layer) {	// return offset and size
	return std::make_tuple<uint32_t, size_t>(getBiasOffset(task, layer), getNumberofUnitsbyTaskLayer(task, layer));
}

inline static std::tuple<uint32_t, size_t> getLayerWeightsandBiasesbyTaskLayer(uint32_t task, uint32_t layer) {	// return offset and size
	return std::make_tuple<uint32_t, size_t>(getLayerOffset(task, layer), getNumberofUnitsbyTaskLayer(task, layer) * (getNumberofUnitsbyTaskLayer(task, layer - 1) + 1));
}

#endif	// MFEA_TASK_HPP
