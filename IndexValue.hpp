#ifndef INDEX_VALUE_H
#define INDEX_VALUE_H

template<typename INDEX_TYPE, typename VALUE_TYPE> struct IndexValue {
	INDEX_TYPE index;
	VALUE_TYPE value;
	
   bool operator<(const IndexValue<INDEX_TYPE, VALUE_TYPE>& other) const {
        return value < other.value;
    }
};

#endif	// INDEX_VALUE_H
