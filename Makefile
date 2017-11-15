program_NAME := mfea

program_C_SRCS   := $(wildcard *.c)
program_CXX_SRCS := $(wildcard *.cpp)
program_CU_SRCS  := $(wildcard *.cu)

program_C_OBJS   := ${program_C_SRCS:.c=.o}
program_CXX_OBJS := ${program_CXX_SRCS:.cpp=.o}
program_CU_OBJS  := ${program_CU_SRCS:.cu=.o}
program_OBJS     := $(program_C_OBJS) $(program_CXX_OBJS) $(program_CU_OBJS)

program_INCLUDE_DIRS :=	/usr/local/cuda/include
program_LIBRARY_DIRS :=	/usr/local/cuda/lib64
program_LIBRARIES    := curand cublas pthread cudnn
#curand_static culibos	# using static linking of curand library for better performance

CPPFLAGS += $(foreach includedir,$(program_INCLUDE_DIRS),-I$(includedir))
LDFLAGS += $(foreach librarydir,$(program_LIBRARY_DIRS),-L$(librarydir))
LDFLAGS += $(foreach library,$(program_LIBRARIES),-l$(library))

CCFLAGS  =
CXXFLAGS =	-std=c++11	--default-stream per-thread	-O2
#--default-stream per-thread	# stream concurrency

CC     =	g++
CXX    =	g++

SM_ARCH  =      -arch=sm_50
NVCC     =      /usr/local/cuda/bin/nvcc

.PHONY: all clean distclean

all: $(program_NAME)

debug: CXXFLAGS = -DDEBUG	-g	-std=c++11	-lineinfo
debug: $(program_NAME)

$(program_NAME): $(program_OBJS)
	$(NVCC) $(SM_ARCH) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $(program_OBJS) -o $(program_NAME)

$(program_CU_OBJS): $(program_CU_SRCS)
	$(NVCC) $(SM_ARCH) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) -c $(program_CU_SRCS)


clean:
	@- $(RM) $(program_NAME)
	@- $(RM) $(program_OBJS)

distclean: clean
