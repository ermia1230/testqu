CXX = nvq++
TARGET_BACKEND ?= qpp-cpu
BUILD_DIR ?= build
EXE := $(BUILD_DIR)/cudaq-benchmark-$(TARGET_BACKEND)

CXXFLAGS ?= -std=c++20 -O3
NVQXXFLAGS ?= --target $(TARGET_BACKEND)
SOURCES := benchmark.cpp circuits.cpp

.PHONY: all clean

all: $(EXE)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(EXE): $(SOURCES) circuits.hpp kernels.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(NVQXXFLAGS) $(SOURCES) -o $(EXE)

clean:
	rm -rf $(BUILD_DIR)
