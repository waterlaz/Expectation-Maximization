CXX       = g++
CXX_FLAGS = -Wall -Wextra -std=c++20 -ggdb

INCLUDE_FLAGS = -I ../include `pkg-config --cflags eigen3`

HEADERS = ../include/*hpp

all: test gaussianEM_example

test: test.cpp $(HEADERS)
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) $< -o $@ 

gaussianEM_example: gaussianEM_example.cpp $(HEADERS)
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) $< -o $@ 

clean:
	rm test gaussianEM_example

