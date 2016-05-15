CFLAGS	= $(shell pkg-config --cflags opencv)
CFLAGS  +=  -I./include
CXXFLAGS = --compiler-options "-std=c++11"
LIBS		= $(shell pkg-config --libs opencv)
LIBS	  += -L/usr/local/cuda/lib -lcudart

KERNELS = 	src/kernels/ID1.cu \
				src/kernels/ID2.cu \
				src/kernels/ID3.cu \
				src/kernels/ID4.cu \


all: detect
	
run: all test.jpg
	@echo "\nRunning...\n"
	./detect test.jpg
	
clean:
	- rm detect *.o
	
main.o: src/main.cpp src/IntegralImage.h src/ImageProcessing.h src/detectFaces.h
	@echo Compiling main.cpp
	@nvcc -c src/main.cpp $(CXXFLAGS)

ImageProcessing.o: src/ImageProcessing.cpp src/ImageProcessing.h
	@echo "Compiling ImageProcessing.cpp"
	@nvcc -c src/ImageProcessing.cpp $(CXXFLAGS)

WindowInfo.o: src/WindowInfo.cpp src/WindowInfo.h
	@echo "Compiling WindowInfo.cpp"
	@nvcc -c src/WindowInfo.cpp $(CFLAGS)

IntegralImage.o: src/IntegralImage.cu src/IntegralImage.h cudaCheckError.o
	@echo "Compiling IntegralImage.cu"
	@nvcc -c src/IntegralImage.cu $(CFLAGS)

cudaCheckError.o: src/cudaCheckError.cu
	@echo "Compiling cudaCheckError.cu"
	@nvcc -c src/cudaCheckError.cu $(CFLAGS)
	
detectFaces.o: src/detectFaces.h src/detectFaces.cu $(KERNELS)
	@echo "Compiling detectFaces.cu"
	@nvcc -c src/detectFaces.cu $(CFLAGS)

detect: main.o IntegralImage.o ImageProcessing.o detectFaces.o cudaCheckError.o WindowInfo.o
	@echo "\nLinking..."
	@nvcc main.o IntegralImage.o ImageProcessing.o detectFaces.o cudaCheckError.o WindowInfo.o $(LIBS) -o detect