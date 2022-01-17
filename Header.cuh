# pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include "math.h"

#define DEBUG

#ifdef DEBUG
#define CUDA_CALL(F) if((F) != cudaSuccess)\
		{printf("Margo::Cuda Error %s at %s: %d\n", cudaGetErrorString(cudaGetLastError()), __FILE__,__LINE__-1);\
		exit(-1);}
#define CUDA_CHECK() if(cudaPeekAtLastError() != cudaSuccess)\
		{printf("Margo::Cuda Error %s at %s: %d\n", cudaGetErrorString(cudaGetLastError()), __FILE__,__LINE__-1);\
		exit(-1);}
#else
#define CUDA_CALL(F) (F)
#define CUDA_CHECK()
#endif

static const int HEADERSIZE = 54;

static const int WIDTH_8K = 7680;
static const  int HEIGHT_8K = 4320;
static const int WIDTH_4K = 3840;
static const  int HEIGHT_4K = 2160;
static const int WIDTH_HD = 1920;
static const  int HEIGHT_HD = 1080;
static const int WORDS_PER_BLOCK = 4;
static const int PIXELS_PER_BLOCK = 6;
static const int WORDS_8K = WIDTH_8K* HEIGHT_8K * WORDS_PER_BLOCK / PIXELS_PER_BLOCK;
static const int COMPONENT_SIZE = 0x3FF;								//10-bits color component
static const int COLOR_OFFSET = 512;                                    //0 - level for U and V
static const int BLACK_LEVEL = 64;										//0 - level for Y
//static const int WHITE_LEVEL = 940;



enum Resolution {Res_8K, Res_4K, Res_HD};
