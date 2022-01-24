
#include "Header.cuh"

// read raw(v210)
int readFrame(char* decklinkFile, unsigned int* words) {

    FILE* input = fopen(decklinkFile, "rb");
    size_t result;

    if (input == NULL) { printf("Open %s file error!\n", decklinkFile); return -1; }

    result = fread(words, sizeof(unsigned int), WORDS_8K, input);

    if (result != WORDS_8K) { printf("Reading %s file error!\n", decklinkFile); return -1; }

    fclose(input);
    return 0;
}

// write bitmap
int writeBitmap(char* filename, uchar3* rgb, int width, int height)
{
    int Real_width = width * 3 + width % 4;		/*	Calculation of actual width to fit 4-byte boundary */
    FILE* Out_Fp;
    unsigned char* Bmp_Data;					/*  Stores one line of image data */
    char filename_EXT[256];

    sprintf(filename_EXT, "%s.bmp", filename);

    Out_Fp = fopen(filename_EXT, "wb");

    if (Out_Fp == NULL) { printf("File %s openning error!\n", filename); return -1; }
    /* Dynamically allocate array area. If it fails, output an error message and exit */
    if ((Bmp_Data = (unsigned char*)calloc(Real_width, sizeof(unsigned char))) == NULL) { printf("Memorry error!\n"); return -1; }

    /* Preparation of header information */
    unsigned char Bmp_headbuf[HEADERSIZE];                      /* Variable for storing header */
    unsigned int Bmp_info_header_size = 40;                     /* Information header size = 40 */
    unsigned int Bmp_header_size = HEADERSIZE;                  /* Header size = 54 */
    long Bmp_height;                                            /* Height (pixels) */
    long Bmp_width;                                             /* Width (pixels) */
    unsigned short Bmp_planes = 1;                              /* Plane number always 1 */
    unsigned short Bmp_color = 24;                              /* Color (bit) 24 */
    long Bmp_comp = 0;                                          /* Compression method 0 */
    long Bmp_xppm = 0;                                          /* Horizontal resolution (ppm) */
    long Bmp_yppm = 0;                                          /* Vertical resolution (ppm) */
    long Bmp_image_size = height * Real_width;                  /* File size of image part (bytes) */
    unsigned long Bmp_size = Bmp_image_size + HEADERSIZE;       /* bmp file size (bytes) */

    Bmp_headbuf[0] = 'B'; Bmp_headbuf[1] = 'M';
    memcpy(Bmp_headbuf + 2, &Bmp_size, sizeof(Bmp_size));
    Bmp_headbuf[6] = Bmp_headbuf[7] = Bmp_headbuf[8] = Bmp_headbuf[9] = 0;
    memcpy(Bmp_headbuf + 10, &Bmp_header_size, sizeof(Bmp_header_size));
    memcpy(Bmp_headbuf + 14, &Bmp_info_header_size, sizeof(Bmp_info_header_size));
    long a = width;
    long b = height;
    memcpy(Bmp_headbuf + 18, &a, sizeof(Bmp_width));
    memcpy(Bmp_headbuf + 22, &b, sizeof(Bmp_height));
    memcpy(Bmp_headbuf + 26, &Bmp_planes, sizeof(Bmp_planes));
    memcpy(Bmp_headbuf + 28, &Bmp_color, sizeof(Bmp_color));
    memcpy(Bmp_headbuf + 30, &Bmp_comp, sizeof(Bmp_comp));
    memcpy(Bmp_headbuf + 34, &Bmp_image_size, sizeof(Bmp_image_size));
    memcpy(Bmp_headbuf + 38, &Bmp_xppm, sizeof(Bmp_xppm));
    memcpy(Bmp_headbuf + 42, &Bmp_yppm, sizeof(Bmp_yppm));

    Bmp_headbuf[46] = Bmp_headbuf[47] = Bmp_headbuf[48] = Bmp_headbuf[49] = 0;
    Bmp_headbuf[50] = Bmp_headbuf[51] = Bmp_headbuf[52] = Bmp_headbuf[53] = 0;

    // Write header information
    fwrite(Bmp_headbuf, sizeof(unsigned char), HEADERSIZE, Out_Fp);

    for (long i = 0; i < height; i++) {
        for (long j = 0; j < width; j++) {
            Bmp_Data[j * 3 + 0] = rgb[width * (height - i - 1) + j].z;      // B
            Bmp_Data[j * 3 + 1] = rgb[width * (height - i - 1) + j].y;      // G
            Bmp_Data[j * 3 + 2] = rgb[width * (height - i - 1) + j].x;  	// R
            //img++;							// A
        }
        for (long j = width * 3; j < Real_width; j++) {
            Bmp_Data[j] = 0;
        }
        fwrite(Bmp_Data, sizeof(unsigned char), Real_width, Out_Fp);
    }

    free(Bmp_Data);
    fclose(Out_Fp);

    return 0;
}

int createBitmap(char* filename, uchar3* rgb, int width, int height) {

    if (writeBitmap(filename, rgb, width, height) != 0) { printf("WriteBmp error in createBitmap!\n"); return -1; }

    return 0;
}


// devecie method 
__device__ void calcAndSetBitmap(short* Y, short* U, short* V, uchar3* pixel) {
    short R, G, B;

    R = *Y * 1.0f * (1024.0f / 876.0f) + *V * 1.5748f * (1024.0f / 896.0f);
    G = *Y * 1.0f * (1024.0f / 876.0f) + *U * (-0.1873f) * (1024.0f / 896.0f) + *V * (-0.4681f) * (1024.0f / 896.0f);
    B = *Y * 1.0f * (1024.0f / 876.0f) + *U * 1.8556f * (1024.0f / 896.0f);

    R = (R > 1023) ? 1023 : (R < 0) ? 0 : R;
    G = (G > 1023) ? 1023 : (G < 0) ? 0 : G;
    B = (B > 1023) ? 1023 : (B < 0) ? 0 : B;


    pixel->x = (unsigned char)(R >> 2); //0;
    pixel->y = (unsigned char)(G >> 2); //255;
    pixel->z = (unsigned char)(B >> 2); //0;
}

// kernel method
__global__ void kernelYUVtoRGB(uchar3* bitmap, int N, unsigned int* words) {

    int indWithinTheGrid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = gridDim.x * blockDim.x;

    short int U, Y, V;
    int limit = N / WORDS_PER_BLOCK;

    for (int i = indWithinTheGrid; i < limit; i += gridStride) {

        int offset422 = i * WORDS_PER_BLOCK;
        int ind = i * PIXELS_PER_BLOCK;

        U = (short int)((words[offset422 + 0] >> 0) & COMPONENT_SIZE) - COLOR_OFFSET;
        Y = (short int)((words[offset422 + 0] >> 10) & COMPONENT_SIZE) - BLACK_LEVEL;
        V = (short int)((words[offset422 + 0] >> 20) & COMPONENT_SIZE) - COLOR_OFFSET;

        calcAndSetBitmap(&Y, &U, &V, &(bitmap[ind + 0]));			//RGB[i + 0]

        Y = (short int)((words[offset422 + 1] >> 0) & COMPONENT_SIZE) - BLACK_LEVEL;

        calcAndSetBitmap(&Y, &U, &V, &(bitmap[ind + 1]));			//RGB[i + 1]

        U = (short int)((words[offset422 + 1] >> 10) & COMPONENT_SIZE) - COLOR_OFFSET;
        Y = (short int)((words[offset422 + 1] >> 20) & COMPONENT_SIZE) - BLACK_LEVEL;
        V = (short int)((words[offset422 + 2] >> 0) & COMPONENT_SIZE) - COLOR_OFFSET;

        calcAndSetBitmap(&Y, &U, &V, &(bitmap[ind + 2]));			//RGB[i + 2]

        Y = (short int)((words[offset422 + 2] >> 10) & COMPONENT_SIZE) - BLACK_LEVEL;

        calcAndSetBitmap(&Y, &U, &V, &(bitmap[ind + 3]));			//RGB[i + 3]

        U = (short int)((words[offset422 + 2] >> 20) & COMPONENT_SIZE) - COLOR_OFFSET;
        Y = (short int)((words[offset422 + 3] >> 0) & COMPONENT_SIZE) - BLACK_LEVEL;
        V = (short int)((words[offset422 + 3] >> 10) & COMPONENT_SIZE) - COLOR_OFFSET;

        calcAndSetBitmap(&Y, &U, &V, &(bitmap[ind + 4]));			//RGB[i + 4]

        Y = (short int)((words[offset422 + 3] >> 20) & COMPONENT_SIZE) - BLACK_LEVEL;

        calcAndSetBitmap(&Y, &U, &V, &(bitmap[ind + 5]));			//RGB[i + 5]
    }
}

void convertYUV2RGB(unsigned int* d_wordsIn, unsigned int* h_wordsIn,
    uchar3* d_rgbOut, uchar3* h_rgbOut,
    int words, int sizeIn, int sizeOut)
{
    // time
    cudaEvent_t start, stop, startK, stopK, startCpyIn, stopCpyIn, startCpyOut, stopCpyOut;
    CUDA_CALL(cudaEventCreate(&start));        CUDA_CALL(cudaEventCreate(&stop));
    CUDA_CALL(cudaEventCreate(&startK));       CUDA_CALL(cudaEventCreate(&stopK));
    CUDA_CALL(cudaEventCreate(&startCpyIn));   CUDA_CALL(cudaEventCreate(&stopCpyIn));
    CUDA_CALL(cudaEventCreate(&startCpyOut));  CUDA_CALL(cudaEventCreate(&stopCpyOut));

    CUDA_CALL(cudaEventRecord(start));
    CUDA_CALL(cudaEventRecord(startCpyIn));

    CUDA_CALL(cudaMemcpy(d_wordsIn, h_wordsIn, sizeIn, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaEventRecord(stopCpyIn));     CUDA_CALL(cudaEventSynchronize(stopCpyIn));
    CUDA_CALL(cudaEventRecord(startK));
    
    // call kernel convert yuv(v210) to rbg 
    kernelYUVtoRGB << < 1024, 1024 >> > (d_rgbOut, words, d_wordsIn);
    CUDA_CHECK();
    CUDA_CALL(cudaEventRecord(stopK));     CUDA_CALL(cudaEventSynchronize(stopK));
    CUDA_CALL(cudaEventRecord(startCpyOut));

    CUDA_CALL(cudaMemcpy(h_rgbOut, d_rgbOut, sizeOut, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaEventRecord(stopCpyOut));  CUDA_CALL(cudaEventSynchronize(stopCpyOut));

    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaEventRecord(stop));  CUDA_CALL(cudaEventSynchronize(stop));
    
    // process time
    float timeT, timeK, timeCpyIn, timeCpyOut;
    CUDA_CALL(cudaEventElapsedTime(&timeT, start, stop));
    CUDA_CALL(cudaEventElapsedTime(&timeCpyIn, startCpyIn, stopCpyIn));
    CUDA_CALL(cudaEventElapsedTime(&timeK, startK, stopK));
    CUDA_CALL(cudaEventElapsedTime(&timeCpyOut, startCpyOut, stopCpyOut));
    printf("\t%f\t%f\t%f\t%f \n", timeT, timeK, timeCpyIn, timeCpyOut);

    CUDA_CALL(cudaEventDestroy(start));        CUDA_CALL(cudaEventDestroy(stop));
    CUDA_CALL(cudaEventDestroy(startK));       CUDA_CALL(cudaEventDestroy(stopK));
    CUDA_CALL(cudaEventDestroy(startCpyIn));   CUDA_CALL(cudaEventDestroy(stopCpyIn));
    CUDA_CALL(cudaEventDestroy(startCpyOut));  CUDA_CALL(cudaEventDestroy(stopCpyOut));
}


int colourConverter(char* decklinkFile)
{
    Resolution mode = Res_8K;
    unsigned int width, height;

    switch (mode) {
    case Res_8K:
        width = WIDTH_8K; height = HEIGHT_8K;
        break;
    case Res_4K:
        width = WIDTH_4K; height = HEIGHT_4K; break;
    case Res_HD:
        width = WIDTH_HD; height = HEIGHT_HD; break;
    }

    unsigned int words = width * height * 4 / 6;
    size_t sizeIn = words * sizeof(unsigned int);
    size_t sizeOut = width * height * sizeof(uchar3);

    unsigned int* h_wordsIn, * d_wordsIn; // cpu - side and gpu - side input data
    uchar3* h_rgbOut, * d_rgbOut; // cpu - side and gpu - side output data

    h_wordsIn = (unsigned int*)malloc(sizeIn);
    h_rgbOut = (uchar3*)malloc(sizeOut);

    if ((h_wordsIn == NULL) || (h_rgbOut == NULL)) { printf("Memorry error!\n"); return -1; }
    if (readFrame(decklinkFile, h_wordsIn) != 0) { printf("readDecklinkFrame error!\n"); return -1; };

    CUDA_CALL(cudaMalloc(&d_wordsIn, sizeIn));
    CUDA_CALL(cudaMalloc(&d_rgbOut, sizeOut));


    printf("Time, ms : Total \t Kernel \t CopyIn \t CopyOut \n");
    int i = 0;
    while (i < 120) {
        ++i;
        convertYUV2RGB(d_wordsIn, h_wordsIn, d_rgbOut, h_rgbOut, words, sizeIn, sizeOut);
    }

    if ((createBitmap("sample02", h_rgbOut, width, height)) != 0) { printf("createBitmap error!\n"); return -1; }

    //free memory
    CUDA_CALL(cudaFree(d_wordsIn));
    CUDA_CALL(cudaFree(d_rgbOut));
    free(h_wordsIn);
    free(h_rgbOut);

    return 0;
}



int main()
{
    if(colourConverter("sample02.raw")) {
        printf(" ERROR!\n");
        return -1;
    }
    
    printf("it is ok!\n");
    return 0;
}
