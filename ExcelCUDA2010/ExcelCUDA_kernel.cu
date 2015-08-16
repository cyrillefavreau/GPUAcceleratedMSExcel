/*
 * GPU Accelerated MSExcel functions
 * Copyright (C) 2011-2015 Cyrille Favreau <cyrille_favreau@hotmail.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * Author: Cyrille Favreau <cyrille_favreau@hotmail.com>
 *
 */

#include <cuda.h>
#include "ExcelCUDA_kernel.h"

/**
* @brief 
*/
__global__ void kernel_performanceStorage( float* objects, float* series, float* performances, int nbObjects )
{
   // Compute the index
   unsigned int x     = blockIdx.x*blockDim.x+threadIdx.x;
   unsigned int y     = blockIdx.y*blockDim.y+threadIdx.y;
   unsigned int index = (y*blockDim.x) + x;

   int objectsIndex = index*nbObjects;

   // Compute performance
   __shared__ float localPerformance[2];
   localPerformance[0] = 1.f; // current performance
   localPerformance[1] = 0.f; // previous performance
   for( int i(0); i<nbObjects; ++i ) {
      localPerformance[1] = localPerformance[0];
      localPerformance[0] = (1.0+objects[objectsIndex+i])*localPerformance[1];

      if( index == 0 ) performances[i] = localPerformance[0];
   }
   // Store performance
   series[index] = localPerformance[0] - 1.0;
}

/**
* @brief Kernel function to be executed on the GPU
* @param ptr Pointer to an array of floats stored in GPU memory
*/
__global__ void kernel_frequencies( float* series, int* frequencies, float range, int nbFrequencies )
{
   // Compute the index
   unsigned int x     = blockIdx.x*blockDim.x+threadIdx.x;
   unsigned int y     = blockIdx.y*blockDim.y+threadIdx.y;
   unsigned int index = (y*blockDim.x) + x;

   float v = series[index]-(-range/2);
   int position = (v/(range/nbFrequencies));
   if( position>=0 && position<nbFrequencies) {
      atomicAdd(&frequencies[position],1);
   }
}
