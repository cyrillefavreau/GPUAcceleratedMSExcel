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

#include <stdio.h>
#include <cuda.h>
#include <curand.h>

#include "ExcelCUDA_wrapper.h"
#include "ExcelCUDA_kernel.h"

inline int success(cudaError_t result)
{
    return result == cudaSuccess;
}

/**
* @brief Run the kernel on the GPU. 
* This function is executed on the Host.
*/
void ExcelCUDA_MonteCarlo( 
   int    nbObjects, 
   int    nbFrequencies, 
   int    nbSeries, 
   float  range, 
   float  mu, 
   float  sigma, 
   float  random, 
   int    nbThreadsPerBlock,
   float* performances,
   int*   frequencies )
{
   // Device buffers
   float* _dObjects;
   float* _dSeries;
   float* _dPerformances;
   int*   _dFrequencies;

   // Device allocation
   cudaMalloc( (void**)&_dObjects, nbSeries*nbObjects*sizeof(float) );
   cudaMalloc( (void**)&_dSeries, nbSeries*sizeof(float) );
   cudaMalloc( (void**)&_dFrequencies, nbFrequencies*sizeof(int) );
   cudaMalloc( (void**)&_dPerformances, nbObjects*sizeof(float) );

   // Create pseudo-random number generator
   curandGenerator_t gen;
   curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
   curandSetPseudoRandomGeneratorSeed(gen, random);
   curandGenerateNormal(gen, _dObjects, nbSeries*nbObjects, mu, sigma );
   curandDestroyGenerator(gen);

   // Reset memory
   cudaMemset( _dFrequencies, 0, nbFrequencies*sizeof(int) );

   dim3 block(8, 8, 1);
   int gridDim = (int)sqrt(float(nbSeries));
   dim3 gridPerformances( gridDim/block.x, gridDim/block.y, 1);
   kernel_performanceStorage<<<gridPerformances,block>>>( _dObjects, _dSeries, _dPerformances, nbObjects  );

   // compute Frequencies 
   gridDim = (int)sqrt(float(nbSeries));
   dim3 gridFrequencies( gridDim/block.x, gridDim/block.y, 1);
   kernel_frequencies<<<gridFrequencies,block>>>( _dSeries, _dFrequencies, range, nbFrequencies );

   cudaMemcpy( performances, _dPerformances, nbObjects*sizeof(float),   cudaMemcpyDeviceToHost);
   cudaMemcpy( frequencies,  _dFrequencies,  nbFrequencies*sizeof(int), cudaMemcpyDeviceToHost);

   cudaFree( _dObjects );
   cudaFree( _dFrequencies );
   cudaFree( _dSeries );
   cudaFree( _dPerformances );
}
