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
#include <stdlib.h>
#include <string.h> 
#include <cmath>
#include <CL/opencl.h>
#include "OCLKernel.h"

#define MAX_SOURCE_SIZE (0x100000)

// OpenCL Objects
cl_device_id     m_hDevices[32];
cl_int           m_currentDevice;
cl_context       m_hContext;
cl_command_queue m_hQueue;
cl_kernel        m_hKernelTest;
cl_kernel        m_hKernelPerformances; 
cl_kernel        m_hKernelFrequencies;

cl_mem m_hDeviceSeries;
cl_mem m_hDeviceObjects;
cl_mem m_hDevicePerformances;
cl_mem m_hDeviceFrequencies;

// Business attributes
float* m_series;
int    m_nbSeries;

float* m_objects;
int    m_nbObjects;

int*   m_frequencies;
int    m_nbFrequencies;

/******************************************************************************/
//	"Polar" version without trigonometric calls
float randn_notrig(float mu, float sigma) 
{
   static bool deviateAvailable=false;	//	flag
   static float storedDeviate;			//	deviate from previous calculation
   float polar, rsquared, var1, var2;

   //	If no deviate has been stored, the polar Box-Muller transformation is 
   //	performed, producing two independent normally-distributed random
   //	deviates.  One is stored for the next round, and one is returned.
   if (!deviateAvailable) {

      //	choose pairs of uniformly distributed deviates, discarding those 
      //	that don't fall within the unit circle
      do {
         var1=2.f*( float(rand())/float(RAND_MAX) ) - 1.f;
         var2=2.f*( float(rand())/float(RAND_MAX) ) - 1.f;
         rsquared=var1*var1+var2*var2;
      } while ( rsquared>=1.0 || rsquared == 0.0);

      //	calculate polar tranformation for each deviate
      polar=sqrt(-2.f*log(rsquared)/rsquared);

      //	store first deviate and set flag
      storedDeviate=var1*polar;
      deviateAvailable=true;

      //	return second deviate
      return var2*polar*sigma + mu;
   }

   //	If a deviate is available from a previous call to this function, it is
   //	returned, and the flag is set to false.
   else {
      deviateAvailable=false;
      return storedDeviate*sigma + mu;
   }
}

/*
 * getErrorDesc
 */
char* getErrorDesc(int err)
{
   switch (err)
   {
   case CL_SUCCESS                        : return "CL_SUCCESS";
   case CL_DEVICE_NOT_FOUND               : return "CL_DEVICE_NOT_FOUND";
   case CL_COMPILER_NOT_AVAILABLE         : return "CL_COMPILER_NOT_AVAILABLE";
   case CL_MEM_OBJECT_ALLOCATION_FAILURE  : return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
   case CL_OUT_OF_RESOURCES               : return "CL_OUT_OF_RESOURCES";
   case CL_OUT_OF_HOST_MEMORY             : return "CL_OUT_OF_HOST_MEMORY";
   case CL_PROFILING_INFO_NOT_AVAILABLE   : return "CL_PROFILING_INFO_NOT_AVAILABLE";
   case CL_MEM_COPY_OVERLAP               : return "CL_MEM_COPY_OVERLAP";
   case CL_IMAGE_FORMAT_MISMATCH          : return "CL_IMAGE_FORMAT_MISMATCH";
   case CL_IMAGE_FORMAT_NOT_SUPPORTED     : return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
   case CL_BUILD_PROGRAM_FAILURE          : return "CL_BUILD_PROGRAM_FAILURE";
   case CL_MAP_FAILURE                    : return "CL_MAP_FAILURE";

   case CL_INVALID_VALUE                  : return "CL_INVALID_VALUE";
   case CL_INVALID_DEVICE_TYPE            : return "CL_INVALID_DEVICE_TYPE";
   case CL_INVALID_PLATFORM               : return "CL_INVALID_PLATFORM";
   case CL_INVALID_DEVICE                 : return "CL_INVALID_DEVICE";
   case CL_INVALID_CONTEXT                : return "CL_INVALID_CONTEXT";
   case CL_INVALID_QUEUE_PROPERTIES       : return "CL_INVALID_QUEUE_PROPERTIES";
   case CL_INVALID_COMMAND_QUEUE          : return "CL_INVALID_COMMAND_QUEUE";
   case CL_INVALID_HOST_PTR               : return "CL_INVALID_HOST_PTR";
   case CL_INVALID_MEM_OBJECT             : return "CL_INVALID_MEM_OBJECT";
   case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
   case CL_INVALID_IMAGE_SIZE             : return "CL_INVALID_IMAGE_SIZE";
   case CL_INVALID_SAMPLER                : return "CL_INVALID_SAMPLER";
   case CL_INVALID_BINARY                 : return "CL_INVALID_BINARY";
   case CL_INVALID_BUILD_OPTIONS          : return "CL_INVALID_BUILD_OPTIONS";
   case CL_INVALID_PROGRAM                : return "CL_INVALID_PROGRAM";
   case CL_INVALID_PROGRAM_EXECUTABLE     : return "CL_INVALID_PROGRAM_EXECUTABLE";
   case CL_INVALID_KERNEL_NAME            : return "CL_INVALID_KERNEL_NAME";
   case CL_INVALID_KERNEL_DEFINITION      : return "CL_INVALID_KERNEL_DEFINITION";
   case CL_INVALID_KERNEL                 : return "CL_INVALID_KERNEL";
   case CL_INVALID_ARG_INDEX              : return "CL_INVALID_ARG_INDEX";
   case CL_INVALID_ARG_VALUE              : return "CL_INVALID_ARG_VALUE";
   case CL_INVALID_ARG_SIZE               : return "CL_INVALID_ARG_SIZE";
   case CL_INVALID_KERNEL_ARGS            : return "CL_INVALID_KERNEL_ARGS";
   case CL_INVALID_WORK_DIMENSION         : return "CL_INVALID_WORK_DIMENSION";
   case CL_INVALID_WORK_GROUP_SIZE        : return "CL_INVALID_WORK_GROUP_SIZE";
   case CL_INVALID_WORK_ITEM_SIZE         : return "CL_INVALID_WORK_ITEM_SIZE";
   case CL_INVALID_GLOBAL_OFFSET          : return "CL_INVALID_GLOBAL_OFFSET";
   case CL_INVALID_EVENT_WAIT_LIST        : return "CL_INVALID_EVENT_WAIT_LIST";
   case CL_INVALID_OPERATION              : return "CL_INVALID_OPERATION";
   case CL_INVALID_GL_OBJECT              : return "CL_INVALID_GL_OBJECT";
   case CL_INVALID_BUFFER_SIZE            : return "CL_INVALID_BUFFER_SIZE";
   case CL_INVALID_MIP_LEVEL              : return "CL_INVALID_MIP_LEVEL";
   default: return "unknown";
   }
}

/*
 * CheckStatus
 */
void CheckStatus( int status ) 
{
   if( status != CL_SUCCESS ) {
      //printf("*** Something went wrong!!: %s ***\n", getErrorDesc(status) );
   }
}

/*
 * OCLKernel constructor
 */
void OCL_intialize( 
   int device,
   int platform,
   int nbSeries, 
   int nbObjects, 
   int nbFrequencies ) 
{
   m_currentDevice = device;
   m_hContext      = nullptr;
   m_hQueue        = nullptr;
   m_series        = nullptr;
   m_objects       = nullptr;
   m_frequencies   = nullptr;
   m_nbObjects     = nbObjects; 
   m_nbSeries      = nbSeries;
   m_nbFrequencies = nbFrequencies;

   cl_int  status(0);
   // Find out how many devices there are
   cl_platform_id platform_id[10];
   cl_uint ret_num_devices;
   cl_uint ret_num_platforms;
   CheckStatus(clGetPlatformIDs(10, platform_id, &ret_num_platforms));
   CheckStatus(clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_ALL, 1, m_hDevices, &ret_num_devices));
   m_hContext = clCreateContext(0, 1, &m_hDevices[m_currentDevice], 0, 0, &status );
   CheckStatus(status);
   m_hQueue = clCreateCommandQueue(m_hContext, m_hDevices[m_currentDevice], 0, &status);
   CheckStatus(status);
}

int OCL_getFrequency(int index)
{
   if( index<m_nbFrequencies )
   {
      return m_frequencies[index];
   }
   return 0;
}

/*
 * compileKernels
 */
void OCL_compileKernels( wchar_t* source )
{
   cl_int status(0);
   size_t source_size(0);
   char buffer[2048];
   size_t len;
 
   // Read kernel from Excel sheet
   size_t wlen(wcslen(source));
   char* src = new char[wlen+1];
   int c(0);
   for( size_t i(0); i<wlen; ++i )
   {
      if( source[i]>=32 || source[i]==10 || source[i]==13 )
      {
         src[c++] = (char)source[i];
      }
   }
   src[c] = 0;
   source_size = strlen(src);

   // Build the performances kernel
   cl_program hProgram = clCreateProgramWithSource( m_hContext, 1, (const char **)&src, (const size_t*)&source_size, &status );
   CheckStatus(status);

   delete [] src;

   CheckStatus(clBuildProgram(hProgram, 1, &m_hDevices[m_currentDevice], 0, 0, 0));
   CheckStatus(clGetProgramBuildInfo(hProgram, m_hDevices[m_currentDevice], CL_PROGRAM_BUILD_LOG, 2048*sizeof(char), &buffer, &len ));
   printf( "%s", buffer );

   m_hKernelTest = clCreateKernel( hProgram, "test", &status );
   CheckStatus(status);

   m_hKernelPerformances = clCreateKernel( hProgram, "performances", &status );
   CheckStatus(status);

   // Build the frequencies kernel
   m_hKernelFrequencies = clCreateKernel( hProgram, "frequencies", &status );
   CheckStatus(status);

   CheckStatus(clReleaseProgram(hProgram));
}

void OCL_initializeDevices( float mu, float sigma )
{
   // Setup host memory
   m_objects = new float[m_nbObjects*m_nbSeries];
#pragma omp parallel
   for( int i(0); i<m_nbObjects*m_nbSeries; ++i )
   {
      m_objects[i] = randn_notrig( mu, sigma );
   }

   m_frequencies = new int[m_nbFrequencies];
   m_series = new float[m_nbSeries];

   // Setup device memory
   m_hDeviceObjects      = clCreateBuffer( m_hContext, CL_MEM_READ_ONLY,  sizeof(float)*m_nbObjects*m_nbSeries, 0, NULL);
   m_hDeviceSeries       = clCreateBuffer( m_hContext, CL_MEM_READ_WRITE, sizeof(float)*m_nbSeries,             0, NULL);
   m_hDeviceFrequencies  = clCreateBuffer( m_hContext, CL_MEM_READ_WRITE, sizeof(int)  *m_nbFrequencies,        0, NULL);
}

/*
 *
 */
void OCL_destroy()
{
   // Clean up
   CheckStatus(clReleaseMemObject(m_hDeviceSeries));
   CheckStatus(clReleaseMemObject(m_hDeviceObjects));
   CheckStatus(clReleaseMemObject(m_hDevicePerformances));
   CheckStatus(clReleaseMemObject(m_hDeviceFrequencies));

   CheckStatus(clReleaseKernel(m_hKernelTest));
   CheckStatus(clReleaseKernel(m_hKernelPerformances));
   CheckStatus(clReleaseKernel(m_hKernelFrequencies));

   CheckStatus(clReleaseCommandQueue(m_hQueue));
   CheckStatus(clReleaseContext(m_hContext));

   delete m_objects;     m_objects     = nullptr;
   delete m_frequencies; m_frequencies = nullptr;
   delete m_series;      m_series      = nullptr;
}

/*
 * runKernel
 */
void OCL_runKernel( float range )
{
   cl_int status(0);
   size_t global_item_size(0);

#if 0
   // --------------------------------------------------------------------------------
   // Run performance kernel
   // --------------------------------------------------------------------------------
   CheckStatus(clSetKernelArg( m_hKernelTest, 0, sizeof(cl_mem), (void*)&m_hDeviceFrequencies ));

   // Run the kernel!!
   global_item_size = m_nbFrequencies;
   CheckStatus(clEnqueueNDRangeKernel(
      m_hQueue, m_hKernelTest, 1, NULL, &global_item_size, 0, 0, NULL, NULL));

   // Series
   memset( m_frequencies, 0, sizeof(int)*m_nbFrequencies );
   CheckStatus(clEnqueueReadBuffer(
      m_hQueue, m_hDeviceFrequencies, CL_FALSE, 0, sizeof(int)*m_nbFrequencies, m_frequencies, 0, NULL, NULL));

#else

   // --------------------------------------------------------------------------------
   // Run performance kernel
   // --------------------------------------------------------------------------------
   CheckStatus(clSetKernelArg( m_hKernelPerformances, 0, sizeof(cl_mem), (void*)&m_hDeviceSeries ));
   CheckStatus(clSetKernelArg( m_hKernelPerformances, 1, sizeof(cl_mem), (void*)&m_hDeviceObjects ));
   CheckStatus(clSetKernelArg( m_hKernelPerformances, 2, sizeof(int),    (void*)&m_nbObjects ));
   CheckStatus(clSetKernelArg( m_hKernelPerformances, 3, sizeof(cl_mem), (void*)&m_hDevicePerformances ));

   // Objects
   CheckStatus(clEnqueueWriteBuffer(
      m_hQueue, m_hDeviceObjects, CL_FALSE, 0, sizeof(float)*m_nbObjects*m_nbSeries, m_objects, 0, NULL, NULL));

   // Run the kernel!!
   global_item_size = m_nbSeries;
   CheckStatus(clEnqueueNDRangeKernel(
      m_hQueue, m_hKernelPerformances, 1, NULL, &global_item_size, 0, 0, NULL, NULL));

   // --------------------------------------------------------------------------------
   // Rune frequencies kernel
   // --------------------------------------------------------------------------------

   CheckStatus(clSetKernelArg( m_hKernelFrequencies, 0, sizeof(cl_mem), (void*)&m_hDeviceSeries ));
   CheckStatus(clSetKernelArg( m_hKernelFrequencies, 1, sizeof(cl_mem), (void*)&m_hDeviceFrequencies ));
   CheckStatus(clSetKernelArg( m_hKernelFrequencies, 2, sizeof(int),    (void*)&m_nbFrequencies ));
   CheckStatus(clSetKernelArg( m_hKernelFrequencies, 3, sizeof(float),  (void*)&range ));
   
   // Objects
   memset( m_frequencies, 0, sizeof(int)*m_nbFrequencies );
   CheckStatus(clEnqueueWriteBuffer(m_hQueue, m_hDeviceFrequencies, CL_FALSE, 0, sizeof(int)*m_nbFrequencies, m_frequencies, 0, NULL, NULL));

   // Run the kernel!!
   global_item_size = m_nbSeries;
   CheckStatus(clEnqueueNDRangeKernel(m_hQueue, m_hKernelFrequencies, 1, NULL, &global_item_size, 0, 0, NULL, NULL));

   // ------------------------------------------------------------
   // Read back the results
   // ------------------------------------------------------------

   // Frequencies
   CheckStatus(clEnqueueReadBuffer(m_hQueue, m_hDeviceFrequencies, CL_FALSE, 0, sizeof(int)*m_nbFrequencies, m_frequencies, 0, NULL, NULL));
#endif // 0

   CheckStatus(clFlush(m_hQueue));
   CheckStatus(clFinish(m_hQueue));
}

