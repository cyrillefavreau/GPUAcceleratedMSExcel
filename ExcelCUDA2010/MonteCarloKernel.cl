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

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable
int positionInRange( float value, float range, int steps ) { 
   float v = value-(-range/2); return (v/(range/steps)); 
}

__kernel void test( __global int* frequencies ) {
   frequencies[get_global_id(0)]=get_global_id(0);
}

__kernel void frequencies( __global float* series, __global int*   frequencies, int nbFrequencies, float range ) { 
   int index=get_global_id(0); 
   int position = positionInRange( series[index], range, nbFrequencies ); 
   if( position>=0 && position<nbFrequencies){ 
      atom_inc(&frequencies[position]);
   }
}

__kernel void performances( __global float* series, __global float* objects, int nbObjects, __global float* performances ) {
   int index=get_global_id(0)*nbObjects;
   float currentPerformance=1.0;
   float previousPerformance=0.0;
   for( int i=0; i<nbObjects; ++i ) {
      previousPerformance = currentPerformance;
	  currentPerformance=(1.0+objects[index+i])*previousPerformance;
   } 
   series[get_global_id(0)] = currentPerformance - 1.0;
}
