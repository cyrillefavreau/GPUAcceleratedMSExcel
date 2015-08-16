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
#include <ctype.h>
#include <windows.h>
#include <math.h>
#include "xlcall.h"

#include "ExcelCUDA.h"
#ifdef CUDA
#include "ExcelCUDA_wrapper.h"
#else
#include "OCLKernel.h"
#endif // CUDA

BOOL APIENTRY DllMain(HMODULE hModule,
                      DWORD  ul_reason_for_call,
                      LPVOID lpReserved
                      )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

// Callback function that must be implemented and exported by every valid XLL.
// The xlAutoOpen function is the recommended place from where to register XLL
// functions and commands, initialize data structures, customize the user
// interface, and so on.
__declspec(dllexport) int WINAPI xlAutoOpen(void)
{
    XLOPER12 xDLL;
    int i;

    // Get the name of the XLL
    Excel12f(xlGetName, &xDLL, 0);

    // Register each of the functions
    for (i = 0 ; i < rgFuncsRows ; i++) 
    {
        Excel12f(xlfRegister, 0, 5,
            (LPXLOPER12)&xDLL,
            (LPXLOPER12)TempStr12(rgFuncs[i][0]),
            (LPXLOPER12)TempStr12(rgFuncs[i][1]),
            (LPXLOPER12)TempStr12(rgFuncs[i][2]),
            (LPXLOPER12)TempStr12(rgFuncs[i][3]));
    }

    // Free the XLL filename
    Excel12f(xlFree, 0, 1, (LPXLOPER12)&xDLL);

    // Return 1 => success
    return 1;
}

// Called by Microsoft Office Excel whenever the XLL is deactivated. The add-in
// is deactivated when an Excel session ends normally. The add-in can be
// deactivated by the user during an Excel session, and this function will be
// called in that case.
__declspec(dllexport) int WINAPI xlAutoClose(void)
{
    int i;

    // Delete all the registered functions
    for (i = 0 ; i < rgFuncsRows ; i++)
        Excel12f(xlfSetName, 0, 1, TempStr12(rgFuncs[i][2]));

    // Return 1 => success
    return 1;
}

// Called by Microsoft Office Excel whenever the user activates the XLL during
// an Excel session by using the Add-In Manager. This function is not called
// when Excel starts up and loads a pre-installed add-in.
__declspec(dllexport) int WINAPI xlAutoAdd(void)
{
    const size_t bufsize = 255;
    const size_t dllsize = 100;
    LPWSTR szBuf = (LPWSTR)malloc(bufsize * sizeof(WCHAR));
    LPWSTR szDLL = (LPWSTR)malloc(dllsize * sizeof(WCHAR));
    XLOPER12 xDLL;
    
    // Get the name of the XLL
    Excel12f(xlGetName, &xDLL, 0);
    wcsncpy_s(szDLL, dllsize, xDLL.val.str + 1, xDLL.val.str[0]);
    szDLL[xDLL.val.str[0]] = (WCHAR)NULL;
    
    // Display dialog
    swprintf_s((LPWSTR)szBuf, 255, L"Adding %s\nBuild %hs - %hs",
        szDLL,
        __DATE__, __TIME__);
    Excel12f(xlcAlert, 0, 2, TempStr12(szBuf), TempInt12(2));

    // Free the XLL filename
    Excel12f(xlFree, 0, 1, (LPXLOPER12)&xDLL);

    free(szBuf);
    free(szDLL);
    return 1;
}

// Called by Microsoft Office Excel just after an XLL worksheet function
// returns an XLOPER/XLOPER12 to it with a flag set that tells it there is
// memory that the XLL still needs to release. This enables the XLL to return
// dynamically allocated arrays, strings, and external references to the
// worksheet without memory leaks.
__declspec(dllexport) void WINAPI xlAutoFree12(LPXLOPER12 pxFree)
{
    if(pxFree->xltype & xltypeMulti)
    {
        int size = pxFree->val.array.rows *
            pxFree->val.array.columns;
        LPXLOPER12 p = pxFree->val.array.lparray;

        for(; size-- > 0; p++)
            if(p->xltype == xltypeStr)
                free(p->val.str);

        free(pxFree->val.array.lparray);
    }
    else if(pxFree->xltype & xltypeStr)
    {
        free(pxFree->val.str);
    }
    else if(pxFree->xltype & xltypeRef)
    {
        free(pxFree->val.mref.lpmref);
    }
    free(pxFree);
}


// Called by Microsoft Office Excel when the Add-in Manager is invoked for the
// first time in an Excel session. This function is used to provide the Add-In
// Manager with information about your add-in.
_declspec(dllexport) LPXLOPER12 WINAPI xlAddInManagerInfo12(LPXLOPER12 xAction)
{
    LPXLOPER12 pxInfo;
    XLOPER12 xIntAction;

    pxInfo = (LPXLOPER12)malloc(sizeof(XLOPER12));

    Excel12f(xlCoerce, &xIntAction, 2, xAction, TempInt12(xltypeInt));
    if(xIntAction.val.w == 1) 
    {
        LPWSTR szDesc = (LPWSTR)malloc(50 * sizeof(WCHAR));
        swprintf_s(szDesc, 50, L"%s", L"\020Example CUDA XLL");
        pxInfo->xltype = xltypeStr;
        pxInfo->val.str = szDesc;
    }
    else 
    {
        pxInfo->xltype = xltypeErr;
        pxInfo->val.err = xlerrValue;
    }

    pxInfo->xltype |= xlbitDLLFree;
    return pxInfo;
}

// getNumberOfRows
// Helper function to get the number of rows in an XLOPER12.
int getNumberOfRows(LPXLOPER12 px)
{
    int n = -1;
    XLOPER12 xMulti;

    switch(px->xltype)
    {
    case xltypeNum:
        n = 1;
        break;
    case xltypeRef:
    case xltypeSRef:
    case xltypeMulti:
        // Multi value, coerce it into a readable form
        if (Excel12f(xlCoerce, &xMulti, 2, px, TempInt12(xltypeMulti)) != xlretUncalced)
        {
            n = xMulti.val.array.rows;
        }
        Excel12f(xlFree, 0, 1, (LPXLOPER12)&xMulti);
        break;
    }
    return n;
}

// extractData
// Helper function for to extract the data from an XLOPER12 into an
// array of n floats. If the XLOPER12 contains a single value then it
// is replicated into all n elements of the array. Otherwise the
// XLOPER12 must contain exactly n rows and one column and the data
// is copied directly into the array.
int extractData(LPXLOPER12 px, int n, float *pdst, int *error)
{
    int ok = 1;
    int i;
    XLOPER12 xMulti;

    switch(px->xltype)
    {
    case xltypeNum:
        // If there is only one value, copy it into each element of
        // the array.
        for (i = 0 ; i < n ; i++)
        {
            pdst[i] = (float)px->val.num;
        }
        break;
    case xltypeRef:
    case xltypeSRef:
    case xltypeMulti:
        // Multi value, coerce it into a readable form
        if (Excel12f(xlCoerce, &xMulti, 2, px, TempInt12(xltypeMulti)) != xlretUncalced)
        {
            // Check number of columns
            if (xMulti.val.array.columns != 1)
                ok = 0;
            if (ok)
            {
                // Check number of rows
                if (xMulti.val.array.rows == 1)
                {
                    for (i = 0 ; i < n ; i++)
                    {
                        pdst[i] = (float)xMulti.val.array.lparray[0].val.num;
                    }
                }
                else if (xMulti.val.array.rows == n)
                {
                    // Extract data into the array
                    for (i = 0 ; ok && i < n ; i++)
                    {
                        switch (xMulti.val.array.lparray[i].xltype)
                        {
                        case xltypeNum:
                            pdst[i] = (float)xMulti.val.array.lparray[i].val.num;
                            break;
                        case xltypeErr:
                            *error = xMulti.val.array.lparray[i].val.err;
                            ok = 0;
                            break;
                        case xltypeMissing:
                            *error = xlerrNum;
                            ok = 0;
                            break;
                        default:
                            *error = xlerrRef;
                            ok = 0;
                        }
                    }
                }
                else
                    ok = 0;
            }
        }
        else
            ok = 0;
        Excel12f(xlFree, 0, 1, (LPXLOPER12)&xMulti);
        break;
    default:
        ok = 0;
    }
    return ok;
}

/**
* @brief  GPUMonteCarlo
* @return 
* @param  dllexport
*/
__declspec(dllexport) LPXLOPER12 WINAPI GPUMonteCarlo(
   int         device,
   int         platform,
   double      mu,
   double      sigma,
   int         nbPerformances,
   int         nbSeries,
   int         nbFrequencies,
   wchar_t*    sourceCode )
{
   LPXLOPER12 result = 0;
   if( nbPerformances != 0 && nbSeries != 0 && nbFrequencies != 0 ) 
   {
      LPXLOPER12 pxRes = (LPXLOPER12) malloc (sizeof(XLOPER12));

      int ok = 1;
      int n = -1;
      int i;
      int error = -1;

#ifdef CUDA
      int*   frequencies  = (int*)malloc(nbFrequencies*sizeof(int));
      float* performances = (float*)malloc(nbPerformances*sizeof(float));
      ok = (performances && frequencies);
#endif // CUDA

      // Run the montecarlo function
      mu = powf(1 + mu, 1 / nbPerformances) - 1;
      sigma = sigma / (nbPerformances*nbPerformances);

      if( ok ) 
      {
#if CUDA
         ExcelCUDA_MonteCarlo(
            nbPerformances, nbFrequencies, nbSeries, 
            3, mu, sigma, rand(), 64, 
            performances, frequencies );
#else
         OCL_intialize( device, platform, nbSeries, nbPerformances, nbFrequencies );
         OCL_initializeDevices( (float)mu, (float)sigma );
         OCL_compileKernels(sourceCode);
         OCL_runKernel( 3.f );
#endif
      }

      // If pricing more than one option then allocate memory for result XLOPER12
      if( ok )
      {
         if ((pxRes->val.array.lparray = (LPXLOPER12)malloc(nbFrequencies*sizeof(XLOPER12))) == NULL)
            ok = 0;
      }

      // Copy the result into the XLOPER12
      if (ok) 
      {
         if (nbFrequencies > 1) 
         {
            for (i = 0 ; i < nbFrequencies ; i++) 
            {
#ifdef CUDA
               pxRes->val.array.lparray[i].val.num = (double)frequencies[i];
#else
               pxRes->val.array.lparray[i].val.num = (double)OCL_getFrequency(i);
#endif // CUDA
               pxRes->val.array.lparray[i].xltype = xltypeNum;
               pxRes->val.array.rows    = nbFrequencies;
               pxRes->val.array.columns = 1;
            }
            pxRes->xltype = xltypeMulti;
            pxRes->xltype |= xlbitDLLFree;
         }
         else 
         {
#ifdef CUDA
            pxRes->val.num = frequencies[0];
#else
            pxRes->val.num = OCL_getFrequency(0);
#endif // CUDA
            pxRes->xltype = xltypeNum;
            pxRes->xltype |= xlbitDLLFree;
         }
      }
      else 
      {
         pxRes->val.err = (error < 0) ? xlerrValue : error;
         pxRes->xltype = xltypeErr;
         pxRes->xltype |= xlbitDLLFree;
      }

#ifdef CUDA
      // Cleanup
      if (frequencies) free(frequencies);
      if (performances) free(performances);
#else
      OCL_destroy();
#endif // CUDA

      result = pxRes;
   }

   // Note that the pxRes will be freed when Excel calls xlAutoFree12
   return result;
}
