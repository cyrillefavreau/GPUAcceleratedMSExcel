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

#pragma once

// The number of functions we are making available to worksheets
#define rgFuncsRows 1

// Define the Excel interface to the xll
// For each function we define:
// name (i.e. the function name in Excel
// arguments
// function to call
static LPWSTR rgFuncs[rgFuncsRows][7] = 
{
    {L"GPUMonteCarlo",      L"UJJBBJJJF%",    L"GPUMonteCarlo",       L"Computing platform, Mu, Sigma, Number of days, Number of series, Size of ResultSet, Kernel" }
};
