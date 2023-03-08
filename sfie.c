/*
  This is v0.1.0 of the code for the C implementation of the SFIE model

  Guest, D. R., ..., and Carney, L. H. (202x). ...

  The code in this file is based on the models described in:

  Nelson, P. C., & Carney, L. H. (2004). A phenomenological model of peripheral and central 
  neural responses to amplitude-modulated tones. The Journal of the Acoustical Society of 
  America, 116(4), 2173-2186.
  
  And the MATLAB code provided in:
  https://www.urmc.rochester.edu/MediaLibraries/URMCMedia/labs/carney-lab/codes/UR_EAR_2020a.zip

  Please cite these papers if you publish any research results obtained with this code or 
  any modified versions of this code.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/**
 * get_alpha_norm
 *
 * Returns filter coefficients to implement an "alpha function" filter
 *
 * @param tau Time constant of alpha function (s)
 * @param fs Sampling rate (Hz)
 * @param t Should always be 1.0? unclear
 * @param B Vector of feedforward filter coefficients, size (2, )
 * @param A Vector of feedback filter coefficients, size (3, )
 */
void get_alpha_norm(double tau, double fs, double t, double *B, double *A) {
    /* Declare variables */
    double a, norm, scale;

    /* Calculate variables needed to compute coefficients */
    a = exp(-1.0 / (fs*tau));
    norm = 1.0 / (pow(tau, 2.0) * (exp(-t/tau) * (-t/tau - 1.0) + 1.0));
    scale = fs * 1/norm;

    /* Compute coefficients and store in B, A */
    B[0] = 0.0;
    B[1] = a;
    A[0] = 1.0 * scale;
    A[1] = -2.0 * a * scale;
    A[2] = pow(a, 2.0) * scale;
}

/**
 * filter_alpha
 * 
 * Applies "alpha function" filter to data at given sample index
 * 
 * @param x Vector of input data, size (totalstim, )
 * @param n Current sample index
 * @param fs Sampling rate (Hz)
 * @param B Vector of feedforward filter coefficients, size (2, )
 * @param A Vector of feedback filter coefficients, size (3, )
 * @param y Vector of filter outputs, size (totalstim, )
 */
void filter_alpha(double *x, int n, double fs, double *B, double *A, double *y) {
    if (n == 0) {
        y[n] = (1/A[0]) * (B[0]*x[n]);
    } else if (n == 1) {
        y[n] = (1/A[0]) * (B[0]*x[n] + B[1]*x[n-1] - A[1]*y[n-1]);
    } else {
        y[n] = (1/A[0]) * (B[0]*x[n] + B[1]*x[n-1] - A[1]*y[n-1] - A[2]*y[n-2]);
    }
}