 /*
  This is v5.6 of the code for subcortical auditory model model of:

  Guest, D. R., Farhadi, A., ..., and Carney, L. H. (202x). ...

  The peripheral stage of this model is derived from the following work, listed in 
  chronological order:

  Zilany, M. S., & Bruce, I. C. (2006). Modeling auditory-nerve responses for high sound 
  pressure levels in the normal and impaired auditory periphery. The Journal of the 
  Acoustical Society of America, 120(3), 1446-1466.

  * Zilany, M.S.A., Bruce, I.C., Nelson, P.C., and Carney, L.H. (2009). "A
  Phenomenological model of the synapse between the inner hair cell and auditory
  nerve : Long-term adaptation with power-law dynamics," Journal of the
  Acoustical Society of America 126(5): 2390-2412.
 
  Ibrahim, R. A., and Bruce, I. C. (2010). "Effects of peripheral tuning
  on the auditory nerve's representation of speech envelope and temporal fine
  structure cues," in The Neurophysiological Bases of Auditory Perception, eds.
  E. A. Lopez-Poveda and A. R. Palmer and R. Meddis, Springer, NY, pp. 429-438.

  Zilany, M.S.A., Bruce, I.C., Ibrahim, R.A., and Carney, L.H. (2013).
  "Improved parameters and expanded simulation options for a model of the
  auditory periphery," in Abstracts of the 36th ARO Midwinter Research Meeting.

  * Zilany, M. S., Bruce, I. C., & Carney, L. H. (2014). Updated parameters and expanded 
  simulation options for a model of the auditory periphery. The Journal of the Acoustical 
  Society of America, 135(1), 283-286.

  * Farhadi, ... (2023) ... TBD

  The peripheral stage was modified by Afagh Farhadi to include a sample-by-sample efferent 
  gain control loop, which is controlled by an auditory brainstem and midbrain model 
  included in this code. This efferent model was modified by Daniel Guest to allow for 
  "crosstalk" between different channels (i.e., efferent responses in one channel can 
  affect gain in others).

  Please cite these papers marked with asterisks (*) if you publish any research results 
  obtained with this code or any modified versions of this code.

  To compare this code to older model versions, please refer to changelog.txt and to the 
  Git commit history. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "complex.hpp"
#include "model.h"

#define MAXSPIKES 1000000
#ifndef TWOPI
#define TWOPI 6.28318530717959
#endif

#ifndef __max
#define __max(a,b) (((a) > (b))? (a): (b))
#endif

#ifndef __min
#define __min(a,b) (((a) < (b))? (a): (b))
#endif

/**
 * model_efferent_simple
 *
 * Simulates peripheral and subcortical responses to a sound-pressure waveform
 *
 * `model_efferent_simple` is a wrapper that provides a simplified interface to the full
 * model function (`model`). More info about the model and detailed descriptions of each
 * parameter in the function signature below are provided in the docstring for the main
 * model function.
 */
void model_efferent_wrapper(double *px, 
							double **randNums_hsr, 
							double **randNums_lsr, 
                            double *cf, 
							int n_chan, 
							double tdres, 
							int totalstim, 
							double *cohc, 
                            double *cihc, 
							int species,
							int powerlaw_mode,
                            double ic_tau_e, 
							double ic_tau_i, 
                            double ic_delay, 
							double ic_amp, 
							double ic_inh, 
							double moc_cutoff,
                            double moc_beta_wdr, 
							double moc_offset_wdr, 
							double moc_beta_ic, 
                            double moc_offset_ic, 
							double moc_weight_wdr, 
                            double moc_weight_ic, 
							double moc_width_wdr, 
                            int n_process,
                            double coef_slow,
                            double tau_short_slow,
                            double tau_long_slow,
                            double coef_fast,
                            double tau_short_fast,
                            double tau_long_fast,
							double **ihcout, 
                            double **anrateout_hsr, 
							double **anrateout_lsr, 
							double **icout,
                            double **gain) {

    /* First, allocate memory for all of the extra output matrices */
    double *controlout[n_chan], *c1out[n_chan], *c2out[n_chan], 
        *expout_hsr[n_chan], *sout1_hsr[n_chan], *sout2_hsr[n_chan], *synout_hsr[n_chan], 
        *expout_lsr[n_chan], *sout1_lsr[n_chan], *sout2_lsr[n_chan], *synout_lsr[n_chan],
        *cnout[n_chan], *mocwdr[n_chan], *mocic[n_chan];

    for (int i = 0; i < n_chan; i++) {
        controlout[i] = (double*) calloc(totalstim, sizeof(double));
        c1out[i] = (double*) calloc(totalstim, sizeof(double));
        c2out[i] = (double*) calloc(totalstim, sizeof(double));
        expout_hsr[i] = (double*) calloc(totalstim, sizeof(double));
        sout1_hsr[i] = (double*) calloc(totalstim, sizeof(double));
        sout2_hsr[i] = (double*) calloc(totalstim, sizeof(double));
        synout_hsr[i] = (double*) calloc(totalstim, sizeof(double));
        expout_lsr[i] = (double*) calloc(totalstim, sizeof(double));
        sout1_lsr[i] = (double*) calloc(totalstim, sizeof(double));
        sout2_lsr[i] = (double*) calloc(totalstim, sizeof(double));
        synout_lsr[i] = (double*) calloc(totalstim, sizeof(double));
        cnout[i] = (double*) calloc(totalstim, sizeof(double));
        mocwdr[i] = (double*) calloc(totalstim, sizeof(double));
        mocic[i] = (double*) calloc(totalstim, sizeof(double));
    }

    /* Call the main model function */
    model(px, randNums_hsr, randNums_lsr, cf,                    // principal inputs
          n_chan, tdres, totalstim, cohc, cihc, species, 100.0,  // IHC/AN parameters
          powerlaw_mode,                                         // powerlaw parameters
          0.5e-3, 2.0e-3, 1.0e-3, 1.5, 0.6,                      // CN parameters
          ic_tau_e, ic_tau_i, ic_delay, ic_amp, ic_inh,          // IC parameters
          moc_cutoff, moc_beta_wdr, moc_offset_wdr, 0.1, 1.0,    // MOC parameters
          moc_beta_ic, moc_offset_ic, 0.1, 1.0, moc_weight_wdr,  // ...
          moc_weight_ic, moc_width_wdr,                          // ...
          n_process, coef_slow, tau_short_slow, tau_long_slow, coef_fast, tau_short_fast, tau_long_fast,
          controlout, c1out, c2out, ihcout, expout_hsr,          // output matrices
          sout1_hsr, sout2_hsr, synout_hsr, expout_lsr,          // ...
          sout1_lsr, sout2_lsr, synout_lsr, anrateout_hsr,       // ...
          anrateout_lsr, cnout, icout, mocwdr, mocic, gain);     // ...

    /* Finally, free allocated memory */
    for (int i = 0; i < n_chan; i++) {
        free(controlout[i]);
        free(c1out[i]);
        free(c2out[i]);
        free(expout_hsr[i]);
        free(sout1_hsr[i]);
        free(sout2_hsr[i]);
        free(synout_hsr[i]);
        free(expout_lsr[i]);
        free(sout1_lsr[i]);
        free(sout2_lsr[i]);
        free(synout_lsr[i]);
        free(cnout[i]);
        free(mocwdr[i]);
        free(mocic[i]);
    }
}

/**
 * model
 *
 * Simulates peripheral and subcortical responses to a sound-pressure waveform
 *
 * Simulates peripheral (inner hair cell, auditory nerve) and subcortical (cochlear nucleus,
 * inferior colliculus) responses to a sound-pressure waveform using a modified version of
 * the periphery model of Zilany, Bruce, and Carney (2014) and a new brainstem and midbrain
 * model described in [[cite]].
 *
 * Arguments to the model function are structured and ordered in the following way:
 * - Inputs are passed as pointers to arrays or matrices of double-precision floating-point
 *   numbers
 * - Parameters are next passed as doubles, integers, and pointers to arrays of doubles.
 *   These parameters govern the behavior of the model.
 * - Outputs are passed as pointers to matrices of doubles. These matrices are modified
 *   in-place by the model function (i.e., changes will be visible to the caller) and are
 *   treated as "returns"
 *
 * Below, the arguments of the model function are described in more detail. For each
 * argument, information is presented in the following format:
 *
 *     @param [in/out] name (N, M) (unit)
 *
 * ... where [in/out] indicates whether the argument is treated as an input (i.e., [in]) or
 * an output (i.e., [out]), name is the name of the argument in the model code, (N, M)
 * indicates the size (if a vector or a matrix), and (unit) indicates the unit of the
 * argument (e.g., Hz) if it has a unit.  
 *
 * In all cases, matrices should have their first dimension equal in length to the number of
 * simulated channels (n_chan) and their second dimension equal to the number of simulated
 * time samples (totalstim). The sizes of other inputs/parameters are described below.
 *
 * In the documentation and code below, several abbreviations are commonplace. They are
 * provided here for convenience:
 * - HSR = high spontaneous rate (auditory-nerve fiber)
 * - LSR = low spontaneous rate (auditory-nerve fiber)
 * - CN = cochlear nucleus
 * - IC = inferior colliculus
 * - MOC = medial olivocochlear (neurons/system)
 *
 * @param [in] px (totalstim, ) (Pa) Vector containing sound-pressure waveform
 * @param [in] randNums_hsr (n_chan, totalstim) Matrix of fractional Gaussian noise for the
 * high-spontaneous-rate signal path, see 2014 paper above for details on how this noise
 * should be synthesized 
 * @param [in] randNums_lsr (n_chan, totalstim) Matrix of fractional Gaussian noise for the
 * low-spontaneous-rate signal path, see 2014 paper above for details on how this noise
 * should be synthesized 
 * @param [in] cf (n_chan, ) (Hz) Vector containing characteristic frequencies for each
 * channel. CFs should be within the range of [[vals]] for cats and [[vals]] for humans
 * @param [in] tdres (s) Sample time resolution, i.e., reciprocal of the sampling rate.
 * Suitable sampling rates are ~100 kHz for human simulations and 200 kHz for cat
 * simulations.
 * @param [in] totalstim Number of samples in the simulation
 * @param [in] cohc (n_chan, 1) Vector containing elements in [0, 1] controlling the 
 * Cohc value for each CF channel. This value is multiplied with the resulting "gain factor"
 * yielded by the MOC gain-control pathway. A value of 0 makes the model fully 
 * "hearing-impaired" and efferent gain control will have no impact on responses. 
 * A value of 1 with the efferent gain control disabled results in a fully "normal-hearing" 
 * response that should match data recorded in anesthetized auditory nerve. 
 * @param [in] cihc (n_chan, 1) Vector containing elements in [0, 1] controlling the 
 * Cihc value for each CF channel. This controls the amplitude of inner-hair-cell responses
 * produced by the C1 filter (it has no effect on responses produced by the C2 filter).
 * @param [in] species Integer indicating which species to simulate (1==cat, 2=human[shera],
 * 3==human[glasberg])
 * @param [in] powerlaw_include_fast Integer indicating whether to include or exclude the
 * fast power law adaptation stage in the auditory-nerve model
 * @param [in] powerlaw_len_memory Integer determining the length of the "memory" of the
 * power-law adpatation stage in the auditory-nerve model
 * @param [in] cn_tau_e (s) Excitatory time constant for SFIE cochlear nucleus stage,
 * reasonable values are in the range of [0.5e-3, 8e-3], default value is 0.5e-3
 * @param [in] cn_tau_i (s) Inhibitory time constant for SFIE cochlear nucleus stage,
 * reasonable values are in the range of [0.5e-3, 8e-3], but this time constant should
 * generally be slightly longer than the excitatory time constant, default value is 2.0e-3
 * @param [in] cn_delay (s) Inhibitory delay time for SFIE cochlear nucleus stage,
 * reasonable values are in the range of [0.5e-3, 4e-3], default value is 1.0e-3
 * @param [in] cn_amp Overall amplitude scalar for SFIE cochlear nucleus stage, reasonable
 * values are in the range of [1.0, 5.0], default value is 1.5
 * @param [in] cn_inh Inhibition strength for SFIE cochlear nucleus stage, reasonable values
 * are in the range of [0.5, 2.0], default value is 0.6
 * @param [in] ic_tau_e (s) Excitatory time constant for SFIE inferior colliculus stage,
 * reasonable values are in the range of [0.5e-3, 8e-3], default value is 1.0e-3
 * @param [in] ic_tau_i (s) Inhibitory time constant for SFIE inferior colliculus stage,
 * reasonable values are in the range of [0.5e-3, 8e-3], but this time constant should
 * generally be slightly longer than the excitatory time constant, default value is 2.0e-3
 * @param [in] ic_delay (s) Inhibitory delay time for SFIE inferior colliculus stage,
 * reasonable values are in the range of [0.5e-3, 4e-3], default value is 1.0e-3
 * @param [in] ic_amp Overall amplitude scalar for SFIE inferior colliculus stage,
 * reasonable values are in the range of [1.0, 5.0], default value is 2.0
 * @param [in] ic_inh Inhibition strength for SFIE inferior colliculus stage, reasonable
 * values are in the range of [0.5, 2.0], default value is 0.9
 * @param [in] moc_cutoff (Hz) Cutoff value for the simple IIR lowpass filter that is used
 * to lowpass filter MOC input signals before applying the MOC nonlinearity, default value
 * is 0.64 Hz
 * @param [in] moc_beta_wdr Control parameter in the WDR-MOC nonlinearity, governs the slope
 * of the nonlinearity such that smaller values yield more gradual slopes between MOC rate
 * and MOC output gain while larger values yield sharper slopes between MOC rate and MOC
 * output gain, default value is 0.01
 * @param [in] moc_offset_wdr (sp/s) Control parameter in the WDR-MOC nonlinearity, shifts
 * the nonlinearity along the x-axis such that gain control is not applied until the WDR-MOC
 * rate exceeds the offset, default value is 0.0
 * @param [in] moc_minrate_wdr Minimum output value for the WDR-MOC nonlinearity, should be
 * in range of [0, 1], default is 0.0
 * @param [in] moc_maxrate_wdr Maximum output value for the WDR-MOC nonlinearity, should be
 * in range of [0, 1], default is 1.0
 * @param [in] moc_beta_ic Control parameter in the IC-MOC nonlinearity, governs the slope
 * of the nonlinearity such that smaller values yield more gradual slopes between MOC rate
 * and MOC output gain while larger values yield sharper slopes between MOC rate and MOC
 * output gain, default value is 0.01
 * @param [in] moc_offset_ic (sp/s) Control parameter in the IC-MOC nonlinearity, shifts the
 * nonlinearity along the x-axis such that gain control is not applied until the IC-MOC rate
 * exceeds the offset, default value is 0.0
 * @param [in] moc_minrate_ic Minimum output value for the IC-MOC nonlinearity, should be in
 * range of [0, 1], default is 0.0
 * @param [in] moc_maxrate_ic Maximum output value for the IC-MOC nonlinearity, should be in
 * range of [0, 1], default is 1.0
 * @param [in] moc_weight_wdr Scalar weight applied to the WDR-MOC rate before it is passed
 * through the WDR-MOC nonlinearity and converted into a gain value. Setting this value to
 * 0.0 disables WDR-driven gain control, default is 1.0.
 * @param [in] moc_weight_ic Scalar weight applied to the IC-MOC rate before it is passed
 * through the IC-MOC nonlinearity and converted into a gain value. Setting this value to
 * 0.0 disables IC-driven gain control, default is 1.0.
 * @param [in] moc_width_wdr (oct) Range of CFs over which WDR-MOC gain control signal from
 * a single channel is sent to OHCs. For example, if moc_width_wdr is set to 1.0, then for a
 * given CF, the WDR-MOC gain control signal for that CF will be applied to the total gain
 * of channels within the range of [-1/2, 1/2] octaves around CF.
 * @param [out] controlout (n_chan, totalstim) Matrix to store output of control-path filter
 * @param [out] c1out (n_chan, totalstim) Matrix to store output of signal-path C1 filter
 * @param [out] c2out (n_chan, totalstim) Matrix to store output of signal-path C2 filter
 * @param [out] ihcout (n_chan, totalstim) Matrix to store output of inner hair cell
 * @param [out] expout_hsr (n_chan, totalstim) Matrix to store output of exponential
 * adaptation stage for high-spontaneous-rate auditory-nerve fiber
 * @param [out] sout1_hsr (n_chan, totalstim) Matrix to store output of slow power-law
 * adaptation stage for high-spontaneousrate auditory-nerve fiber
 * @param [out] sout2_hsr (n_chan, totalstim) Matrix to store output of fast power-law
 * adaptation stage for high-spontaneous-rate auditory-nerve fiber
 * @param [out] synout_hsr (n_chan, totalstim) Matrix to store output of synapse for
 * high-spontaneous-rate auditory-nerve fiber
 * @param [out] expout_lsr (n_chan, totalstim) Matrix to store output of exponential
 * adaptation stage for low-spontaneous-rate auditory-nerve fiber
 * @param [out] sout1_lsr (n_chan, totalstim) Matrix to store output of slow power-law
 * adaptation stage for low-spontaneousrate auditory-nerve fiber
 * @param [out] sout2_lsr (n_chan, totalstim) Matrix to store output of fast power-law
 * adaptation stage for low-spontaneous-rate auditory-nerve fiber
 * @param [out] synout_lsr (n_chan, totalstim) Matrix to store output of synapse for
 * low-spontaneous-rate auditory-nerve fiber
 * @param [out] anrateout_lsr (n_chan, totalstim) (sp/s) Matrix to store output
 * instantaneous firing rate of low-spontaneous-rate auditory-nerve fiber
 * @param [out] anrateout_hsr (n_chan, totalstim) (sp/s) Matrix to store output
 * instantaneous firing rate of high-spontaneous-rate auditory-nerve fiber
 * @param [out] cnout (n_chan, totalstim) (sp/s) Matrix to store output of cochlear nucleus
 * stage
 * @param [out] icout (n_chan, totalstim) (sp/s) Matrix to store output of inferior
 * colliculus stage totalstim)
 * @param [out] mocwdr (n_chan, totalstim) (sp/s) Matrix to store the WDR-MOC rates, which
 * are lowpass-filtered LSR rates
 * @param [out] mocic (n_chan, totalstim) (sp/s) Matrix to store the IC-MOC rates, which are
 * lowpass-filtered IC rates
 * @param [out] gain (n_chan, totalstim) Matrix to store time-varying "gain" value in [0, 1]
 */
void model(
    double *px, 
    double **randNums_hsr, 
    double **randNums_lsr, 
    double *cf, 
    int n_chan, 
    double tdres, 
    int totalstim, 
    double *cohc, 
    double *cihc, 
    int species, 
    double spont, 
    int powerlaw_mode,
    double cn_tau_e, 
    double cn_tau_i, 
    double cn_delay, 
    double cn_amp, 
    double cn_inh,
    double ic_tau_e, 
    double ic_tau_i, 
    double ic_delay, 
    double ic_amp, 
    double ic_inh,
    double moc_cutoff, 
    double moc_beta_wdr, 
    double moc_offset_wdr, 
    double moc_minrate_wdr, 
    double moc_maxrate_wdr, 
    double moc_beta_ic, 
    double moc_offset_ic, 
    double moc_minrate_ic, 
    double moc_maxrate_ic, 
    double moc_weight_wdr, 
    double moc_weight_ic, 
    double moc_width_wdr,
    int n_process,
    double coef_slow,
    double tau_short_slow,
    double tau_long_slow,
    double coef_fast,
    double tau_short_fast,
    double tau_long_fast,
    double **controlout, 
    double **c1out, 
    double **c2out, 
    double **ihcout, 
    double **expout_hsr, 
    double **sout1_hsr, 
    double **sout2_hsr, 
    double **synout_hsr,
    double **expout_lsr, 
    double **sout1_lsr, 
    double **sout2_lsr, 
    double **synout_lsr,
    double **anrateout_hsr, 
    double **anrateout_lsr, 
    double **cnout, 
    double **icout,
    double **mocwdr, 
    double **mocic, 
    double **gain
) {
    /* Declare variables used in parallel-exponential PLA approximation system */
    // Storing these values here temporarily 
    // int n_process = 40;                           // how many exponential processes in approx for PLA
    // double coef_slow = 6.48e-2 * (100.0/40.0);    // scalar coefficient used in approx for sout1
    // double tau_short_slow = 1e-4;                 // short time constant used in approx for sout1
    // double tau_long_slow = 1e7;                   // long time constant used in approx for sout1
    // double coef_fast = 1.285867e2 * (100.0/40.0); // scalar coefficient used in approx for sout2
    // double tau_short_fast = 0.0096;               // short time constant used in approx for sout2
    // double tau_long_fast = 6.5575e3;              // long time constant used in approx for sout2

    /* Declare pointers to store convenient references to LSR/HSR pathway stages */
    double** randNums[2] = {randNums_hsr, randNums_lsr};
    double** expout[2] = {expout_hsr, expout_lsr};
    double** sout1[2] = {sout1_hsr, sout1_lsr};
    double** sout2[2] = {sout2_hsr, sout2_lsr};
    double** synout[2] = {synout_hsr, synout_lsr};
    double** anrateout[2] = {anrateout_hsr, anrateout_lsr};

    /* Declare temporary variables for the BM/IHC stage outputs */
    double wbout1, wbout, ohcnonlinout, ohcout, tauc1, rsigma, wb_gain, c1vihctmp, 
           c2vihctmp, me_curr;

    /* Declare parameters in the BM/IHC stage (all varying by channel) */
    double bmplace[n_chan], centerfreq[n_chan], Taumin[n_chan], Taumax[n_chan], 
           bmTaumin[n_chan], bmTaumax[n_chan], ratiobm[n_chan], bmTaubm[n_chan], 
           TauWBMax[n_chan], TauWBMin[n_chan], tauwb[n_chan], wbgain[n_chan],
           delay_latency[n_chan], lasttmpgain[n_chan];
    int grdelay[n_chan], len_delay_latency[n_chan];

    /* Declare and allocate for tmpgain */
    double *tmpgain[n_chan];
    for (int i = 0; i < n_chan; i++) {
        tmpgain[i] = (double*) calloc(totalstim, sizeof(double));
    }

    /* Declare and allocate for meout */
    double *meout = calloc(totalstim, sizeof(double));

    /* Declare and initialize state variables for WbGammaTone filter */
    double wbphase[n_chan];
    COMPLEX *wbgtf[n_chan];
    COMPLEX *wbgtfl[n_chan];
    for (int i = 0; i < n_chan; i++) {
        wbgtf[i] = (COMPLEX*) calloc(4, sizeof(COMPLEX));
        wbgtfl[i] = (COMPLEX*) calloc(4, sizeof(COMPLEX));
        wbphase[i] = 0.0;
        for (int j = 0; j < 4; j++) {
            wbgtfl[i][j] = compmult(0, compexp(0));
            wbgtf[i][j]  = compmult(0, compexp(0));
        }
    }

    /* Declare and initialize state variables for OhcLowPass filter */
    double *ohc[n_chan], *ohcl[n_chan];
    for (int i = 0; i < n_chan; i++) {
        ohc[i] = (double*) calloc(4, sizeof(double));
        ohcl[i] = (double*) calloc(4, sizeof(double));
    }

    /* Declare and initialize state variables for C1ChirpFilt filter */
    double C1gain_norm[n_chan], C1initphase[n_chan];
    for (int i = 0; i < n_chan; i++) {
        C1gain_norm[i] = 0.0;
        C1initphase[i] = 0.0;
    }
    double*** C1input = (double***) malloc(n_chan * sizeof(double**));
    for (int i = 0; i < n_chan; i++) {
        C1input[i] = (double**) malloc(12 * sizeof(double*));
        for (int j = 0; j < 12; j++) {
            C1input[i][j] = (double*) malloc(4 * sizeof(double));
        }
    }
    double*** C1output = (double***) malloc(n_chan * sizeof(double**));
    for (int i = 0; i < n_chan; i++) {
        C1output[i] = (double**) malloc(12 * sizeof(double*));
        for (int j = 0; j < 12; j++) {
            C1output[i][j] = (double*) malloc(4 * sizeof(double));
        }
    }
    for (int i = 0; i < n_chan; i++) {
        for (int j = 0; j < 12; j++) {
            for (int k = 0; k < 4; k++) {
                C1input[i][j][k] = 0.0;
                C1output[i][j][k] = 0.0;
            }
        }
    }

    /* Declare and initialize state variables for C2ChirpFilt filter */
    double C2gain_norm[n_chan], C2initphase[n_chan];
    for (int i = 0; i < n_chan; i++) {
        C2gain_norm[i] = 0.0;
        C2initphase[i] = 0.0;
    }
    double*** C2input = (double***) malloc(n_chan * sizeof(double**));
    for (int i = 0; i < n_chan; i++) {
        C2input[i] = (double**) malloc(12 * sizeof(double*));
        for (int j = 0; j < 12; j++) {
            C2input[i][j] = (double*) malloc(4 * sizeof(double));
        }
    }
    double*** C2output = (double***) malloc(n_chan * sizeof(double**));
    for (int i = 0; i < n_chan; i++) {
        C2output[i] = (double**) malloc(12 * sizeof(double*));
        for (int j = 0; j < 12; j++) {
            C2output[i][j] = (double*) malloc(4 * sizeof(double));
        }
    }
    for (int i = 0; i < n_chan; i++) {
        for (int j = 0; j < 12; j++) {
            for (int k = 0; k < 4; k++) {
                C2input[i][j][k] = 0.0;
                C2output[i][j][k] = 0.0;
            }
        }
    }

    /* Declare and initialize variables for the IHC lowpass filter */
    double *ihc[n_chan], *ihcl[n_chan];
    for (int i = 0; i < n_chan; i++) {
        ihc[i] = (double*) calloc(8, sizeof(double));
        ihcl[i] = (double*) calloc(8, sizeof(double));
    }

    /* Declare variables used in the AN stage that are fixed across channels or reused */
    double alpha1 = 2.5e-6*100e3; 
    double beta1  = 5e-4; 
    double alpha2 = 1e-2*100e3; 
    double beta2  = 1e-1;

    /* Declare other variables used in the AN stage (all vary by channel/fiber type) */
    int n_fiber_type = 2;  // HSR==0, LSR==1
    double sponts[2] = {100.0, 0.1}; 
    double synstrength[n_fiber_type][n_chan], synslope[n_fiber_type][n_chan], 
           VI[n_fiber_type][n_chan], PG[n_fiber_type][n_chan], CG[n_fiber_type][n_chan],
           VL[n_fiber_type][n_chan], PL[n_fiber_type][n_chan], I1[n_fiber_type][n_chan], 
           I2[n_fiber_type][n_chan], 
           E1[n_fiber_type][n_chan][n_process], E2[n_fiber_type][n_chan][n_process], 
           D1[n_fiber_type][n_chan][n_process], D2[n_fiber_type][n_chan][n_process],
           CI[n_fiber_type][n_chan], CL[n_fiber_type][n_chan], 
           CIlast[n_fiber_type][n_chan];

    /* Loop over channels to calculate parameters, coefficients, etc. for each channel */
    for (int c = 0; c < n_chan; c++) {
        /* Calculate the CF for the control-path wideband filter by shifting 1.2 mm basal
           relative to nominal CF (pg 1450, Zilany and Bruce, 2006) */
        if (species == 1) {
            bmplace[c] = 11.9 * log10(0.80 + cf[c]/456.0); /* Calculate the location on basilar membrane from CF */
            centerfreq[c] = 456.0 * (pow(10, (bmplace[c]+1.2)/11.9) - 0.80); /* Shift the center freq */
        }
        else {
            bmplace[c] = (35/2.1) * log10(1.0 + cf[c]/165.4); /* Calculate the location on basilar membrane from CF */
            centerfreq[c] = 165.4 * (pow(10, (bmplace[c]+1.2)/(35/2.1)) - 1.0); /* Shift the center freq */
        }

        /* Determine time constants for cochlear filters that are time-invariant */
        Get_tauwb(c, cf, species, 3, Taumax, Taumin);
        TauWBMax[c] = Taumin[c] + 0.2*(Taumax[c]-Taumin[c]); // Eq 4, Zilany and Bruce (2006)
        TauWBMin[c] = TauWBMax[c]/Taumax[c]*Taumin[c];
        Get_taubm(c, cf, species, Taumax, bmTaumax, bmTaumin, ratiobm);
        bmTaubm[c] = cohc[c]*(bmTaumax[c]-bmTaumin[c]) + bmTaumin[c];
        tauwb[c] = TauWBMax[c] + (bmTaubm[c]-bmTaumax[c]) * 
            (TauWBMax[c]-TauWBMin[c])/(bmTaumax[c]-bmTaumin[c]);
        wbgain[c] = gain_groupdelay(tdres, centerfreq[c], cf[c], tauwb[c], &grdelay[c]);
        tmpgain[c][0] = wbgain[c];
        lasttmpgain[c] = wbgain[c];

        /* Calculate delay time introduced by cochlea/hair cells/synapse/etc
           (to account for measured spike latency in ANF recordings) */
        delay_latency[c] = delay_cat(cf[c]);  /* Delay time in s */
        len_delay_latency[c] = __max(0, (int) ceil(delay_latency[c]/tdres));  /* Delay time in samples*/

        /* Set parameters for double-exponential adaptation (based on fiber type / spont) */
        for (int t = 0; t < n_fiber_type; t++) {
            initialize_ws1988_adaptation(cf[c], sponts[t], &synstrength[t][c], 
                                         &synslope[t][c], &VI[t][c], &PG[t][c], &VL[t][c], 
                                         &PL[t][c], &CG[t][c], &CI[t][c], &CL[t][c]);
        }
    }

    /* Set parameters for power-law adaptation */
    for (int t = 0; t < n_fiber_type; t++) {
        for (int i = 0; i < n_chan; i++) {
            I1[t][i] = 0.0;
            I2[t][i] = 0.0;
        }
    }

    /* Calculate parameters for approximate power-law adaptation (sout1) */
    double delta = (log(tau_long_slow) - log(tau_short_slow)) / (n_process - 1);
    double tau_temp = 0.0;
    for (int t = 0; t < n_fiber_type; t++) {
        for (int i = 0; i < n_chan; i++) {
            for (int p = 0; p < n_process; p++) {
                // Initialize exponential process states at 0
                E1[t][i][p] = 0.0;

                // Determine time constant for this step, convert to decay coef and save
                tau_temp = exp(log(tau_short_slow) + delta * p);
                D1[t][i][p] = exp(-(1/(1/tdres)) / tau_temp);
            }
        }
    }

    /* Calculate parameters for approximate power-law adaptation (sout2) */
    delta = (log(tau_long_fast) - log(tau_short_fast)) / (n_process - 1);
    tau_temp = 0.0;
    for (int t = 0; t < n_fiber_type; t++) {
        for (int i = 0; i < n_chan; i++) {
            for (int p = 0; p < n_process; p++) {
                // Initialize exponential process states at 0
                E2[t][i][p] = 0.0;

                // Determine time constant for this step, convert to decay coef and save
                tau_temp = exp(log(tau_short_fast) + delta * p);
                D2[t][i][p] = exp(-(1/(1/tdres)) / tau_temp);
            }
        }
    }

    /* Declare variables used in the subcortical stage */
    double cn_B_e[2], cn_A_e[3], cn_B_i[2], cn_A_i[3];
    int cn_len_delay = (int) floor(cn_delay * 1/tdres);
    double *cn_i[n_chan];
    double *cn_e_tmp[n_chan];
    double *cn_i_tmp[n_chan];
    double cntmp = 0.0;
    for (int i = 0; i < n_chan; i++) {
        cn_i[i] = (double*) calloc(totalstim, sizeof(double));
        cn_e_tmp[i] = (double*) calloc(totalstim, sizeof(double));
        cn_i_tmp[i] = (double*) calloc(totalstim, sizeof(double));
    }

    double ic_B_e[2], ic_A_e[3], ic_B_i[2], ic_A_i[3];
    int ic_len_delay = (int) floor(ic_delay * 1/tdres);
    double *ic_i[n_chan];
    double *ic_e_tmp[n_chan];
    double *ic_i_tmp[n_chan];
    double ictmp = 0.0;
    for (int i = 0; i < n_chan; i++) {
        ic_i[i] = (double*) calloc(totalstim, sizeof(double));
        ic_e_tmp[i] = (double*) calloc(totalstim, sizeof(double));
        ic_i_tmp[i] = (double*) calloc(totalstim, sizeof(double));
    }
    double ic_tau_e_tmp = 0.0;
    double ic_tau_i_tmp = 0.0;
    double ic_delay_tmp = 0.0;

    /* Calculate CN params (constant across CFs) */
    get_alpha_norm(cn_tau_e, 1/tdres, 1.0, cn_B_e, cn_A_e);
    get_alpha_norm(cn_tau_i, 1/tdres, 1.0, cn_B_i, cn_A_i);

    /* Calculate variables for the MOC stage */
    double moc_d = exp(-TWOPI * (moc_cutoff/(1/tdres)));
    double gain_ic = 0.0;
    double gain_wdr = 0.0;
    int n_chan_incl = 0;
    double n_cf_per_oct = 0.0; 
    if (n_chan == 1) {
        n_cf_per_oct = 1.0;
    } else {
        n_cf_per_oct = n_chan/(log2(cf[n_chan-1]) - log2(cf[0]));
    }

    /* Create connectivity matrix for MOC WDR pathway
    At the end of each step of the time loop, we must determine which MOC signals from other
    channels are "allowed" to contribute to the gain factor calculated for the present 
    channel. We require that the distance between the current channel and a given channel,
    in terms of their CFs, be less than 1/2 of the paramter moc_width_wdr (in octaves). To
    avoid overhead associated with computation of log2 on each step, we precompute a 
    connectivity matrix that instantiates this rule and then use it in the loop below. This
    matrix, moc_connected below, encodes whether channel i should receive MOC WDR 
    contributions from channel j */
    int moc_connected[n_chan][n_chan];
    for (int i=0; i < n_chan; i++) {
        for (int j=0; j < n_chan; j++) {
            if ( fabs(log2(cf[i]) - log2(cf[j])) <= (moc_width_wdr/2) ) {
                moc_connected[i][j] = 1;
            } else {
                moc_connected[i][j] = 0;
            }
        }
    }

    /* Calculate middle-ear output */
    middle_ear(px, tdres, totalstim, species, meout);

    /* Compute the model, looping over time steps */
    for (int n=0; n<totalstim; n++) {
        /* Looping over channels/CFs */
        for (int c=0; c < n_chan; c++) {
            /* Delay input signal according to delay time for this channel */
            if ((n - len_delay_latency[c]) < 0) {
                me_curr = 0.0;
            } else {
                me_curr = meout[n - len_delay_latency[c]]; 
            }

            /* Pass signal through control-path filter */
            wbout1 = WbGammaTone(me_curr, tdres, centerfreq[c], n, tauwb[c], wbgain[c], 3, 
                                 &wbphase[c], wbgtf[c], wbgtfl[c]);
            wbout = pow((tauwb[c]/TauWBMax[c]), 3) * wbout1 * 10e3 *__max(1, cf[c]/5e3);

            /* Determine time constants for cochlear filters */
            Get_taubm(c, cf, species, Taumax, bmTaumax, bmTaumin, ratiobm);
            if (n == 0) {
                bmTaubm[c] = cohc[c]*(bmTaumax[c]-bmTaumin[c]) + bmTaumin[c];
            } else {
                bmTaubm[c] = (gain[c][n-1])*(bmTaumax[c]-bmTaumin[c]) + bmTaumin[c];
            }
            tauwb[c] = TauWBMax[c] + (bmTaubm[c]-bmTaumax[c]) * 
                (TauWBMax[c]-TauWBMin[c])/(bmTaumax[c]-bmTaumin[c]);
            wbgain[c] = gain_groupdelay(tdres, centerfreq[c], cf[c], tauwb[c], &grdelay[c]);

            /* Pass the control-path signal through the OHC model (nonlinear transduction and 
            lowpass filtering) */
            ohcnonlinout = Boltzman(wbout, 7.0, 12.0, 5.0, 5.0);
            ohcout = OhcLowPass(ohcnonlinout, tdres, 600, n, 1.0, 2, ohc[c], ohcl[c]);
            controlout[c][n] = NLafterohc(ohcout, bmTaumin[c], bmTaumax[c], 7.0);

            /* Determine time constant and shift of C1 filter poles based on output of OHCs */
            if (n == 0) {
                tauc1 = cohc[c]*(controlout[c][n]-bmTaumin[c]) + bmTaumin[c];
            } else {
                tauc1 = (gain[c][n-1])*(controlout[c][n]-bmTaumin[c]) + bmTaumin[c];
            }
            rsigma = 1/tauc1 - 1/bmTaumax[c];
            tauwb[c] = TauWBMax[c] + (tauc1-bmTaumax[c]) * 
                (TauWBMax[c]-TauWBMin[c])/(bmTaumax[c]-bmTaumin[c]);

            /* Do some sort of groupdelay manipulation??? TODO Clarify */
            wb_gain = gain_groupdelay(tdres, centerfreq[c], cf[c], tauwb[c], &grdelay[c]);
            if ((grdelay[c]+n) < totalstim) {
                tmpgain[c][grdelay[c]+n] = wb_gain;
            }
            if (tmpgain[c][n] == 0) {
                tmpgain[c][n] = lasttmpgain[c];
            }
            wbgain[c] = tmpgain[c][n];
            lasttmpgain[c] = wbgain[c];

            /* Apply signal-path C1 filter */
            c1out[c][n] = C1ChirpFilt(me_curr, tdres, cf[c], n, bmTaumax[c], rsigma, 
                                      &C1gain_norm[c], &C1initphase[c], C1input[c], 
                                      C1output[c]);

            /* Apply parallel-path C2 filter */
            c2out[c][n] = C2ChirpFilt(me_curr, tdres, cf[c], n, bmTaumax[c], 1/ratiobm[c], 
                                      &C2gain_norm[c], &C2initphase[c], C2input[c], 
                                      C2output[c]);

            /* Apply IHC model: NL input-output function and lowpass filtering */
            c1vihctmp  = NLogarithm(cihc[c]*c1out[c][n], 0.1, 3.0, cf[c]);
            c2vihctmp = -NLogarithm(c2out[c][n]*fabs(c2out[c][n])*cf[c]/10*cf[c]/2e3, 0.2, 
                                    1.0, cf[c]); /* C2 transduction output */
            ihcout[c][n] = IhcLowPass(c1vihctmp+c2vihctmp, tdres, 3000, n, 1.0, 7, ihc[c], 
                                      ihcl[c]);

            /* Loop over fiber types */
            for (int t = 0; t < n_fiber_type; t++) {
                /* Apply Westerman and Smith three-store-diffusion adaptation */
                apply_ws1988_adaptation(ihcout[c], n, synstrength[t][c], synslope[t][c], 
                                        VI[t][c], PG[t][c], VL[t][c], PL[t][c], CG[t][c], 
                                        tdres, &CI[t][c], &CL[t][c], &CIlast[t][c],
                                        expout[t][c]);

                /* Apply power-law adaptation */
                if (powerlaw_mode == 1) {
                    apply_powerlaw_adaptation(expout[t][c], randNums[t][c], &I1[t][c], 
                                            &I2[t][c], n, alpha1, beta1, alpha2, beta2, tdres,
                                            sout1[t][c], sout2[t][c]);
                } else if (powerlaw_mode == 2) {
                    apply_powerlaw_adaptation_iir(expout[t][c], randNums[t][c], &I1[t][c],
                        &I2[t][c], E1[t][c], E2[t][c], D1[t][c], D2[t][c], n_process, n,
                        coef_slow, coef_fast, tdres, sout1[t][c], sout2[t][c]);
                }
                synout[t][c][n] = sout1[t][c][n] + sout2[t][c][n];

                /* Compute instantaneous rate from synapse output */
                anrateout[t][c][n] = synout[t][c][n] / (1.0 + 0.75e-3 * synout[t][c][n]);
            }

            /* Delay rate output for SFIE cochlear nucleus inhibitory pathway */
            delay_signal(anrateout[0][c], n, cn_len_delay, cn_i[c]);

            /* Apply SFIE filters for cochlear nucleus stage */
            filter_alpha(anrateout[0][c], n, 1/tdres, cn_B_e, cn_A_e, cn_e_tmp[c]);
            filter_alpha(cn_i[c], n, 1/tdres, cn_B_i, cn_A_i, cn_i_tmp[c]);

            /* Compute output signal and rectify */
            cntmp = tdres * (cn_amp*cn_e_tmp[c][n] - (cn_amp*cn_inh)*cn_i_tmp[c][n]);
            cnout[c][n] = hw_rectify(cntmp);

            /* Update IC parameters based on CF */
            if (cf[c]/4.0 > 64.0) {
                /* Since this is a high CF (cf/4 > 64 Hz), we merely update back to ref values */
                ic_len_delay = (int) floor(ic_delay * 1/tdres);
                get_alpha_norm(ic_tau_e, 1/tdres, 1.0, ic_B_e, ic_A_e);
                get_alpha_norm(ic_tau_i, 1/tdres, 1.0, ic_B_i, ic_A_i);

           } else {
                /* Since this is a low CF channel (cf/4 < 64 Hz), we need to calc temp values of IC params */
                ic_tau_e_tmp = 1.0 / (10.0 * cf[c]/4.0);
                ic_tau_i_tmp = ic_tau_e_tmp * 1.5;
                ic_delay_tmp = ic_tau_e_tmp * 2.0;

                /* Update IC delay value and parameters values */
                /* Since this is a high CF (cf/4 > 64 Hz), we merely update back to ref values */
                ic_len_delay = (int) floor(ic_delay_tmp * 1/tdres);
                get_alpha_norm(ic_tau_e_tmp, 1/tdres, 1.0, ic_B_e, ic_A_e);
                get_alpha_norm(ic_tau_i_tmp, 1/tdres, 1.0, ic_B_i, ic_A_i);
           }

            /* Delay CN output for SFIE inferior colliculus inhibitory pathway */
            delay_signal(cnout[c], n, ic_len_delay, ic_i[c]);

            /* Apply SFIE filters for inferior colliculus stage */
            filter_alpha(cnout[c], n, 1/tdres, ic_B_e, ic_A_e, ic_e_tmp[c]);
            filter_alpha(ic_i[c], n, 1/tdres, ic_B_i, ic_A_i, ic_i_tmp[c]);

            /* Compute output signal and rectify */
            ictmp = tdres * (ic_amp*ic_e_tmp[c][n] - (ic_amp*ic_inh)*ic_i_tmp[c][n]);
            icout[c][n] = hw_rectify(ictmp);

            /* Lowpass filter LSR and IC responses to generate WDR-MOC and IC-MOC rates */
            filter_lowpass_iir(anrateout_lsr[c], n, moc_d, mocwdr[c]);
            filter_lowpass_iir(icout[c], n, moc_d, mocic[c]);

            /* Below, we generate gain value (in [0, 1]) for next sample */

            /* First, calculate IC-path contribution to gain-control signal */
            /* To do so, we just pass the output of MOC-IC (mocic) through MOC input-output
            nonlinearity */
            gain_ic = moc_nonlinearity(moc_weight_ic * mocic[c][n], moc_beta_ic, 
                                       moc_offset_ic, moc_maxrate_ic, moc_minrate_ic);

            /* Next, calculate WDR-path contribution to gain-control signal */
            /* To do so, we initialize gain_wdr at 1.0. Then, for those channels with CFs
            that are less than moc_width_wdr/2 octaves away from the current channel, we
            pass MOC-WDR (mocwdr) outputs through the MOC input-output nonlinearity and
            multiply the resulting gain values together with gain_wdr */
            gain_wdr = 1.0;
            for (int subc = 0; subc < n_chan; subc++) {
                if (moc_connected[c][subc]) {
                    gain_wdr = gain_wdr * 
                        moc_nonlinearity(moc_weight_wdr * mocwdr[subc][n], moc_beta_wdr, 
                                         moc_offset_wdr, moc_maxrate_wdr, moc_minrate_wdr);
                }
            }

            /* Normalize the WDR-driven gain factor */
            /* Normalization happens in one of two ways, depending on moc_width_wdr:
            1) If moc_width_wdr == 0.0, then the above loop reduces to 
                >>> gain_wdr = moc_nonlinearity(moc_weight_wdr * mocwdr[c][n]))
            i.e., each channel's gain factor is affected only by responses in that channel.
            Thus, we assume that no normalization should occur, so the gain_wdr value 
            determined by the above block is left unchanged.
            2) If moc_width_wdr > 0.0, then the gain_wdr value determined above reflects
            contributions from multiple channels. We seek to normalize the final gain factor
            with respect to the number of channels that contribute. Roughly, this value is
            n_cf_per_oct * moc_width_wdr. If we denote this value x, we normalize by:
                >>> gain_wdr = gain_wdr ^ (1/x)
            By raising to 1/x power, we essentially transform the computation of gain_wdr
            value into a geometric mean of the individual channels' gain factors. */
            if (moc_width_wdr > 0.0) {
                gain_wdr = pow(gain_wdr, 1/(n_cf_per_oct*moc_width_wdr));
            }

            /* Store final gain result, which is cohc * gain_wdr * gain_ic */
            gain[c][n] = cohc[c] * gain_wdr * gain_ic;
        }
    }

    /* Free dynamic memory */
    for (int i = 0; i < n_chan; i++) {
        free(tmpgain[i]);
        free(cn_i[i]);
        free(cn_e_tmp[i]);
        free(cn_i_tmp[i]);
        free(ic_i[i]);
        free(ic_e_tmp[i]);
        free(ic_i_tmp[i]);
    }
    free(meout);
    for (int i = 0; i < n_chan; i++) {
        free(wbgtf[i]);
        free(wbgtfl[i]);
        free(ohc[i]);
        free(ohcl[i]);
        free(ihc[i]);
        free(ihcl[i]);
    }
    for (int i = 0; i < n_chan; i++) {
        for (int j = 0; j < 12; j++) {
            free(C1input[i][j]);
            free(C1output[i][j]);
            free(C2input[i][j]);
            free(C2output[i][j]);
        }
        free(C1input[i]);
        free(C1output[i]);
        free(C2input[i]);
        free(C2output[i]);
    }
    free(C1input);
    free(C1output);
    free(C2input);
    free(C2output);
}

/**
 * hw_rectify
 * 
 * Applies half-wave rectification to a single number
 * 
 * @param x Value to rectify
 */
double hw_rectify(double x) {
    if (x < 0.0) x = 0.0;
    return x;
}

/**
 * delay_signal
 * 
 * Simple signal delay between two vectors for a constant number of samples
 * 
 * @param input Vector containing input signal
 * @param n Current sample index
 * @param len_delay Length of delay in samples
 * @param output Vector containing output signal
 * 
 */
void delay_signal(double *input, int n, int len_delay, double *output) {
    if ((n - len_delay) < 0) {
        output[n] = 0.0;
    } else {
        output[n] = input[n - len_delay]; 
    }
}

/**
 * filter_lowpass_iir
 * 
 * Applies a single-pole IIR lowpass filter to a vector of data sample by sample
 * 
 * Implements filter defined by recurrence relation:
 *   y[n] = (1-d)*x[n] + d*y[n-1]
 * 
 * Note that decay value (d) is related to time constant (tau) by:
 *   d = exp(-1/tau)
 * 
 * Note also that d is related to the (normalized) cutoff frequency (f) by
 *   f = -ln(d)/(2*pi)
 * 
 * @param x Input vector
 * @param n Current sample index
 * @param d Decay value
 * @param y Output vector
 */
void filter_lowpass_iir(double *x, int n, double d, double *y) {
    if (n == 0) {
        y[n] = (1-d)*x[n];
    } else {
        y[n] = (1-d)*x[n] + d*y[n-1];
    }
}

/**
 * moc_nonlinearity
 * 
 * Simple nonlineariy relating lowpass-filtered MOC input to MOC output
 * 
 * Nonlinearity that is applied to lowpass-filtered MOC input to genereate MOC output. 
 * Outputs are in the range (minrate, maxrate). Monotonically decreases from zero with 
 * optional offset. If maxrate=1.0 and minrate~=0.0, then the MOC output can be interpreted
 * as a cohc value in the model code (is on the same scale).
 * 
 * @param x Input value
 * @param beta Parameter of nonlinearity, lower values correspond to more gradual slopes
 * @param offset Values below offset will return maxrate, values above offset will apply
 *   nonlinearity
 * @param maxrate Maximum output 
 * @param minrate Minimum output 
 */
double moc_nonlinearity(double x, double beta, double offset, double maxrate, 
                        double minrate) {
    if (x < offset) {
        return maxrate;
    } else {
        return ((maxrate-minrate) * 1.0/(1.0 + pow(beta*(x-offset), 2.0))) + minrate;
    }
}

/**
 * middle_ear
 * 
 * Filters a sound-pressure waveform with a cat- or human-type middle-ear filter to result 
 * in an output stapes motion waveform that can drive the following stage of the model.
 * 
 * @param px Sound-pressure waveform in Pa
 * @param tdres time resolution (s), or reciprocal of the sampling rate (1/Hz)
 * @param totalstim number of samples in the simulation
 * @param species what species to simulate (1==cat, 2=human[shera], 3==human[glasberg])
 * @param meout Vector in which to store output of middle-ear filter, length should match totalstim
 */
void middle_ear(double *px, double tdres, int totalstim, int species, double *meout)
{
    /* Variables for middle-ear model */
	double megainmax;
    double *mey1, *mey2, *mey3;
    double fp,C,m11,m12,m13,m14,m15,m16,m21,m22,m23,m24,m25,m26,m31,m32,m33,m34,m35,m36;
    int n;

    /* Allocate memory for the temporary variables in the middle-ear model */
	mey1 = (double*)calloc(totalstim,sizeof(double));
	mey2 = (double*)calloc(totalstim,sizeof(double));
	mey3 = (double*)calloc(totalstim,sizeof(double));

    /* Prewarping and related constants for the middle ear */
    fp = 1e3;  /* prewarping frequency 1 kHz */
    C  = TWOPI*fp/tan(TWOPI/2*fp*tdres);

    /* Configure middle-ear filter coefficient for cat */
    /* Simplified version from Bruce et al. (JASA 2003) */
    if (species == 1) {
        m11 = C/(C + 693.48);                    m12 = (693.48 - C)/C;            m13 = 0.0;
        m14 = 1.0;                               m15 = -1.0;                      m16 = 0.0;
        m21 = 1/(pow(C,2) + 11053*C + 1.163e8);  m22 = -2*pow(C,2) + 2.326e8;     m23 = pow(C,2) - 11053*C + 1.163e8; 
        m24 = pow(C,2) + 1356.3*C + 7.4417e8;    m25 = -2*pow(C,2) + 14.8834e8;   m26 = pow(C,2) - 1356.3*C + 7.4417e8;
        m31 = 1/(pow(C,2) + 4620*C + 909059944); m32 = -2*pow(C,2) + 2*909059944; m33 = pow(C,2) - 4620*C + 909059944;
        m34 = 5.7585e5*C + 7.1665e7;             m35 = 14.333e7;                  m36 = 7.1665e7 - 5.7585e5*C;
        megainmax=41.1405;
    }
    /* Configure middle-ear filter coefficient for human */
    /* Based on Pascal et al. (JASA 1998)  */
    else {
        m11=1/(pow(C,2)+5.9761e+003*C+2.5255e+007); m12=(-2*pow(C,2)+2*2.5255e+007);
        m13=(pow(C,2)-5.9761e+003*C+2.5255e+007);   m14=(pow(C,2)+5.6665e+003*C);             
        m15=-2*pow(C,2);					        m16=(pow(C,2)-5.6665e+003*C);
        m21=1/(pow(C,2)+6.4255e+003*C+1.3975e+008); m22=(-2*pow(C,2)+2*1.3975e+008);
        m23=(pow(C,2)-6.4255e+003*C+1.3975e+008);   m24=(pow(C,2)+5.8934e+003*C+1.7926e+008); 
        m25=(-2*pow(C,2)+2*1.7926e+008);	        m26=(pow(C,2)-5.8934e+003*C+1.7926e+008);
        m31=1/(pow(C,2)+2.4891e+004*C+1.2700e+009); m32=(-2*pow(C,2)+2*1.2700e+009);
        m33=(pow(C,2)-2.4891e+004*C+1.2700e+009);   m34=(3.1137e+003*C+6.9768e+008);     
        m35=2*6.9768e+008;				            m36=(-3.1137e+003*C+6.9768e+008);
        megainmax=2;
    };

    /* Implement middle-ear filter */
 	for (n=0; n < totalstim; n++) {
        if (n==0) {
            mey1[0]  = m11*px[0];
            if (species>1) mey1[0] = m11*m14*px[0];
            mey2[0]  = mey1[0]*m24*m21;
            mey3[0]  = mey2[0]*m34*m31;
            meout[0] = mey3[0]/megainmax ;
        }
        else if (n==1) {
            mey1[1] = m11*(-m12*mey1[0] + px[1] - px[0]);
            if (species>1) mey1[1] = m11*(-m12*mey1[0]+m14*px[1]+m15*px[0]);
            mey2[1] = m21*(-m22*mey2[0] + m24*mey1[1] + m25*mey1[0]);
            mey3[1] = m31*(-m32*mey3[0] + m34*mey2[1] + m35*mey2[0]);
            meout[1] = mey3[1]/megainmax;
        }
        else {
            mey1[n] = m11*(-m12*mey1[n-1] + px[n] - px[n-1]);
            if (species>1) mey1[n]= m11*(-m12*mey1[n-1]-m13*mey1[n-2]+m14*px[n]+m15*px[n-1]+m16*px[n-2]);
            mey2[n] = m21*(-m22*mey2[n-1] - m23*mey2[n-2] + m24*mey1[n] + m25*mey1[n-1] + m26*mey1[n-2]);
            mey3[n] = m31*(-m32*mey3[n-1] - m33*mey3[n-2] + m34*mey2[n] + m35*mey2[n-1] + m36*mey2[n-2]);
            meout[n] = mey3[n]/megainmax;
        };
    }

    /* Freeing dynamic memory allocated earlier */
    free(mey1); free(mey2); free(mey3);
}

/**
 * Get_tauwb
 * 
 * Calculate tau (time constant) values for the control-path wideband filter
 * 
 * @param idx_chan Index selecting which channel we are processing
 * @param cf Vector of characteristic frequencies (Hz)
 * @param species What species to simulate (1==cat, 2=human[shera], 3==human[glasberg])
 * @param order Order of the wideband Gammatone filter
 * @param taumax Filter time constant at low sound levels
 * @param taumin Filter time constant at high sound levels
 * 
 * TODO Revise implementation to match others (no indexing)
 */
void Get_tauwb(int idx_chan, double *cf, int species, int order, double *taumax, 
               double *taumin) {
    double Q10, bw, gain, ratio;

    /* Calculate gain */
    gain = 52.0/2.0 * (tanh(2.2*log10(cf[idx_chan]/0.6e3)+0.15) + 1.0);  // Eq 6, Zilany and Bruce (2007)

    /* Limit gain to 15-60 dB range */
    if (gain > 60.0) gain = 60.0;  
    if (gain < 15.0) gain = 15.0;

    /* Calculate ratio of TauMin/TauMax according to the gain and order */
    ratio = pow(10, (-gain/(20.0*order)));  // pg 1451, top left, Zilany and Bruce (2006)

    /* Calculate Q10 values for cats (species == 1) or humans (Shera == 2, Glasberg == 3) */
    /* Values for cat come from ... */
    /* Values for Shera come from Shera et al. (PNAS 2002) */
    /* Values for Glasberg come from Glasberg and Moore (Hear. Res. 1999) */
    if (species == 1) {
        Q10 = pow(10, 0.4708*log10(cf[idx_chan]/1e3) + 0.4664);  // pg 1451, top left, Zilany and Bruce (2006)
    }
    if (species==2) {
        Q10 = pow((cf[idx_chan]/1000), 0.3) * 12.7*0.505 + 0.2085;
    }
    else {
        Q10 = cf[idx_chan] / 24.7 / (4.37*(cf[idx_chan]/1000)+1)*0.505 + 0.2085;
    }

    /* Calculate bandwidth, taumax, and taumin */
    bw = cf[idx_chan]/Q10;
    taumax[idx_chan] = 2.0/(TWOPI*bw);  // pg 1451, top left, Zilany and Bruce (2006)
    taumin[idx_chan] = taumax[idx_chan]*ratio; // Eq 5, Zilany and Bruce (2006)
}

/**
 * Get_taubm
 * 
 * Calculate tau (time constant) values for the signal-path narrowband filter
 * 
 * @param idx_chan Index selecting which channel we are processing
 * @param cf Vector of characteristic frequencies (Hz)
 * @param species What species to simulate (1==cat, 2=human[shera], 3==human[glasberg])
 * @param taumax ??
 * @param bmTaumax ???
 * @param bmTaumin ???
 * @param ratio ???
 */
void Get_taubm(int idx_chan, double *cf, int species, double *taumax, double *bmTaumax,
               double *bmTaumin, double *ratio) {
    double gain,factor,bwfactor;

    /* Calculate gain */
    gain = 52.0/2.0 * (tanh(2.2*log10(cf[idx_chan]/0.6e3)+0.15) + 1.0);  // Eq 6, Zilany and Bruce (2007)

    /* Limit gain to 15-60 dB range */
    if (gain > 60.0) gain = 60.0;  
    if (gain < 15.0) gain = 15.0;

    /* Calculate bmTaumax, bmTaumin, ratio */
    bwfactor = 0.7;
    factor   = 2.5;
    ratio[idx_chan]  = pow(10, (-gain/(20.0*factor))); 
    bmTaumax[idx_chan] = taumax[idx_chan]/bwfactor;
    bmTaumin[idx_chan] = bmTaumax[idx_chan]*ratio[idx_chan];     
}

/**
 * gain_groupdelay
 * 
 * ???
 * 
 * @param tdres time resolution (s), or reciprocal of the sampling rate (1/Hz)
 * @param centerfreq ???
 * @param cf Characteristic frequency (Hz)
 * @param tau ???
 * @param grdelay ???
 */
double gain_groupdelay(double tdres, double centerfreq, double cf, double tau, int *grdelay) { 
    double tmpcos,dtmp2,c1LP,c2LP,tmp1,tmp2,wb_gain;

    /* Calculate constants and parameters */
    tmpcos = cos(TWOPI*(centerfreq-cf)*tdres);
    dtmp2 = tau*2.0/tdres;
    c1LP = (dtmp2-1)/(dtmp2+1);
    c2LP = 1.0/(dtmp2+1);
    tmp1 = 1+c1LP*c1LP-2*c1LP*tmpcos;
    tmp2 = 2*c2LP*c2LP*(1+tmpcos);

    /* Calculate ??? */
    wb_gain = pow(tmp1/tmp2, 1.0/2.0);

    /* Calculate ??? */
    (*grdelay) = (int)floor((0.5-(c1LP*c1LP-c1LP*tmpcos)/(1+c1LP*c1LP-2*c1LP*tmpcos)));

    return(wb_gain);
}

/**
 * delay_cat
 * 
 * For a given CF, return the delay associated with peripheral transduction in cat
 * 
 * Based on ???
 * 
 * @param cf Characteristic frequency (Hz)
 */
double delay_cat(double cf) {  
    double A0,A1,x,delay;

    /* Calculate constants and parameters */
    A0 = 3.0;  
    A1 = 12.5;
    x = 11.9 * log10(0.80 + cf / 456.0);

    /* Calculate delay time */
    delay = A0 * exp( -x/A1 ) * 1e-3;
  
    return(delay);
}

/**
 * delay_human
 * 
 * For a given CF, return the delay associated with peripheral transduction in human
 * 
 * Based on Harte et al. (JASA 2009)
 * 
 * @param cf Characteristic frequency (Hz)
 */
double delay_human(double cf) {  
    double A,B,delay;

    /* Calculate constants and parameters */
    A = -0.37;  
    B = 11.09/2;

    /* Calculate delay time */
    delay = B * pow(cf * 1e-3,A)*1e-3;
  
    return(delay);
}

/**
 * Boltzmann
 * 
 * Nonlinearity relating control-path filter input to output prior to lowpass filter
 * 
 * See Eq 7, Zilany and Bruce (2006) for more details about this function
 * 
 * @param x Input value
 * @param asym Parameter controlling ratio of (positive) max output to negative (min) output
 * @param s0 Parameter of Boltzmann function
 * @param s1 Parameter of Boltzmann function
 * @param x1 Parameter of Boltzmann function
 */
double Boltzman(double x, double asym, double s0, double s1, double x1){
    double shift,x0,out1,out;

    /* Calculate shift and x0 values */
    shift = 1.0/(1.0+asym);
    x0    = s0*log((1.0/shift-1)/(1+exp(x1/s1)));

    /* Calculate output values */  
    out1 = 1.0/(1.0+exp(-(x-x0)/s0)*(1.0+exp(-(x-x1)/s1)))-shift;
	out = out1/(1-shift);

    return(out);
}
  
/**
 * OhcLowPass
 * 
 * Outer-hair-cell lowpass filter
 * 
 * @param x input value (???)
 * @param tdres time resolution (s), or reciprocal of the sampling rate (1/Hz)
 * @param Fc cutoff frequency (Hz)
 * @param n Current sample of processing (used to initialize static memory when n == 0)
 * @param gain Scalar gain applied to input
 * @param order Filter order 
 */
double OhcLowPass(double x, double tdres, double Fc, int n, double gain, int order,
                  double *ohc, double *ohcl) {
    // static double ohc[4],ohcl[4];
    double c,c1LP,c2LP;
    int i,j;

    /* If we're on the first sample, initialize static memory to zeros */
    // if (n == 0) {
    //     for (i=0; i<(order+1); i++) {
    //         ohc[i] = 0;
    //         ohcl[i] = 0;
    //     }
    // }    

    /* Calculate filter coefficients */ 
    c = 2.0/tdres;
    c1LP = ( c - TWOPI*Fc ) / ( c + TWOPI*Fc );
    c2LP = TWOPI*Fc / (TWOPI*Fc + c);

    /* Implement filter */
    ohc[0] = x*gain;
    for (i=0; i<order; i++) {
        ohc[i+1] = c1LP*ohcl[i+1] + c2LP*(ohc[i]+ohcl[i]);
    }
    for (j=0; j<=order; j++) {
        ohcl[j] = ohc[j];
    } 
    
    return(ohc[order]);
}

/**
 * IhcLowPass
 * 
 * Inner-hair-cell lowpass filter
 * 
 * @param x input value (???)
 * @param tdres time resolution (s), or reciprocal of the sampling rate (1/Hz)
 * @param Fc cutoff frequency (Hz)
 * @param n Current sample of processing (used to initialize static memory when n == 0)
 * @param gain Scalar gain applied to input
 * @param order Filter order 
 */
double IhcLowPass(double x, double tdres, double Fc, int n, double gain, int order,
                  double *ihc, double *ihcl) {
    double C,c1LP,c2LP;
    int i,j;

    /* If we're on the first sample, initialize static memory to zeros */
    if (n==0) {
        for(i=0; i<(order+1); i++) {
            ihc[i] = 0;
            ihcl[i] = 0;
        }
    }     

    /* Calculate filter coefficients */ 
    C = 2.0/tdres;
    c1LP = ( C - TWOPI*Fc ) / ( C + TWOPI*Fc );
    c2LP = TWOPI*Fc / (TWOPI*Fc + C);

    /* Implement the filter */
    ihc[0] = x*gain;
    for (i=0; i<order;i++) {
        ihc[i+1] = c1LP*ihcl[i+1] + c2LP*(ihc[i]+ihcl[i]);
    }
    for (j=0; j<=order;j++) { 
        ihcl[j] = ihc[j];
    }

    return(ihc[order]);
}

/**
 * NLafterohc
 * 
 * Nonlinearity to transform output of OHCLowPass to time-varying C1 time constant
 * 
 * See pg 1451, top right, Zilany and Bruce (2006) for more details.
 * 
 * @param x input value (output of OHC lowpass filter)
 * @param taumin ???
 * @param taumax ???
 * @param asym ???
 */
double NLafterohc(double x, double taumin, double taumax, double asym) {    
	double R,dc,R1,s0,x1,out,minR;

    /* Calculate constants and parameters */
	minR = 0.05;  // ratio of asymptotic lower bound of time constant to taumax 
    R  = taumin/taumax;
	if (R < minR) minR = 0.5*R;
    else minR = minR;
    
    dc = (asym-1)/(asym+1.0)/2.0-minR;  // estimate of DC component of control-path output at high levels
    R1 = R-minR;

    s0 = -dc/log(R1/(1-minR));
    x1  = fabs(x);

    /* Calculate output, limiting by taumin and taumax */
    out = taumax*(minR+(1.0-minR)*exp(-x1/s0));
	if (out<taumin) out = taumin; 
    if (out>taumax) out = taumax;

    return(out);
}

/**
 * NLogarithm
 * 
 * Inner-hair-cell nonlinearity 
 * 
 * @param x input value (???)
 * @param slope ???
 * @param asym ???
 * @param cf Characteristic frequency (Hz)
 */
double NLogarithm(double x, double slope, double asym, double cf) {
    double corner,strength,xx,splx,asym_t;

    /* Calculate constants and parameters */
    corner    = 80; 
    strength  = 20.0e6/pow(10,corner/20);

    /* Calculate output */
    xx = log(1.0+strength*fabs(x))*slope;
    if (x<0) {
		splx = 20*log10(-x/20e-6);
		asym_t = asym -(asym-1)/(1+exp(splx/5.0));
		xx = -1/asym_t*xx;
	};  

    return(xx);
}

double WbGammaTone(double x, double tdres, double centerfreq, int n, double tau, 
                   double gain, int order, double *wbphase, COMPLEX *wbgtf, 
                   COMPLEX *wbgtfl) {
    double delta_phase,dtmp,c1LP,c2LP,out;
    int i,j;

  delta_phase = -TWOPI*centerfreq*tdres;
  (*wbphase) += delta_phase;
  
  dtmp = tau*2.0/tdres;
  c1LP = (dtmp-1)/(dtmp+1);
  c2LP = 1.0/(dtmp+1);
  wbgtf[0] = compmult(x,compexp((*wbphase)));                 /* FREQUENCY SHIFT */
  
  for(j = 1; j <= order; j++)                              /* IIR Bilinear transformation LPF */
  wbgtf[j] = comp2sum(compmult(c2LP*gain, comp2sum(wbgtf[j-1],wbgtfl[j-1])),
      compmult(c1LP,wbgtfl[j]));
  out = REAL(compprod(compexp(-(*wbphase)), wbgtf[order])); /* FREQ SHIFT BACK UP */
  
  for(i=0; i<=order;i++) wbgtfl[i] = wbgtf[i];
  return(out);
}

double C1ChirpFilt(double x, double tdres,double cf, int n, double taumax, double rsigma,
                   double *C1gain_norm, double *C1initphase, double **C1input, 
                   double **C1output)
{
    // static double C1input[12][4], C1output[12][4];

    double ipw, ipb, rpa, pzero, rzero;
	double sigma0,fs_bilinear,CF,norm_gain,phase,c1filterout;
	int i,r,order_of_pole,half_order_pole,order_of_zero;
	double temp, dy, preal, pimg;

	COMPLEX p[11]; 
	
	/* Defining initial locations of the poles and zeros */
	/*======== setup the locations of poles and zeros =======*/
	  sigma0 = 1/taumax;
	  ipw    = 1.01*cf*TWOPI-50;
	  ipb    = 0.2343*TWOPI*cf-1104;
	  rpa    = pow(10, log10(cf)*0.9 + 0.55)+ 2000;
	  pzero  = pow(10,log10(cf)*0.7+1.6)+500;

	/*===============================================================*/     
         
     order_of_pole    = 10;             
     half_order_pole  = order_of_pole/2;
     order_of_zero    = half_order_pole;

	 fs_bilinear = TWOPI*cf/tan(TWOPI*cf*tdres/2);
     rzero       = -pzero;
	 CF          = TWOPI*cf;
   
   if (n==0)
   {		  
	p[1].x = -sigma0;     

    p[1].y = ipw;

	p[5].x = p[1].x - rpa; p[5].y = p[1].y - ipb;

    p[3].x = (p[1].x + p[5].x) * 0.5; p[3].y = (p[1].y + p[5].y) * 0.5;

    p[2]   = compconj(p[1]);    p[4] = compconj(p[3]); p[6] = compconj(p[5]);

    p[7]   = p[1]; p[8] = p[2]; p[9] = p[5]; p[10]= p[6];

	   (*C1initphase) = 0.0;
       for (i=1;i<=half_order_pole;i++)          
	   {
           preal     = p[i*2-1].x;
		   pimg      = p[i*2-1].y;
	       (*C1initphase) = (*C1initphase) + atan(CF/(-rzero))-atan((CF-pimg)/(-preal))-atan((CF+pimg)/(-preal));
	   };

	/*===================== Initialize C1input & C1output =====================*/

      for (i=1;i<=(half_order_pole+1);i++)          
      {
		   C1input[i][3] = 0; 
		   C1input[i][2] = 0; 
		   C1input[i][1] = 0;
		   C1output[i][3] = 0; 
		   C1output[i][2] = 0; 
		   C1output[i][1] = 0;
      }

	/*===================== normalize the gain =====================*/
    
      (*C1gain_norm) = 1.0;
      for (r=1; r<=order_of_pole; r++)
		   (*C1gain_norm) = (*C1gain_norm)*(pow((CF - p[r].y),2) + p[r].x*p[r].x);
      
   };
     
    norm_gain= sqrt((*C1gain_norm))/pow(sqrt(CF*CF+rzero*rzero),order_of_zero);
	
	p[1].x = -sigma0 - rsigma;

	p[1].y = ipw;

	p[5].x = p[1].x - rpa; p[5].y = p[1].y - ipb;

    p[3].x = (p[1].x + p[5].x) * 0.5; p[3].y = (p[1].y + p[5].y) * 0.5;

    p[2] = compconj(p[1]); p[4] = compconj(p[3]); p[6] = compconj(p[5]);

    p[7] = p[1]; p[8] = p[2]; p[9] = p[5]; p[10]= p[6];

    phase = 0.0;
    for (i=1;i<=half_order_pole;i++)          
    {
           preal = p[i*2-1].x;
		   pimg  = p[i*2-1].y;
	       phase = phase-atan((CF-pimg)/(-preal))-atan((CF+pimg)/(-preal));
	};

	rzero = -CF/tan(((*C1initphase)-phase)/order_of_zero);

   /*%==================================================  */
	/*each loop below is for a pair of poles and one zero */
   /*%      time loop begins here                         */
   /*%==================================================  */
 
       C1input[1][3]=C1input[1][2]; 
	   C1input[1][2]=C1input[1][1]; 
	   C1input[1][1]= x;

       for (i=1;i<=half_order_pole;i++)          
       {
           preal = p[i*2-1].x;
		   pimg  = p[i*2-1].y;
		  	   
           temp  = pow((fs_bilinear-preal),2)+ pow(pimg,2);
		   

           /*dy = (input[i][1] + (1-(fs_bilinear+rzero)/(fs_bilinear-rzero))*input[i][2]
                                 - (fs_bilinear+rzero)/(fs_bilinear-rzero)*input[i][3] );
           dy = dy+2*output[i][1]*(fs_bilinear*fs_bilinear-preal*preal-pimg*pimg);

           dy = dy-output[i][2]*((fs_bilinear+preal)*(fs_bilinear+preal)+pimg*pimg);*/
		   
	       dy = C1input[i][1]*(fs_bilinear-rzero) - 2*rzero*C1input[i][2] - (fs_bilinear+rzero)*C1input[i][3]
                 +2*C1output[i][1]*(fs_bilinear*fs_bilinear-preal*preal-pimg*pimg)
			     -C1output[i][2]*((fs_bilinear+preal)*(fs_bilinear+preal)+pimg*pimg);

		   dy = dy/temp;

		   C1input[i+1][3] = C1output[i][2]; 
		   C1input[i+1][2] = C1output[i][1]; 
		   C1input[i+1][1] = dy;

		   C1output[i][2] = C1output[i][1]; 
		   C1output[i][1] = dy;
       }

	   dy = C1output[half_order_pole][1]*norm_gain;  /* don't forget the gain term */
	   c1filterout= dy/4.0;   /* signal path output is divided by 4 to give correct C1 filter gain */
	                   
     return (c1filterout);
}  

/* -------------------------------------------------------------------------------------------- */
/** Parallelpath C2 filter: same as the signal-path C1 filter with the OHC completely impaired */

double C2ChirpFilt(double xx, double tdres,double cf, int n, double taumax, double fcohc,
                   double *C2gain_norm, double *C2initphase, double **C2input, 
                   double **C2output)
{
	double ipw, ipb, rpa, pzero, rzero;

	double sigma0,fs_bilinear,CF,norm_gain,phase,c2filterout;
	int    i,r,order_of_pole,half_order_pole,order_of_zero;
	double temp, dy, preal, pimg;

	COMPLEX p[11]; 	
    
    /*================ setup the locations of poles and zeros =======*/

	  sigma0 = 1/taumax;
	  ipw    = 1.01*cf*TWOPI-50;
      ipb    = 0.2343*TWOPI*cf-1104;
	  rpa    = pow(10, log10(cf)*0.9 + 0.55)+ 2000;
	  pzero  = pow(10,log10(cf)*0.7+1.6)+500;
	/*===============================================================*/     
         
     order_of_pole    = 10;             
     half_order_pole  = order_of_pole/2;
     order_of_zero    = half_order_pole;

	 fs_bilinear = TWOPI*cf/tan(TWOPI*cf*tdres/2);
     rzero       = -pzero;
	 CF          = TWOPI*cf;
   	    
    if (n==0)
    {		  
	p[1].x = -sigma0;     

    p[1].y = ipw;

	p[5].x = p[1].x - rpa; p[5].y = p[1].y - ipb;

    p[3].x = (p[1].x + p[5].x) * 0.5; p[3].y = (p[1].y + p[5].y) * 0.5;

    p[2] = compconj(p[1]); p[4] = compconj(p[3]); p[6] = compconj(p[5]);

    p[7] = p[1]; p[8] = p[2]; p[9] = p[5]; p[10]= p[6];

	   (*C2initphase) = 0.0;
       for (i=1;i<=half_order_pole;i++)         
	   {
           preal     = p[i*2-1].x;
		   pimg      = p[i*2-1].y;
	       (*C2initphase) = (*C2initphase) + atan(CF/(-rzero))-atan((CF-pimg)/(-preal))-atan((CF+pimg)/(-preal));
	   };

	/*===================== Initialize C2input & C2output =====================*/

      for (i=1;i<=(half_order_pole+1);i++)          
      {
		   C2input[i][3] = 0; 
		   C2input[i][2] = 0; 
		   C2input[i][1] = 0;
		   C2output[i][3] = 0; 
		   C2output[i][2] = 0; 
		   C2output[i][1] = 0;
      }
    
    /*===================== normalize the gain =====================*/
    
     (*C2gain_norm) = 1.0;
     for (r=1; r<=order_of_pole; r++)
		   (*C2gain_norm) = (*C2gain_norm)*(pow((CF - p[r].y),2) + p[r].x*p[r].x);
    };
     
    norm_gain= sqrt((*C2gain_norm))/pow(sqrt(CF*CF+rzero*rzero),order_of_zero);
    
	p[1].x = -sigma0*fcohc;

	p[1].y = ipw;

	p[5].x = p[1].x - rpa; p[5].y = p[1].y - ipb;

    p[3].x = (p[1].x + p[5].x) * 0.5; p[3].y = (p[1].y + p[5].y) * 0.5;

    p[2] = compconj(p[1]); p[4] = compconj(p[3]); p[6] = compconj(p[5]);

    p[7] = p[1]; p[8] = p[2]; p[9] = p[5]; p[10]= p[6];

    phase = 0.0;
    for (i=1;i<=half_order_pole;i++)          
    {
           preal = p[i*2-1].x;
		   pimg  = p[i*2-1].y;
	       phase = phase-atan((CF-pimg)/(-preal))-atan((CF+pimg)/(-preal));
	};

	rzero = -CF/tan(((*C2initphase)-phase)/order_of_zero);	
   /*%==================================================  */
   /*%      time loop begins here                         */
   /*%==================================================  */

       C2input[1][3]=C2input[1][2]; 
	   C2input[1][2]=C2input[1][1]; 
	   C2input[1][1]= xx;

      for (i=1;i<=half_order_pole;i++)          
      {
           preal = p[i*2-1].x;
		   pimg  = p[i*2-1].y;
		  	   
           temp  = pow((fs_bilinear-preal),2)+ pow(pimg,2);
		   
           /*dy = (input[i][1] + (1-(fs_bilinear+rzero)/(fs_bilinear-rzero))*input[i][2]
                                 - (fs_bilinear+rzero)/(fs_bilinear-rzero)*input[i][3] );
           dy = dy+2*output[i][1]*(fs_bilinear*fs_bilinear-preal*preal-pimg*pimg);

           dy = dy-output[i][2]*((fs_bilinear+preal)*(fs_bilinear+preal)+pimg*pimg);*/
		   
	      dy = C2input[i][1]*(fs_bilinear-rzero) - 2*rzero*C2input[i][2] - (fs_bilinear+rzero)*C2input[i][3]
                 +2*C2output[i][1]*(fs_bilinear*fs_bilinear-preal*preal-pimg*pimg)
			     -C2output[i][2]*((fs_bilinear+preal)*(fs_bilinear+preal)+pimg*pimg);

		   dy = dy/temp;

		   C2input[i+1][3] = C2output[i][2]; 
		   C2input[i+1][2] = C2output[i][1]; 
		   C2input[i+1][1] = dy;

		   C2output[i][2] = C2output[i][1]; 
		   C2output[i][1] = dy;

       };

	  dy = C2output[half_order_pole][1]*norm_gain;
	  c2filterout= dy/4.0;
	  
	  return (c2filterout); 
}   
