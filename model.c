 /*
  This is v6.00 of the code for subcortical auditory model model of:

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

  * Farhadi, A., Jennings, S. G., Strickland, E. A., and Carney, L. H. (2023). “Subcortical 
  auditory model including efferent dynamic gain control with inputs from cochlear nucleus 
  and inferior colliculus,” The Journal of the Acoustical Society of America, 154, 
  3644–3659. doi:10.1121/10.0022578

  * Guest, D. R., and Carney, L. H. (2024). “A fast and accurate approximation of power-law 
  adaptation for auditory computational models,” The Journal of the Acoustical Society of 
  America, 156, 3954–3957. doi:10.1121/10.0034457

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
 * ///// START EXPERIMENTAL NEW MOC PARAMETERS /////
 * @param [in] moc_beta Parameter in the WDR-MOC nonlinearity, governs the slope
 * of the nonlinearity such that smaller values yield more gradual slopes between MOC rate
 * and MOC output gain while larger values yield sharper slopes between MOC rate and MOC
 * output gain, default value is 0.015 (per ARO2024 / HWK2024)
 * @param [in] moc_offset (sp/s) Parameter in the WDR-MOC nonlinearity, shifts the
 * nonlinearity along the x-axis such that gain control is not applied until the *weighted*
 * WDR-MOC rate exceeds the offset, default value is 0.0
 * @param [in] moc_minval Minimum output value for the WDR-MOC nonlinearity, should be
 * in range of [0, 1], default is 0.0
 * @param [in] moc_maxval Maximum output value for the WDR-MOC nonlinearity, should be
 * in range of [0, 1], default is 1.0
 * @param [in] moc_weight (n_chan, 1) Weight applied to the WDR-MOC rate before it is passed
 * through the WDR-MOC nonlinearity and converted into a gain value. Setting this value to
 * 0.0 disables WDR-driven gain control. This is a vector-valued parameter where each
 * element is applied to weight responses from the corresponding CF.
 * @param [in] moc_width (oct) Range of CFs over which MOC gain control signal from
 * a single channel is sent to OHCs. This is parametrized in terms of the standard deviation
 * of a Gaussian weighting function centered at CF.
 *  * ///// STOP EXPERIMENTAL NEW MOC PARAMETERS /////
 * @param [in] dur_settle (s) Duration of "settle time", during which MOC outputs are fixed
 * at zero and no inputs are provided to the MOC lowpass filters. This is intended to 
 * give the model some time to settle before gain control "kicks in" and begins influencing
 * responses. Generally, it is assumed that the acoustic waveform input to the model will
 * have a period of silence corresponding to dur_settle at the beginning before any sound.
 * @param [in] moc_delay (s) Duration of delay before MOC signal contributes to any changes
 * in ongoing gain factor in the peripheral stage.
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
    double *moc_beta, 
    double *moc_offset, 
    double moc_minval,
    double moc_maxval,
    double *moc_weight, 
    double moc_width,
    double dur_settle,
    double moc_delay,
    int moc_fix_gain,
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
    double **gain,
    double **gainpostmix
) {
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
            C1input[i][j] = (double*) calloc(4, sizeof(double));
        }
    }
    double*** C1output = (double***) malloc(n_chan * sizeof(double**));
    for (int i = 0; i < n_chan; i++) {
        C1output[i] = (double**) malloc(12 * sizeof(double*));
        for (int j = 0; j < 12; j++) {
            C1output[i][j] = (double*) calloc(4, sizeof(double));
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
            C2input[i][j] = (double*) calloc(4, sizeof(double));
        }
    }
    double*** C2output = (double***) malloc(n_chan * sizeof(double**));
    for (int i = 0; i < n_chan; i++) {
        C2output[i] = (double**) malloc(12 * sizeof(double*));
        for (int j = 0; j < 12; j++) {
            C2output[i][j] = (double*) calloc(4, sizeof(double));
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
    double gaininst = 0.0;

    /* Tweak alpha1 and alpha2 to get PLA to match, we have to compensate for fs and scale differences */
    if (powerlaw_mode == 2) {
        alpha1 = alpha1 / 10e3;  // this was done in updated Synapse code, weights are referenced to that implementation so we have to match
        alpha2 = alpha2 / 10e3;  // this was done in updated Synapse code, weights are referenced to that implementation so we have to match
        alpha1 = alpha1 * (tdres / (1/10e3));  // reduce weights by factor of fs/10e3 to compensate for computing more samples
        alpha2 = alpha2 * (tdres / (1/10e3));  // reduce weights by factor of fs/10e3 to compensate for computing more samples
    }

    /* Select n_process, tau, and w from Guest and Carney (2024) for PLA approximation */
    int n_process = 14;  
    double tau_slow[14];
    double tau_fast[14];
    for (int i = 0; i < 14; i++) {
        tau_slow[i] = 5e-4 * pow(10.0, 1/exp(1.0) * i);
        tau_fast[i] = 1e-1 * pow(10.0, 1/exp(1.0) * i);
    }
    double w_slow[14] = {
        1054.1349144510866, 235.42021095822022, 351.3091743124357, 99.00123234954474, 
        55.18423650003196, 28.99454378212968, 6.556134147763605, 6.558380224204848, 
        1.1576087874250394, 0.995488845827021, 0.3588672871386332, 0.1573449044190812, 
        0.010428823220777147, 0.08773889583510958
    };
    double w_fast[14] = {
        6.106637716398411, 1.1558964083697898, 1.3095958543425545, 0.785695677692722, 
        0.21835528692662018, 0.10344785429373701, 0.08413927488982781, 0.001596356536824024, 
        0.018886711336962816, 0.0008089617759213521, 0.002806098243601203, 0.0006529927704604911, 
        3.13953727422695e-5, 0.0004490670084957763
    };

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
    for (int t = 0; t < n_fiber_type; t++) {
        for (int i = 0; i < n_chan; i++) {
            for (int p = 0; p < n_process; p++) {
                // Initialize exponential process states at 0
                E1[t][i][p] = 0.0;
                E2[t][i][p] = 0.0;

                // Determine time constant for this step, convert to decay coef and save
                D1[t][i][p] = 1 - exp(-(1/(1/tdres)) / tau_slow[p]);
                D2[t][i][p] = 1 - exp(-(1/(1/tdres)) / tau_fast[p]);
            }
        }
    }

    /* Declare variables used in the subcortical stage */
    int len_settle = (int) round(dur_settle * 1/tdres); 
    int len_moc_delay = (int) round(moc_delay * 1/tdres); 
    if (len_moc_delay == 0) {  // ensure that delay is at least one sample
        len_moc_delay = 1;
    }

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

    /* Calculate weight matrix for MOC system
       Below, on each time step, every channel will produce a gain-factor value based on LSR
       (and possibly IC) rates, falling in the range (0, 1] and stored in the matrix
       `gain[c][n]`. We refer to these gain-factor values as "premix" values, because they
       are determined only by what is happening in the same channel. At the close of the
       channel loop on each time step, we will then derive "postmix" gain-factor values,
       stored in `gainpostmix[c][n]`. These values are determined by computing a weighted
       geometric average of all channels' premix gain-factor values. Each channel has its
       own set of weights. For an example channel i, these weights are determined by the 
       following equation:
           normal_pdf(d, 0.0, moc_weight_wdr)
       where `normal_pdf` is a function that computes a normalized Gaussian function, `d` is
       the distance between channel i and channel j CFs in octaves, and `moc_weight_wdr` is 
       the standard deviation of the Gaussian in octaves. Below, we pre-compute these weights
       and store them in a matrix before initiating the main time loop.
    */
    double moc_wb_weights[n_chan][n_chan];
    for (int i=0; i < n_chan; i++) {
        for (int j=0; j < n_chan; j++) {
                moc_wb_weights[i][j] = normal_pdf(log2(cf[i]/cf[j]), 0.0, moc_width);
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

            /* Pre-compute scaled gain factor for this channel and time step */
            if (!((n == 0) | (((n - len_moc_delay) < 0)))) {
                gaininst = gainpostmix[c][n-len_moc_delay];
            }

            /* Determine time constants for cochlear filters */
            Get_taubm(c, cf, species, Taumax, bmTaumax, bmTaumin, ratiobm);
            if ((n == 0) | (((n - len_moc_delay) < 0))) {  // On first sample or if we have not exceeded delay, use cohc[c]
                bmTaubm[c] = cohc[c]*(bmTaumax[c]-bmTaumin[c]) + bmTaumin[c];
            } else {                                       // Otherwise use gain[c][n - len_moc_delay]
                bmTaubm[c] = gaininst*(bmTaumax[c]-bmTaumin[c]) + bmTaumin[c];
            }
            tauwb[c] = TauWBMax[c] + (bmTaubm[c]-bmTaumax[c]) * 
                (TauWBMax[c]-TauWBMin[c])/(bmTaumax[c]-bmTaumin[c]);
            wbgain[c] = gain_groupdelay(tdres, centerfreq[c], cf[c], tauwb[c], &grdelay[c]);

            /* Pass the control-path signal through the OHC model (nonlinear transduction and 
            lowpass filtering) */
            ohcnonlinout = Boltzman(wbout, 7.0, 12.0, 5.0, 5.0);
            ohcout = OhcLowPass(ohcnonlinout, tdres, 600, n, 1.0, 2, ohc[c], ohcl[c]);
            controlout[c][n] = NLafterohc(ohcout, bmTaumin[c], bmTaumax[c], 7.0);

            /* Determine time constant and shift of C1 filter poles based on output of OHCs 
               If moc_fix_gain == true, then we simply use the gain factor specified by 
               gain[c][n] to determine tauc1
               If moc_fix_gain == false, we use either cohc[c] or a delayed copy of gain[c][n]
             */
            if ((n == 0) | (((n - len_moc_delay) < 0))) {  // On first sample or if we have not exceeded delay, use cohc[c]
                tauc1 = cohc[c]*(controlout[c][n]-bmTaumin[c]) + bmTaumin[c];
            } else {                                       // Otherwise use gain[c][n - len_moc_delay]
                tauc1 = gaininst*(controlout[c][n]-bmTaumin[c]) + bmTaumin[c];
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
                        &I2[t][c], E1[t][c], E2[t][c], D1[t][c], D2[t][c], w_slow, w_fast, n_process, n,
                        alpha1, alpha2, tdres, sout1[t][c], sout2[t][c]);
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
            if (n < len_settle) { 
                mocwdr[c][n] = 0.0;
                mocic[c][n] = 0.0;
            } else {
                filter_lowpass_iir(anrateout_lsr[c], n, moc_d, mocwdr[c]);
                filter_lowpass_iir(icout[c], n, moc_d, mocic[c]);
            }

            /* Below, we transform IC and WDR rates in this current channel into a 
               "gain factor", which will influence subsequent cochlear gain. In brief,
               lowpass-filtered IC rates (mocic) and lowpass-filtered LSR rates (mocwdr)
               are scaled and summed before being passed through a single input-output
               nonlinearity that maps rates (sp/s) a to gain factor in [0, 1]. 
             */
            if (!moc_fix_gain) {
                gain[c][n] = cohc[c] * moc_nonlinearity(moc_weight[c] * mocwdr[c][n], moc_beta[c], moc_offset[c], moc_maxval, moc_minval);
            }
        } // End main channel loop

        /* Loop over channels/CFs again once all channels are done being calculated for this 
           time step. In this next channel loop, we mix gain signals across channels. This
           cannot take place until the first channel loop is done because we need access to
           the gain factors in all channels at this time step. Since gain factors naturally
           live on a log scale, we combine gain factors from different channels according to
           a geometric average. The temporary variable contr is used to store the number of 
           contributing channels to each calculation. The final result is stored in the 
           matrix gainpostmix (i.e., gain after mixing across channels).
         */
        if (moc_width > 0.0) {
            for (int c=0; c < n_chan; c++) {
                double weight_total = 0.0;
                for (int subc = 0; subc < n_chan; subc++) {
                    gainpostmix[c][n] = gainpostmix[c][n] * pow(gain[subc][n], moc_wb_weights[c][subc]);
                    weight_total = weight_total + moc_wb_weights[c][subc];
                }
                gainpostmix[c][n] = pow(gainpostmix[c][n], 1 / weight_total);
            }
        } else {
            for (int c=0; c < n_chan; c++) {
                gainpostmix[c][n] = gain[c][n];
            }
        }
    } // end main time loop

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
 * normal_pdf
 *
 * Return value of normal distribution probability density function at specified point
 *
 * @param x Input to function
 * @param mu Mean of distribution
 * @param sigma Standard deviation of distribution
 */
double normal_pdf(double x, double mu, double sigma) {
    double var = pow(sigma, 2.0);
    double normfac = 1/sqrt(TWOPI * var);
    return normfac * exp(-1 * pow(x-mu, 2.0)/(2*var));
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
        return ((maxrate-minrate) * 1.0/(1.0 + pow(beta*(x-offset), 2.0))) + minrate; // remove 1.0
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

/**
 * C1ChirpFilt
 * 
 * C1-path chirping peripheral filter code.
 * 
 * @param x Input value for current time sample
 * @param tdres Reciprocal of sampling rate (s)
 * @param cf Characteristic frequency (Hz)
 * @param n Order of filter
 * @param taumax 
 * @param rsigma 
 * @param C1gain_norm Pointer to the normalized gain value
 * @param C1initphase Pointer to the initial phase
 * @param C1input 2D array for input values
 * @param C1output 2D array for output values
 * @return Filtered output value
 */
double C1ChirpFilt(double x, double tdres,double cf, int n, double taumax, double rsigma,
                   double *C1gain_norm, double *C1initphase, double **C1input, 
                   double **C1output) {
    double ipw, ipb, rpa, pzero, rzero, sigma0, fs_bilinear, CF, norm_gain, phase, c1filterout, temp, dy, preal, pimg;
	int i,r,order_of_pole,half_order_pole,order_of_zero;
	COMPLEX p[11]; 
	
    /* Configure locations of poles and zeros based on inputs */    
    sigma0 = 1/taumax;
    ipw    = 1.01*cf*TWOPI-50;
    ipb    = 0.2343*TWOPI*cf-1104;
    rpa    = pow(10, log10(cf)*0.9 + 0.55)+ 2000;
    pzero  = pow(10,log10(cf)*0.7+1.6)+500;
    order_of_pole    = 10;             
    half_order_pole  = order_of_pole/2;
    order_of_zero    = half_order_pole;
    fs_bilinear = TWOPI*cf/tan(TWOPI*cf*tdres/2);
    rzero       = -pzero;
    CF          = TWOPI*cf;

    /* If this is the first sample, initialize the state variables */ 
    if (n==0) {		  
        p[1].x = -sigma0;     
        p[1].y = ipw;
        p[5].x = p[1].x - rpa; 
        p[5].y = p[1].y - ipb;
        p[3].x = (p[1].x + p[5].x) * 0.5; 
        p[3].y = (p[1].y + p[5].y) * 0.5;
        p[2] = compconj(p[1]);    
        p[4] = compconj(p[3]); 
        p[6] = compconj(p[5]);
        p[7] = p[1]; 
        p[8] = p[2]; 
        p[9] = p[5]; 
        p[10] = p[6];

        /* Unknown ? */
        (*C1initphase) = 0.0;
        for (i=1; i<=half_order_pole; i++) {
            preal     = p[i*2-1].x;
            pimg      = p[i*2-1].y;
            (*C1initphase) = (*C1initphase) + atan(CF/(-rzero))-atan((CF-pimg)/(-preal))-atan((CF+pimg)/(-preal));
        };

        for (i = 1; i <= (half_order_pole+1); i++) {
            C1input[i][3] = 0; 
            C1input[i][2] = 0; 
            C1input[i][1] = 0;
            C1output[i][3] = 0; 
            C1output[i][2] = 0; 
            C1output[i][1] = 0;
        }

        /* Compute normalized gain*/
        (*C1gain_norm) = 1.0;
        for (r = 1; r <= order_of_pole; r++) {
            (*C1gain_norm) = (*C1gain_norm)*(pow((CF - p[r].y),2) + p[r].x*p[r].x);
        }
    };
     
    norm_gain= sqrt((*C1gain_norm))/pow(sqrt(CF*CF+rzero*rzero),order_of_zero);
	p[1].x = -sigma0 - rsigma;
	p[1].y = ipw;
	p[5].x = p[1].x - rpa; 
    p[5].y = p[1].y - ipb;
    p[3].x = (p[1].x + p[5].x) * 0.5; 
    p[3].y = (p[1].y + p[5].y) * 0.5;
    p[2] = compconj(p[1]); 
    p[4] = compconj(p[3]); 
    p[6] = compconj(p[5]);
    p[7] = p[1]; 
    p[8] = p[2]; 
    p[9] = p[5]; 
    p[10]= p[6];

    phase = 0.0;
    for (i = 1; i <= half_order_pole; i++) {
           preal = p[i*2-1].x;
		   pimg  = p[i*2-1].y;
	       phase = phase-atan((CF-pimg)/(-preal))-atan((CF+pimg)/(-preal));
	};

	rzero = -CF/tan(((*C1initphase)-phase)/order_of_zero);

    C1input[1][3]=C1input[1][2]; 
    C1input[1][2]=C1input[1][1]; 
    C1input[1][1]= x;

    for (i = 1; i <= half_order_pole; i++) {
        preal = p[i*2-1].x;
        pimg  = p[i*2-1].y;
        temp  = pow((fs_bilinear-preal),2)+ pow(pimg,2);

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

/**
 * C2ChirpFilt
 * 
 * C2-path chirping peripheral filter code.
 * 
 * @param xx Input value for current time sample?
 * @param tdres Reciprocal of sampling rate (s)
 * @param cf Characteristic frequency (Hz)
 * @param n Order of filter
 * @param taumax  ??
 * @param fcohc   ??
 * @param C2gain_norm Pointer to the normalized gain value
 * @param C2initphase Pointer to the initial phase
 * @param C2input 2D array for input values
 * @param C2output 2D array for output values
 * @return Filtered output value
 */
double C2ChirpFilt(double xx, double tdres,double cf, int n, double taumax, double fcohc,
                   double *C2gain_norm, double *C2initphase, double **C2input, 
                   double **C2output) {
	double ipw, ipb, rpa, pzero, rzero, sigma0, fs_bilinear, CF, norm_gain,phase, c2filterout, temp, dy, preal, pimg;
	int    i,r,order_of_pole,half_order_pole,order_of_zero;
	COMPLEX p[11]; 	

    /* Configure locations of poles and zeros based on inputs */    
    sigma0 = 1/taumax;
    ipw    = 1.01*cf*TWOPI-50;
    ipb    = 0.2343*TWOPI*cf-1104;
    rpa    = pow(10, log10(cf)*0.9 + 0.55)+ 2000;
    pzero  = pow(10,log10(cf)*0.7+1.6)+500;
    order_of_pole    = 10;             
    half_order_pole  = order_of_pole/2;
    order_of_zero    = half_order_pole;
    fs_bilinear = TWOPI*cf/tan(TWOPI*cf*tdres/2);
    rzero       = -pzero;
    CF          = TWOPI*cf;

    /* If this is the first sample, initialize the state variables */ 
    if (n==0) {		  
        p[1].x = -sigma0;     
        p[1].y = ipw;
        p[5].x = p[1].x - rpa; p[5].y = p[1].y - ipb;
        p[3].x = (p[1].x + p[5].x) * 0.5; 
        p[3].y = (p[1].y + p[5].y) * 0.5;
        p[2] = compconj(p[1]); 
        p[4] = compconj(p[3]); 
        p[6] = compconj(p[5]);
        p[7] = p[1]; 
        p[8] = p[2]; 
        p[9] = p[5]; 
        p[10]= p[6];

        /* Unknown ? */
        (*C2initphase) = 0.0;
        for (i = 1; i <= half_order_pole; i++) {
            preal     = p[i*2-1].x;
            pimg      = p[i*2-1].y;
            (*C2initphase) = (*C2initphase) + atan(CF/(-rzero))-atan((CF-pimg)/(-preal))-atan((CF+pimg)/(-preal));
        };

        /* Normalize gain? */
        (*C2gain_norm) = 1.0;
        for (r=1; r<=order_of_pole; r++) {
            (*C2gain_norm) = (*C2gain_norm)*(pow((CF - p[r].y),2) + p[r].x*p[r].x);
        }
    };

    /* Do some calculations? */ 
    norm_gain= sqrt((*C2gain_norm))/pow(sqrt(CF*CF+rzero*rzero),order_of_zero);
	p[1].x = -sigma0*fcohc;
	p[1].y = ipw;
	p[5].x = p[1].x - rpa; 
    p[5].y = p[1].y - ipb;
    p[3].x = (p[1].x + p[5].x) * 0.5; 
    p[3].y = (p[1].y + p[5].y) * 0.5;
    p[2] = compconj(p[1]); 
    p[4] = compconj(p[3]); 
    p[6] = compconj(p[5]);
    p[7] = p[1]; 
    p[8] = p[2]; 
    p[9] = p[5]; 
    p[10]= p[6];

    /* Do some more calculations? */
    phase = 0.0;
    for (i = 1; i <= half_order_pole; i++) {
           preal = p[i*2-1].x;
		   pimg  = p[i*2-1].y;
	       phase = phase-atan((CF-pimg)/(-preal))-atan((CF+pimg)/(-preal));
	};

	rzero = -CF/tan(((*C2initphase)-phase)/order_of_zero);	

    C2input[1][3]=C2input[1][2]; 
    C2input[1][2]=C2input[1][1]; 
    C2input[1][1]= xx;

    /* Loop through taps? */
    for (i = 1; i <= half_order_pole; i++) {
        preal = p[i*2-1].x;
        pimg  = p[i*2-1].y;
        temp  = pow((fs_bilinear-preal),2)+ pow(pimg,2);
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

    /* Scale output and return */
    dy = C2output[half_order_pole][1]*norm_gain;
    c2filterout= dy/4.0;

    return (c2filterout); 
}   


