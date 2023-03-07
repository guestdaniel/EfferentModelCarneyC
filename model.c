 /*
  This is v0.1.1 of the code for subcortical auditory model model of:

  Guest, D. R., ..., and Carney, L. H. (202x). ...

  The peripheral stage of this model is derived from the work of:

  Zilany, M.S.A., Bruce, I.C., Nelson, P.C., and Carney, L.H. (2009). "A
  Phenomenological model of the synapse between the inner hair cell and auditory
  nerve : Long-term adaptation with power-law dynamics," Journal of the
  Acoustical Society of America 126(5): 2390-2412.
 
  with the modifications described in:

  Ibrahim, R. A., and Bruce, I. C. (2010). "Effects of peripheral tuning
  on the auditory nerve's representation of speech envelope and temporal fine
  structure cues," in The Neurophysiological Bases of Auditory Perception, eds.
  E. A. Lopez-Poveda and A. R. Palmer and R. Meddis, Springer, NY, pp. 429-438.

  Zilany, M.S.A., Bruce, I.C., Ibrahim, R.A., and Carney, L.H. (2013).
  "Improved parameters and expanded simulation options for a model of the
  auditory periphery," in Abstracts of the 36th ARO Midwinter Research Meeting.

  Zilany, M. S., Bruce, I. C., & Carney, L. H. (2014). Updated parameters and expanded 
  simulation options for a model of the auditory periphery. The Journal of the Acoustical 
  Society of America, 135(1), 283-286.

  The peripheral stage was modified to include a sample-by-sample efferent gain control 
  loop, which is controlled by an auditory brainstem and midbrain model included in this
  code.

  Please cite these papers if you publish any research results obtained with this code or 
  any modified versions of this code.

  To compare this code to older model versions, please refer to changelog.txt and to the 
  git commit history. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "complex.hpp"

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
 * inferior colliculus, medial olivary complex) responses to a sound-pressure waveform using
 * a modified version of the periphery model of Zilany, Bruce, and Carney (2014) and a
 * new brainstem and midbrain model described in [[cite]].
 * 
 * @param px Vector containing sound-pressure waveform (Pa), size (totalstim, )
 * @param randNums Matrix of fractional Gaussian noise, size (n_chan, totalstim), see 
 *   2014 paper for details on how this noise should be synthesized. Each channel's noise
 *   should be independent.
 * @param cf Vector containing characteristic frequencies for each channel (Hz), 
 *  size (n_chan, )
 * @param tdres Sample time resolution (s), i.e., reciprocal of the sampling rate (1/Hz)
 * @param totalstim Number of samples in the simulation
 * @param cohc Scalar value in [0, 1] controlling the maximum [[cohc]]
 * @param cihc Scalar value in [0, 1] controlling the maximum output of the inner hair cells
 * @param species Integer indicating what species to simulate (1==cat, 2=human[shera], 3==human[glasberg])
 * @param spont Float indicating what fiber type to simulate (100.0==HSR, 4.0==MSR, 0.1==LSR)
 * @param meout Vector to store output of middle-ear filter, size (totalstim, )
 * @param controlout Matrix to store output of control-path filter, size (n_chan, totalstim)
 * @param c1out Matrix to store output of signal-path C1 filter, size (n_chan, totalstim)
 * @param c1vihcout Matrix to store output of signal-path C1 filter after IHC transduction, 
 *   size (n_chan, totalstim)
 * @param c2out Matrix to store output of signal-path C2 filter, size (n_chan, totalstim)
 * @param c2vihcout Matrix to store output of signal-path C2 filter after IHC transduction, 
 *   size (n_chan, totalstim)
 * @param ihcout Matrix to store output of inner hair cell, size (n_chan, totalstim)
 * @param synout Matrix to store output of synapse, size (n_chan, totalstim)
 * @param exponOut Matrix to store output of exponential adaptation stage, size (n_chan, totalstim)
 * @param sout1 Matrix to store output of slow power-law adaptation stage, size (n_chan, totalstim)
 * @param sout2 Matrix to store output of fast power-law adaptation stage, size (n_chan, totalstim)
 */
void model(double *px, double **randNums, double *cf, int n_chan, double tdres, 
           int totalstim, double cohc, double cihc, int species, double spont, 
           double *meout, double **controlout, double **c1out, double **c1vihcout, 
           double **c2out, double **c2vihcout, double **ihcout, double **synout, 
           double **exponOut, double **sout1, double **sout2) {
    /* Declare variables for BM stage requiring dynamic allocation, and allocate */
    double *tmpgain[n_chan];
    for (int i = 0; i < n_chan; i++) {
        tmpgain[i] = (double*) calloc(totalstim, sizeof(double));
    }

    /* Declare variables for the BM/IHC stage to temporarily store outputs */
    double wbout1, wbout, ohcnonlinout, ohcout, tauc1, rsigma, wb_gain, c1vihctmp, c2vihctmp,
           me_curr;

    /* Declare variables for parameters in the BM/IHC stage that vary by channel */
    double bmplace[n_chan], centerfreq[n_chan], Taumin[n_chan], Taumax[n_chan], 
           bmTaumin[n_chan], bmTaumax[n_chan], ratiobm[n_chan], bmTaubm[n_chan], 
           TauWBMax[n_chan], TauWBMin[n_chan], tauwb[n_chan], wbgain[n_chan],
           delay_latency[n_chan], lasttmpgain[n_chan];
    int grdelay[n_chan], len_delay_latency[n_chan];

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

    /* Delcare and initialize state variables for OhcLowPass filter */
    double *ohc[n_chan], *ohcl[n_chan];
    for (int i = 0; i < n_chan; i++) {
        ohc[i] = (double*) calloc(4, sizeof(double));
        ohcl[i] = (double*) calloc(4, sizeof(double));
    }

    /* Delcare and initialize state variables for C1ChirpFilt filter */
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

    /* Initialize variables for the IHC lowpass filter */
    double *ihc[n_chan], *ihcl[n_chan];
    for (int i = 0; i < n_chan; i++) {
        ihc[i] = (double*) calloc(8, sizeof(double));
        ihcl[i] = (double*) calloc(8, sizeof(double));
    }

    /* Temporary */
    double tmp, PPI, CIlast, temp;

    /* Declare variables used in the AN stage that are fixed across channels */
    double alpha1, beta1, alpha2, beta2, PImax, Asp, TauR, TauST, Ar_Ast, PTS,
           k1, k2;

    /* Declare variables used in the AN stage that vary by channel */
    double cf_factor[n_chan], kslope[n_chan], Ass[n_chan], Aon[n_chan], AR[n_chan],
           AST[n_chan], Prest[n_chan], CG[n_chan], gamma1[n_chan], gamma2[n_chan],
           VI0[n_chan], VI1[n_chan], VI[n_chan], alpha[n_chan], beta[n_chan], 
           theta1[n_chan], theta2[n_chan], theta3[n_chan], PL[n_chan], PG[n_chan],
           CI[n_chan], CL[n_chan], VL[n_chan], vsat[n_chan], tmpst[n_chan], 
           synstrength[n_chan], synslope[n_chan], I1[n_chan], I2[n_chan];

    /* Declare variables used in the AN stage to temporarily store outputs */

    /* Allocate memory for cochlear filtering and hair cell stage */

    /* Declare functions used in model */
	void middle_ear(double *, double, int, int, double *);
    void Get_tauwb(int, double *, int, int, double *, double *);
	void Get_taubm(int, double *, int, double *, double *, double *, double *);
    double gain_groupdelay(double, double, double, double, int *);
    double WbGammaTone(double, double, double, int, double, double, int, double *, COMPLEX *, COMPLEX *);
	double Boltzman(double, double, double, double, double);
    double NLafterohc(double, double, double, double);
    double OhcLowPass(double, double, double, int, double, int, double *, double *);
	double C1ChirpFilt(double, double,double, int, double, double, double *, double *, double **, double **);
	double C2ChirpFilt(double, double,double, int, double, double, double *, double *, double **, double **);
    double NLogarithm(double, double, double, double);
    double IhcLowPass(double, double, double, int, double, int, double *, double *);
    double delay_cat(double);
    double delay_human(double);

    /* Calculate middle-ear output */
    middle_ear(px, tdres, totalstim, species, meout);

    /* Calculate all parameters, coefficients, etc. for each channel */
    for (int c = 0; c < n_chan; c++) {
        /* Calculate the center frequency for the control-path wideband filter as center
        frequency corresponding to a 1.2-mm basal shift from requested CF.

        This shift seeks to account for observations that suppression tuning curves have
        slightly higher tip frequencies than excitatory tuning curves (pg. 1450, Zilany
        and Bruce, 2006).

        The details of this shift depends on the species in question (cat vs human).
        */
        if (species == 1) {
            bmplace[c] = 11.9 * log10(0.80 + cf[c]/456.0); /* Calculate the location on basilar membrane from CF */
            centerfreq[c] = 456.0 * (pow(10, (bmplace[c]+1.2)/11.9) - 0.80); /* Shift the center freq */
        }
        else {
            bmplace[c] = (35/2.1) * log10(1.0 + cf[c]/165.4); /* Calculate the location on basilar membrane from CF */
            centerfreq[c] = 165.4 * (pow(10, (bmplace[c]+1.2)/(35/2.1)) - 1.0); /* Shift the center freq */
        }

        /* Determine parameters for the ??? */
        Get_tauwb(c, cf, species, 3, Taumax, Taumin);

        /* Determine parameters for the signal-path C1 filter */
        Get_taubm(c, cf, species, Taumax, bmTaumax, bmTaumin, ratiobm);
        bmTaubm[c] = cohc*(bmTaumax[c]-bmTaumin[c]) + bmTaumin[c];

        /* Determine parameters for the control-path wideband filter */
        TauWBMax[c] = Taumin[c] + 0.2*(Taumax[c]-Taumin[c]); // Eq 4 - [[2006]]
        TauWBMin[c] = TauWBMax[c]/Taumax[c]*Taumin[c];
        tauwb[c] = TauWBMax[c] + (bmTaubm[c]-bmTaumax[c])*(TauWBMax[c]-TauWBMin[c])/(bmTaumax[c]-bmTaumin[c]);
        wbgain[c] = gain_groupdelay(tdres, centerfreq[c], cf[c], tauwb[c], &grdelay[c]);
        tmpgain[c][0] = wbgain[c];
        lasttmpgain[c] = wbgain[c];

        /* Calculate latency delay time (to account for measured spike latency) */
        delay_latency[c] = delay_cat(cf[c]);  /* Delay time in s */
        len_delay_latency[c] = __max(0, (int) ceil(delay_latency[c]/tdres));  /* Delay time in samples*/

        /* Set parameters for power-law adaptation */
        alpha1 = 2.5e-6*100e3; 
        beta1  = 5e-4; 
        alpha2 = 1e-2*100e3; 
        beta2  = 1e-1; 
        for (int i = 0; i < n_chan; i++) {
            I1[i] = 0.0;
            I2[i] = 0.0;
        }

        /* Set parameters for double-exponential adaptation (based on fiber type / spont) */
        if (spont == 100) {
            cf_factor[c] = __min(800, pow(10, 0.29*cf[c]/1e3 + 0.7));
        } else if (spont == 4) {
            cf_factor[c] = __min(50, 2.5e-4*cf[c]*4 + 0.2);
        } else {
            cf_factor[c] = __min(1.0, 2.5e-4*cf[c]*0.1 + 0.15);
        }
        
        PImax  = 0.6;                /* PI2 : Maximum of the PI(PI at steady state) */
        Asp = spont*3.0;             /* Spontaneous Firing Rate */
        TauR   = 2e-3;               /* Rapid Time Constant eq.10 */
        TauST  = 60e-3;              /* Short Time Constant eq.10 */
        Ar_Ast = 6;                  /* Ratio of Ar/Ast */
        PTS    = 3;                  /* Peak to Steady State Ratio, characteristic of PSTH */
        k1     = -1/TauR;                          /* eq.8 & eq.10 */
        k2     = -1/TauST;                         /* eq.8 & eq.10 */

        kslope[c] = (1+50.0)/(5+50.0)*cf_factor[c]*20.0*PImax;
        Ass[c] = 800*(1+cf[c]/100e3);    /* Steady State Firing Rate eq.10 */

        Aon[c] = PTS*Ass[c];                          /* Onset rate = Ass+Ar+Ast eq.10 */
        AR[c] = (Aon[c]-Ass[c])*Ar_Ast/(1+Ar_Ast);      /* Rapid component magnitude: eq.10 */
        AST[c] = Aon[c]-Ass[c]-AR[c];                       /* Short time component: eq.10 */
        Prest[c] = PImax/Aon[c]*Asp;                    /* eq.A15 */
        CG[c] = (Asp*(Aon[c]-Asp))/(Aon[c]*Prest[c]*(1-Asp/Ass[c]));    /* eq.A16 */
        gamma1[c] = CG[c]/Asp;                           /* eq.A19 */
        gamma2[c] = CG[c]/Ass[c];                           /* eq.A20 */
                /* eq.A21 & eq.A22 */
        VI0[c]    = (1-PImax/Prest[c])/(gamma1[c]*(AR[c]*(k1-k2)/CG[c]/PImax+k2/Prest[c]/gamma1[c]-k2/PImax/gamma2[c]));
        VI1[c]    = (1-PImax/Prest[c])/(gamma1[c]*(AST[c]*(k2-k1)/CG[c]/PImax+k1/Prest[c]/gamma1[c]-k1/PImax/gamma2[c]));
        VI[c]  = (VI0[c]+VI1[c])/2;
        alpha[c]  = gamma2[c]/k1/k2;       /* eq.A23,eq.A24 or eq.7 */
        beta[c]   = -(k1+k2)*alpha[c];     /* eq.A23 or eq.7 */
        theta1[c] = alpha[c]*PImax/VI[c];
        theta2[c] = VI[c]/PImax;
        theta3[c] = gamma2[c]-1/PImax;

        PL[c]  = ((beta[c]-theta2[c]*theta3[c])/theta1[c]-1)*PImax;  /* eq.4' */
        PG[c]  = 1/(theta3[c]-1/PL[c]);                        /* eq.5' */
        VL[c]  = theta1[c]*PL[c]*PG[c];                           /* eq.3' */
        CI[c]  = Asp/Prest[c];                              /* CI at rest, from eq.A3,eq.A12 */
        CL[c]  = CI[c]*(Prest[c]+PL[c])/PL[c];                       /* CL at rest, from eq.1 */

        if (kslope[c] >= 0) vsat[c] = kslope[c]+Prest[c];
        tmpst[c] = log(2)*vsat[c]/Prest[c];
        if(tmpst[c]<400) synstrength[c] = log(exp(tmpst[c])-1);
        else synstrength[c] = tmpst[c];
        synslope[c] = Prest[c]/log(2)*synstrength[c];
    }

    /* Compute the main model loop*/
    for (int n=0; n<totalstim; n++) {
        for (int c=0; c < n_chan; c++) {
            /* Delay signal according to delay time */
            if ((n - len_delay_latency[c]) < 0) {
                me_curr = 0.0;
            } else {
                me_curr = meout[n - len_delay_latency[c]]; 
            }

            /* Pass signal through control-path filter */
            wbout1 = WbGammaTone(me_curr, tdres, centerfreq[c], n, tauwb[c], wbgain[c], 3, &wbphase[c], wbgtf[c], wbgtfl[c]);
            wbout = pow((tauwb[c]/TauWBMax[c]), 3) * wbout1 * 10e3 *__max(1, cf[c]/5e3);

            /* Pass the control-path signal through the OHC model (nonlinear transduction and 
            lowpass filtering) */
            ohcnonlinout = Boltzman(wbout, 7.0, 12.0, 5.0, 5.0);
            ohcout = OhcLowPass(ohcnonlinout, tdres, 600, n, 1.0, 2, ohc[c], ohcl[c]);
            controlout[c][n] = NLafterohc(ohcout, bmTaumin[c], bmTaumax[c], 7.0);

            /* Determine time constant and shift of C1 filter poles based on output of OHCs */
            tauc1 = cohc*(controlout[c][n]-bmTaumin[c]) + bmTaumin[c]; 
            rsigma = 1/tauc1 - 1/bmTaumax[c];
            tauwb[c] = TauWBMax[c] + (tauc1-bmTaumax[c])*(TauWBMax[c]-TauWBMin[c])/(bmTaumax[c]-bmTaumin[c]);

            /* Do some sort of groupdelay manipulation??? TODO Fix this */
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
            c1out[c][n] = C1ChirpFilt(me_curr, tdres, cf[c], n, bmTaumax[c], rsigma, &C1gain_norm[c], &C1initphase[c], C1input[c], C1output[c]);

            /* Apply parallel-path C2 filter */
            c2out[c][n] = C2ChirpFilt(me_curr, tdres, cf[c], n, bmTaumax[c], 1/ratiobm[c], &C2gain_norm[c], &C2initphase[c], C2input[c], C2output[c]);

            /* Apply IHC model: NL input-output function and lowpass filtering */
            c1vihctmp  = NLogarithm(cihc*c1out[c][n], 0.1, 3.0, cf[c]);
            c1vihcout[c][n] = c1vihctmp;  /* store sample output in vector */
            c2vihctmp = -NLogarithm(c2out[c][n]*fabs(c2out[c][n])*cf[c]/10*cf[c]/2e3, 0.2, 1.0, cf[c]); /* C2 transduction output */
            c2vihcout[c][n] = c2vihctmp;  /* store sample output in vector */
            ihcout[c][n] = IhcLowPass(c1vihctmp+c2vihctmp, tdres, 3000, n, 1.0, 7, ihc[c], ihcl[c]);

            /* Apply double-exponential adaptation */
            tmp = synstrength[c]*(ihcout[c][n]);
            if(tmp<400) tmp = log(1 + exp(tmp));
            PPI = synslope[c]/synstrength[c]*tmp;
            CIlast = CI[c];
            CI[c] = CI[c] + (tdres/VI[c])*(-PPI*CI[c] + PL[c]*(CL[c]-CI[c]));
            CL[c] = CL[c] + (tdres/VL[c])*(-PL[c]*(CL[c]-CIlast) + PG[c]*(CG[c]-CL[c]));
            if (CI[c] < 0) {
                temp = 1/PG[c] + 1/PL[c] + 1/PPI;
                CI[c] = CG[c]/(PPI*temp);
                CL[c] = CI[c]*(PPI+PL[c])/PL[c];
            }
            exponOut[c][n] = CI[c]*PPI;

            /* Apply power-law adaptation */
            sout1[c][n]  = __max(0, exponOut[c][n] + randNums[c][n] - alpha1*I1[c]);
            sout2[c][n] = __max(0, exponOut[c][n] - alpha2 * I2[c]);
            I1[c] = 0.0; I2[c] = 0.0;
            for (int j = 0; j < n+1; ++j) {
                I1[c] += (sout1[c][j])*tdres/((n-j)*tdres + beta1);
                I2[c] += (sout2[c][j])*tdres/((n-j)*tdres + beta2);
            }
            synout[c][n] = sout1[c][n] + sout2[c][n];
        }
    }

    /* Free dynamic memory */
    for (int i = 0; i < n_chan; i++) {
        free(tmpgain[i]);
    }
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
 */
void Get_tauwb(int idx_chan, double *cf, int species, int order, double *taumax, double *taumin) {
    double Q10, bw, gain, ratio;

    /* Calculate gain for cats (species == 1) or humans */
    if (species == 1) {
        gain = 52.0/2.0 * (tanh(2.2*log10(cf[idx_chan]/0.6e3)+0.15) + 1.0);  // Eq 6 - [[2006]] TODO resolve imperfect match
    } 
    else {
        gain = 52.0/2.0 * (tanh(2.2*log10(cf[idx_chan]/0.6e3)+0.15) + 1.0);
    } 

    /* Limit gain to 15-60 dB range */
    if (gain > 60.0) gain = 60.0;  
    if (gain < 15.0) gain = 15.0;

    /* Calculate ratio of TauMin/TauMax according to the gain and order */
    ratio = pow(10, (-gain/(20.0*order)));  // pg 1451, top left - [[2006]]

    /* Calculate Q10 values for cats (species == 1) or humans (Shera == 2, Glasberg == 3) */
    /* Values for cat come from ... */
    /* Values for Shera come from Shera et al. (PNAS 2002) */
    /* Values for Glasberg come from Glasberg and Moore (Hear. Res. 1999) */
    if (species == 1) {
        Q10 = pow(10, 0.4708*log10(cf[idx_chan]/1e3) + 0.4664);  // pg 1451, top left - [[2006]]
    }
    if (species==2) {
        Q10 = pow((cf[idx_chan]/1000), 0.3) * 12.7*0.505 + 0.2085;
    }
    else {
        Q10 = cf[idx_chan] / 24.7 / (4.37*(cf[idx_chan]/1000)+1)*0.505 + 0.2085;
    }

    /* Calculate bandwidth, taumax, and taumin */
    bw = cf[idx_chan]/Q10;
    taumax[idx_chan] = 2.0/(TWOPI*bw);  // pg 1451, top left - [[2006]]
    taumin[idx_chan] = taumax[idx_chan]*ratio; // Eq 5 - [[2006]]
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
 * 
 * TODO Add references to equations 
 */
void Get_taubm(int idx_chan, double *cf, int species, double *taumax, double *bmTaumax,
               double *bmTaumin, double *ratio) {
    double gain,factor,bwfactor;

    /* Calculate gain for cats (species == 1) or humans */
    if (species == 1) {
        gain = 52.0/2.0 * (tanh(2.2*log10(cf[idx_chan]/0.6e3)+0.15) + 1.0);
    }
    else {
        gain = 52.0/2.0 * (tanh(2.2*log10(cf[idx_chan]/0.6e3)+0.15) + 1.0); 
    }

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
 * Boltzman
 * 
 * Boltzman nonlinearity relating input ? to output ?
 * 
 * @param x input value (???)
 * @param asym Parameter controlling ratio of (positive) max output to negative (min) output
 * @param s0 Parameter controlling ???
 * @param s1 Parameter controlling ???
 * @param x1 Parameter controlling ???
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
 * Nonlinearity applied to OHC output
 * 
 * @param x input value (???)
 * @param taumin ???
 * @param taumax ???
 * @param asym ???
 */
double NLafterohc(double x,double taumin, double taumax, double asym) {    
	double R,dc,R1,s0,x1,out,minR;

    /* Calculate constants and parameters */
	minR = 0.05;
    R  = taumin/taumax;
    
	if (R < minR) minR = 0.5*R;
    else minR = minR;
    
    dc = (asym-1)/(asym+1.0)/2.0-minR;
    R1 = R-minR;

    /* TODO: Update comment /// This is for new nonlinearity */
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
