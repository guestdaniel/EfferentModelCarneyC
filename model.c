 /*
  This is v0.0.12 of the code for subcortical auditory model model of:

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
  E. A. Lopez-Poveda and A. R. Palmer and R. Meddis, Springer, NY, pp. 429�438.

  Zilany, M.S.A., Bruce, I.C., Ibrahim, R.A., and Carney, L.H. (2013).
  "Improved parameters and expanded simulation options for a model of the
  auditory periphery," in Abstracts of the 36th ARO Midwinter Research Meeting.

  The peripheral stage was modified to include a sample-by-sample efferent gain control 
  loop, which is controlled by an auditory brainstem and midbrain model included in this
  code.

  Please cite these papers if you publish any research
  results obtained with this code or any modified versions of this code.

  Changelog:
  0.0.11 // a0f27528b0127e77d0e64acf6e0ac9e43d681456 [Helios]
        Removed power-law zeropadding ("delaypoint") system and moved power-law adaptation
        into primary loop. Renamed several variables related to the initial "traveling-wave
        delay" to improve clarity. 
  0.0.10 // 14a4116e964679d5e864ffda65acbf0f61937087 [Helios]
        Removed downsampling stage from the IHC/synapse model.
  0.0.9 // dbeed3c05d6468f961ec679417dc93103a2881ab [Helios]
        Moved double-exponential adaptation stage into main model loop.
  0.0.8 // 53c503450b3d6337b41fe5fe4ec6f2f39032815f [Helios]
        Converted "traveling-wave delay" that accounts for differences in ANF spiking 
        latency as a function of CF into a sample-wise implementation consisting of a 
        short delay after middle-ear filtering and before cochlear filtering.
  0.0.7 // 03ce225a9fc5ae1e8b1deeebf4358d8c1d7f7ac2
        Removed logic related to `implnt`, which controls whether to use the actual or 
        approximate implementation of power-law adaptation. Left the argument in the 
        signature of `model()` for convenience, but it now effectively does nothing.
  0.0.6 // 703fbaf9a89e1068358edb6769429e9d9f2e72aa [Helios]
        Added additional model stages after IHC, up to synapse output.
  0.0.4 // fd04a31c75961d0b0282f8f1e605d3cb54921a69 [Helios]
        Adjusted formatting and comments in a few underlying C functions (NLogarithm, 
        IhcLowPass, etc.)
  0.0.3 // 4c6b89f83f52eeea8318969c1f1b167d18546233 [Helios]
        Removed nrep as an option from the new model code. If users want to repeat 
        simulations, they should pass in a repeated stimulus or repeat the call (instead of
        using the nrep functionality provided in 2014/2018 model code).
  0.0.2 // 430393b4070a0b047c76e904cf46e2d1a99fd3f0 [Helios]
        Implemented full IHC signal path into `model()`, adjusted code to allow the caller to 
        get many time-varying outputs from different model stages as outputs so they can be 
        inspected post hoc.
  0.0.1 // 72e5632dd5825e105e6804a0bcf73791116488d2 [Helios]
        Started process of merging IHCAN.c and Synapse.c, part of the original 2014 model 
        code, into a single file, model.c. Created a new function, `model()`, intended to
        implement both the BM/IHC and AN model stages into a single samplewise loop.

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
 * 
 */
void model(double *px, double *randNums, double cf, double tdres, int totalstim,
           double cohc, double cihc, int species, double spont, double noiseType, 
           double implnt, double *meout, 
           double *controlout, double *c1out, double *c1vihcout, 
           double *c2out, double *c2vihcout, double *ihcout, double *synout,
           double *exponOut, double *powerLawIn, double *sout1, double *sout2, double *(*decimate)(double *, int, int)) {
    /* Declare variables used in the cochlear-filtering and hair-cell stage */
    double *tmpgain;
    double bmplace, centerfreq, gain, TauWBMax, TauWBMin, bmTaubm, tauwb, wbgain, 
           lasttmpgain, wbout1, wbout, ohcasym, ihcasym, ohcnonlinout, ohcout, tmptauc1, 
           tauc1, rsigma, wb_gain, c1filterouttmp, c2filterouttmp, c1vihctmp, c2vihctmp,
           me_curr, delay_transmission;
    int bmorder, wborder, grd, len_delay_transmission;
    int n, i; 
    double Taumin[1],Taumax[1], bmTaumin[1], bmTaumax[1], ratiobm[1];
    int grdelay[1];

    /* Declare variables used in the auditory-nerve stage */
    int z, b;
    int resamp = (int)ceil(1 / (tdres * 10e3));
    double incr = 0.0;

    double alpha1, beta1, I1, alpha2, beta2, I2, binwidth;
    int k, j, indx, q;
    double synstrength, synslope, CI, CL, PG, CG, VL, PL, VI;
    double cf_factor, PImax, kslope, Ass, Asp, TauR, TauST, Ar_Ast, PTS, Aon,
        AR, AST, Prest, gamma1, gamma2, k1, k2;
    double VI0, VI1, alpha, beta, theta1, theta2, theta3, vsat, tmpst, tmp, PPI,
        CIlast, temp;

    double *synouttmp;
    double *ihcDims;

    /* Declare variables used in the subcortical stage */

    /* Allocate memory for cochlear filtering and hair cell stage */
    tmpgain = (double*) calloc(totalstim, sizeof(double));

    /* Declare functions used in model */
	void middle_ear(double *, double, int, int, double *);
    double Get_tauwb(double, int, int, double *, double *);
	double Get_taubm(double, int, double, double *, double *, double *);
    double gain_groupdelay(double, double, double, double, int *);
    double WbGammaTone(double, double, double, int, double, double, int);
	double Boltzman(double, double, double, double, double);
    double NLafterohc(double, double, double, double);
    double OhcLowPass(double, double, double, int, double, int);
	double C1ChirpFilt(double, double,double, int, double, double);
	double C2ChirpFilt(double, double,double, int, double, double);
    double NLogarithm(double, double, double, double);
    double IhcLowPass(double, double, double, int, double, int);
    double delay_cat(double);
    double delay_human(double);

    /* Calculate middle-ear output */
    middle_ear(px, tdres, totalstim, species, meout);

	/* Calculate the center frequency for the control-path wideband filter
	   from the location on basilar membrane, based on Greenwood (JASA 1990) */
	if (species == 1) {
        /* Cat frequency shift corresponding to 1.2 mm */
        bmplace = 11.9 * log10(0.80 + cf / 456.0); /* Calculate the location on basilar membrane from CF */
        centerfreq = 456.0*(pow(10,(bmplace+1.2)/11.9)-0.80); /* Shift the center freq */
    }
	else {
        /* Human frequency shift corresponding to 1.2 mm */
        bmplace = (35/2.1) * log10(1.0 + cf / 165.4); /* Calculate the location on basilar membrane from CF */
        centerfreq = 165.4*(pow(10,(bmplace+1.2)/(35/2.1))-1.0); /* Shift the center freq */
    }

    /* Calculate parameters associated with cochlear gain */
	if(species == 1) { 
        gain = 52.0/2.0*(tanh(2.2*log10(cf/0.6e3)+0.15)+1.0);
    }
    else {
        gain = 52.0/2.0*(tanh(2.2*log10(cf/0.6e3)+0.15)+1.0);
    }

    /* Limit gain to range of 15 to 60 dB */
    if (gain > 60.0) gain = 60.0;
    if (gain < 15.0) gain = 15.0;

	/* Determine parameters for the ??? */
	bmorder = 3;
	Get_tauwb(cf, species, bmorder, Taumax, Taumin);
	/* taubm = cohc*(Taumax[0]-Taumin[0])+Taumin[0]; */

    /* Determine parameters for the signal-path C1 filter */
	Get_taubm(cf, species, Taumax[0], bmTaumax, bmTaumin, ratiobm);
	bmTaubm = cohc*(bmTaumax[0]-bmTaumin[0])+bmTaumin[0];

    /* Determine parameters for the control-path wideband filter */
	wborder = 3;
    TauWBMax = Taumin[0]+0.2*(Taumax[0]-Taumin[0]);
	TauWBMin = TauWBMax/Taumax[0]*Taumin[0];
    tauwb = TauWBMax+(bmTaubm-bmTaumax[0])*(TauWBMax-TauWBMin)/(bmTaumax[0]-bmTaumin[0]);
	wbgain = gain_groupdelay(tdres,centerfreq,cf,tauwb,grdelay);
	tmpgain[0] = wbgain;
	lasttmpgain = wbgain;

    /* Determine parameters for OHC/IHC transduction nonlinearities */
	ohcasym  = 7.0;    
	ihcasym  = 3.0;

    /* Calculate transmission delay time (to account for measured spike latency) */
    delay_transmission = delay_cat(cf);  /* Delay time in s */
    len_delay_transmission = __max(0, (int) ceil(delay_transmission/tdres));  /* Delay time in samples*/

    /* Set parameters for power-law */
    binwidth = tdres;
    alpha1 = 2.5e-6*100e3; beta1 = 5e-4; I1 = 0;
    alpha2 = 1e-2*100e3; beta2 = 1e-1; I2 = 0;

    /* Set parameters for double-exponential adaptation */
    if (spont==100) cf_factor = __min(800,pow(10,0.29*cf/1e3 + 0.7));
    if (spont==4)   cf_factor = __min(50,2.5e-4*cf*4+0.2);
    if (spont==0.1) cf_factor = __min(1.0,2.5e-4*cf*0.1+0.15);

    PImax  = 0.6;                /* PI2 : Maximum of the PI(PI at steady state) */
    kslope = (1+50.0)/(5+50.0)*cf_factor*20.0*PImax;
    /* Ass    = 300*TWOPI/2*(1+cf/100e3); */  /* Older value: Steady State Firing Rate eq.10 */
    Ass    = 800*(1+cf/100e3);    /* Steady State Firing Rate eq.10 */

    if (implnt==1) Asp = spont*3.0;   /* Spontaneous Firing Rate if actual implementation */
    if (implnt==0) Asp = spont*2.75; /* Spontaneous Firing Rate if approximate implementation */
    TauR   = 2e-3;               /* Rapid Time Constant eq.10 */
    TauST  = 60e-3;              /* Short Time Constant eq.10 */
    Ar_Ast = 6;                  /* Ratio of Ar/Ast */
    PTS    = 3;                  /* Peak to Steady State Ratio, characteristic of PSTH */

    Aon    = PTS*Ass;                          /* Onset rate = Ass+Ar+Ast eq.10 */
    AR     = (Aon-Ass)*Ar_Ast/(1+Ar_Ast);      /* Rapid component magnitude: eq.10 */
    AST    = Aon-Ass-AR;                       /* Short time component: eq.10 */
    Prest  = PImax/Aon*Asp;                    /* eq.A15 */
    CG  = (Asp*(Aon-Asp))/(Aon*Prest*(1-Asp/Ass));    /* eq.A16 */
    gamma1 = CG/Asp;                           /* eq.A19 */
    gamma2 = CG/Ass;                           /* eq.A20 */
    k1     = -1/TauR;                          /* eq.8 & eq.10 */
    k2     = -1/TauST;                         /* eq.8 & eq.10 */
            /* eq.A21 & eq.A22 */
    VI0    = (1-PImax/Prest)/(gamma1*(AR*(k1-k2)/CG/PImax+k2/Prest/gamma1-k2/PImax/gamma2));
    VI1    = (1-PImax/Prest)/(gamma1*(AST*(k2-k1)/CG/PImax+k1/Prest/gamma1-k1/PImax/gamma2));
    VI  = (VI0+VI1)/2;
    alpha  = gamma2/k1/k2;       /* eq.A23,eq.A24 or eq.7 */
    beta   = -(k1+k2)*alpha;     /* eq.A23 or eq.7 */
    theta1 = alpha*PImax/VI;
    theta2 = VI/PImax;
    theta3 = gamma2-1/PImax;

    PL  = ((beta-theta2*theta3)/theta1-1)*PImax;  /* eq.4' */
    PG  = 1/(theta3-1/PL);                        /* eq.5' */
    VL  = theta1*PL*PG;                           /* eq.3' */
    CI  = Asp/Prest;                              /* CI at rest, from eq.A3,eq.A12 */
    CL  = CI*(Prest+PL)/PL;                       /* CL at rest, from eq.1 */

    if(kslope>=0)  vsat = kslope+Prest;
    tmpst  = log(2)*vsat/Prest;
    if(tmpst<400) synstrength = log(exp(tmpst)-1);
    else synstrength = tmpst;
    synslope = Prest/log(2)*synstrength;

    /* Compute the main model loop*/
    k = 0;
    for (n=0; n<totalstim; n++) {
        /* Delay signal according to delay time */
        if ((n - len_delay_transmission) < 0) {
            me_curr = 0.0;
        } else {
            me_curr = meout[n - len_delay_transmission]; /* TODO Transmission or ?? */
        }

        /* Pass signal through control-path filter */
        wbout1 = WbGammaTone(me_curr, tdres, centerfreq, n, tauwb, wbgain, wborder);
        wbout = pow((tauwb/TauWBMax), wborder) * wbout1 * 10e3 *__max(1, cf/5e3);

        /* Pass the control-path signal through the OHC model (nonlinear transduction and 
           lowpass filtering) */
        ohcnonlinout = Boltzman(wbout, ohcasym, 12.0, 5.0, 5.0);
		ohcout = OhcLowPass(ohcnonlinout, tdres, 600, n, 1.0, 2);
		tmptauc1 = NLafterohc(ohcout, bmTaumin[0], bmTaumax[0], ohcasym);
        controlout[n] = tmptauc1;  /* store sample output in vector */

        /* Determine time constant and shift of C1 filter poles based on output of OHCs */
		tauc1 = cohc*(tmptauc1-bmTaumin[0]) + bmTaumin[0]; 
		rsigma = 1/tauc1 - 1/bmTaumax[0];

		tauwb = TauWBMax + (tauc1-bmTaumax[0])*(TauWBMax-TauWBMin)/(bmTaumax[0]-bmTaumin[0]);

        wb_gain = gain_groupdelay(tdres, centerfreq, cf, tauwb, grdelay);

		grd = grdelay[0];

        if ((grd+n) < totalstim) {
            tmpgain[grd+n] = wb_gain;
        }
        if (tmpgain[n] == 0) {
			tmpgain[n] = lasttmpgain;
        }

		wbgain = tmpgain[n];
		lasttmpgain = wbgain;

        /* Apply signal-path C1 filter */
	    c1filterouttmp = C1ChirpFilt(me_curr, tdres, cf, n, bmTaumax[0], rsigma);
        c1out[n] = c1filterouttmp;  /* store sample output in vector */

        /* Apply parallel-path C2 filter */
		c2filterouttmp  = C2ChirpFilt(me_curr, tdres, cf, n, bmTaumax[0], 1/ratiobm[0]);
        c2out[n] = c2filterouttmp;  /* store sample output in vector */

	    /* Apply IHC model: NL input-output function and lowpass filtering */
        c1vihctmp  = NLogarithm(cihc*c1filterouttmp, 0.1, ihcasym, cf);
        c1vihcout[n] = c1vihctmp;  /* store sample output in vector */
		c2vihctmp = -NLogarithm(c2filterouttmp*fabs(c2filterouttmp)*cf/10*cf/2e3, 0.2, 1.0, cf); /* C2 transduction output */
        c2vihcout[n] = c2vihctmp;  /* store sample output in vector */
        ihcout[n] = IhcLowPass(c1vihctmp+c2vihctmp, tdres, 3000, n, 1.0, 7);

        /* Apply double-exponential adaptation */
        tmp = synstrength*(ihcout[n]);
        if(tmp<400) tmp = log(1+exp(tmp));
        PPI = synslope/synstrength*tmp;
        CIlast = CI;
        CI = CI + (tdres/VI)*(-PPI*CI + PL*(CL-CI));
        CL = CL + (tdres/VL)*(-PL*(CL - CIlast) + PG*(CG - CL));
        if (CI < 0) {
            temp = 1/PG+1/PL+1/PPI;
            CI = CG/(PPI*temp);
            CL = CI*(PPI+PL)/PL;
        }
        exponOut[n] = CI*PPI;

        /* Apply power-law adaptation */
        sout1[k]  = __max(0, exponOut[n] + randNums[n] - alpha1*I1);
        sout2[k] = __max(0, exponOut[n] - alpha2 * I2);
        I1 = 0; I2 = 0;
        for (j=0; j<k+1; ++j) {
            I1 += (sout1[j])*binwidth/((k-j)*binwidth + beta1);
            I2 += (sout2[j])*binwidth/((k-j)*binwidth + beta2);
        }
        synout[k] = sout1[k] + sout2[k];
        k = k+1;
    }

    /* Free dynamic memory */
    free(tmpgain);
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
 * For a given CF, species, and order, return tau values for the control-path wideband filter
 * 
 * @param cf Characteristic frequency (Hz)
 * @param species what species to simulate (1==cat, 2=human[shera], 3==human[glasberg])
 * @param order ???
 * @param taumax ???
 * @param taumin ???
 */
double Get_tauwb(double cf, int species, int order, double *taumax, double *taumin) {
    double Q10, bw, gain, ratio;

    /* Calculate gain for cats (species == 1) or humans */
    if (species == 1) {
        gain = 52.0/2.0*(tanh(2.2*log10(cf/0.6e3)+0.15)+1.0);
    } 
    else {
        gain = 52.0/2.0*(tanh(2.2*log10(cf/0.6e3)+0.15)+1.0);
    } 

    /* Limit gain to 15-60 dB range */
    if(gain>60.0) gain = 60.0;  
    if(gain<15.0) gain = 15.0;

    /* Calculate ratio of TauMin/TauMax according to the gain and order */
    ratio = pow(10,(-gain/(20.0*order)));

    /* Calculate Q10 values for cats (species == 1) or humans (Glasberg == 2, Shera == 3) */
    /* Values for Shera come from Shera et al. (PNAS 2002) */
    /* Values for Glasberg come from Glasberg and Moore (Hear. Res. 1999) */
    if (species == 1) {
        Q10 = pow(10,0.4708*log10(cf/1e3)+0.4664);
    }
    if (species==2) {
        Q10 = pow((cf/1000),0.3)*12.7*0.505+0.2085;
    }
    else {
        Q10 = cf/24.7/(4.37*(cf/1000)+1)*0.505+0.2085;
    }

    /* Calculate bandwidth, taumax, and taumin */
    bw = cf/Q10;
    taumax[0] = 2.0/(TWOPI*bw);
    taumin[0] = taumax[0]*ratio;
    
    return 0;
}

/**
 * Get_taubm
 * 
 * For a given CF, species, and order, return tau values for the signal-path narrowband filter
 * 
 * @param cf Characteristic frequency (Hz)
 * @param species what species to simulate (1==cat, 2=human[shera], 3==human[glasberg])
 * @param taumax ??
 * @param bmTaumax ???
 * @param bmTaumin ???
 * @param ratio ???
 */
double Get_taubm(double cf, int species, double taumax,double *bmTaumax,double *bmTaumin, 
                 double *ratio) {
    double gain,factor,bwfactor;

    /* Calculate gain for cats (species == 1) or humans */
    if (species == 1) {
        gain = 52.0/2.0*(tanh(2.2*log10(cf/0.6e3)+0.15)+1.0);
    }
    else {
        gain = 52.0/2.0*(tanh(2.2*log10(cf/0.6e3)+0.15)+1.0); 
    }

    /* Limit gain to 15-60 dB range */
    if(gain>60.0) gain = 60.0;  
    if(gain<15.0) gain = 15.0;

    /* Calculate bmTaumax, bmTaumin, ratio */
    bwfactor = 0.7;
    factor   = 2.5;
    ratio[0]  = pow(10,(-gain/(20.0*factor))); 
    bmTaumax[0] = taumax/bwfactor;
    bmTaumin[0] = bmTaumax[0]*ratio[0];     

  return 0;
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
double gain_groupdelay(double tdres,double centerfreq, double cf, double tau,int *grdelay) { 
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
    grdelay[0] = (int)floor((0.5-(c1LP*c1LP-c1LP*tmpcos)/(1+c1LP*c1LP-2*c1LP*tmpcos)));

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
double OhcLowPass(double x, double tdres, double Fc, int n, double gain, int order) {
    static double ohc[4],ohcl[4];
    double c,c1LP,c2LP;
    int i,j;

    /* If we're on the first sample, initialize static memory to zeros */
    if (n == 0) {
        for (i=0; i<(order+1); i++) {
            ohc[i] = 0;
            ohcl[i] = 0;
        }
    }    

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
double IhcLowPass(double x, double tdres, double Fc, int n, double gain, int order) {
    static double ihc[8],ihcl[8];
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

double WbGammaTone(double x,double tdres,double centerfreq, int n, double tau,double gain,int order)
{
  static double wbphase;
  static COMPLEX wbgtf[4], wbgtfl[4];

  double delta_phase,dtmp,c1LP,c2LP,out;
  int i,j;
  
  if (n==0)
  {
      wbphase = 0;
      for(i=0; i<=order;i++)
      {
            wbgtfl[i] = compmult(0,compexp(0));
            wbgtf[i]  = compmult(0,compexp(0));
      }
  }
  
  delta_phase = -TWOPI*centerfreq*tdres;
  wbphase += delta_phase;
  
  dtmp = tau*2.0/tdres;
  c1LP = (dtmp-1)/(dtmp+1);
  c2LP = 1.0/(dtmp+1);
  wbgtf[0] = compmult(x,compexp(wbphase));                 /* FREQUENCY SHIFT */
  
  for(j = 1; j <= order; j++)                              /* IIR Bilinear transformation LPF */
  wbgtf[j] = comp2sum(compmult(c2LP*gain,comp2sum(wbgtf[j-1],wbgtfl[j-1])),
      compmult(c1LP,wbgtfl[j]));
  out = REAL(compprod(compexp(-wbphase), wbgtf[order])); /* FREQ SHIFT BACK UP */
  
  for(i=0; i<=order;i++) wbgtfl[i] = wbgtf[i];
  return(out);
}

double C1ChirpFilt(double x, double tdres,double cf, int n, double taumax, double rsigma)
{
    static double C1gain_norm, C1initphase; 
    static double C1input[12][4], C1output[12][4];

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

	   C1initphase = 0.0;
       for (i=1;i<=half_order_pole;i++)          
	   {
           preal     = p[i*2-1].x;
		   pimg      = p[i*2-1].y;
	       C1initphase = C1initphase + atan(CF/(-rzero))-atan((CF-pimg)/(-preal))-atan((CF+pimg)/(-preal));
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
    
      C1gain_norm = 1.0;
      for (r=1; r<=order_of_pole; r++)
		   C1gain_norm = C1gain_norm*(pow((CF - p[r].y),2) + p[r].x*p[r].x);
      
   };
     
    norm_gain= sqrt(C1gain_norm)/pow(sqrt(CF*CF+rzero*rzero),order_of_zero);
	
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

	rzero = -CF/tan((C1initphase-phase)/order_of_zero);

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

double C2ChirpFilt(double xx, double tdres,double cf, int n, double taumax, double fcohc)
{
	static double C2gain_norm, C2initphase;
    static double C2input[12][4];  static double C2output[12][4];
   
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

	   C2initphase = 0.0;
       for (i=1;i<=half_order_pole;i++)         
	   {
           preal     = p[i*2-1].x;
		   pimg      = p[i*2-1].y;
	       C2initphase = C2initphase + atan(CF/(-rzero))-atan((CF-pimg)/(-preal))-atan((CF+pimg)/(-preal));
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
    
     C2gain_norm = 1.0;
     for (r=1; r<=order_of_pole; r++)
		   C2gain_norm = C2gain_norm*(pow((CF - p[r].y),2) + p[r].x*p[r].x);
    };
     
    norm_gain= sqrt(C2gain_norm)/pow(sqrt(CF*CF+rzero*rzero),order_of_zero);
    
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

	rzero = -CF/tan((C2initphase-phase)/order_of_zero);	
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
