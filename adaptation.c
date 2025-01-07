#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef __max
#define __max(a,b) (((a) > (b))? (a): (b))
#endif

#ifndef __min
#define __min(a,b) (((a) < (b))? (a): (b))
#endif

/**
 * initialize_ws1988_adaptation
 * 
 * Calculates params and initial values for state variables for W&S three-store adaptation
 * 
 * Calculates parameter values, coefficients, and initial state-variable values for 
 * the Westerman and Smith (1988) three-store diffusion model of auditory-nerve synaptic 
 * adaptation:
 * 
 * Westerman, L. A., & Smith, R. L. (1988). A diffusion model of the transient response of 
 * the cochlear inner hair cell synapse. The Journal of the Acoustical Society of America, 
 * 83(6), 2266-2276.
 * 
 * Note that the first two arguments, cf and spont, configure parameters to be CF- and
 * spontaneous-rate dependent in a way that is unique to the Zilany/Bruce/Carney auditory
 * periphery model. The remaining arguments are pointers to various output variables.
 * Notably, the synaptic gain, relating IHC potential to synaptic output, varies as a 
 * function of CF to ensure model thresholds match empirical thresholds (pg 1454, Zilany and
 * Bruce 2006).
 * 
 * See apply_ws1988_adaptation for more details on background and implementation
 * 
 * @param cf Characteristic frequency (Hz)
 * @param spont Spontaneous rate (sp/s), either 100.0==HSR, 4.0==MSR, or 0.1==LSR
 * @param synstrength_out Pointer to output for synstrength variable
 * @param synslope_out Pointer to output for synslope variable
 * @param VI_out Pointer to output for VI variable
 * @param PG_out Pointer to output for PG variable
 * @param VL_out Pointer to output for VL variable
 * @param PL_out Pointer to output for PL variable
 * @param CG_out Pointer to output for CG variable
 * @param CI_out Pointer to output for CI variable
 * @param CL_out Pointer to output for CL variable
 */
void initialize_ws1988_adaptation(double cf, double spont, double *synstrength_out, 
                                  double *synslope_out, double *VI_out, double *PG_out,
                                  double *VL_out, double *PL_out, double *CG_out, 
                                  double *CI_out, double *CL_out) {
    /* Declare variables */
    double cf_factor, PImax, Asp, TauR, TauST, Ar_Ast, PTS, k1, k2, kslope, Ass, Aon, AR, 
           AST, Prest, CG, gamma1, gamma2, VI0, VI1, VI, alpha, beta, theta1, theta2, 
           theta3, PL, PG, VL, CI, CL, vsat, tmpst, synstrength, synslope;

    /* Set parameters for double-exponential adaptation (based on fiber type / spont) */
    if (spont == 100) {
        cf_factor = __min(800, pow(10, 0.29*cf/1e3 + 0.7));
    } else if (spont == 4) {
        cf_factor = __min(50, 2.5e-4*cf*4 + 0.2);
    } else {
        cf_factor = __min(1.0, 2.5e-4*cf*0.1 + 0.15);
    }
    
    PImax  = 0.6;                /* PI2 : Maximum of the PI(PI at steady state) */
    Asp = spont*3.0;             /* Spontaneous Firing Rate */
    TauR   = 2e-3;               /* Rapid Time Constant eq.10 */
    TauST  = 60e-3;              /* Short Time Constant eq.10 */
    Ar_Ast = 6;                  /* Ratio of Ar/Ast */
    PTS    = 3;                  /* Peak to Steady State Ratio, characteristic of PSTH */
    k1     = -1/TauR;            /* eq.8 & eq.10 */
    k2     = -1/TauST;           /* eq.8 & eq.10 */

    kslope = (1+50.0)/(5+50.0)*cf_factor*20.0*PImax;
    Ass = 800*(1+cf/100e3);  /* Steady State Firing Rate eq.10 */

    Aon = PTS*Ass;                            /* Onset rate = Ass+Ar+Ast eq.10 */
    AR = (Aon-Ass)*Ar_Ast/(1+Ar_Ast);      /* Rapid component magnitude: eq.10 */
    AST = Aon-Ass-AR;                   /* Short time component: eq.10 */
    Prest = PImax/Aon*Asp;                    /* eq.A15 */
    CG = (Asp*(Aon-Asp))/(Aon*Prest*(1-Asp/Ass));    /* eq.A16 */
    gamma1 = CG/Asp;                          /* eq.A19 */
    gamma2 = CG/Ass;                       /* eq.A20 */
    /* below: eq.A21 & eq.A22 */
    VI0    = (1-PImax/Prest) / 
        (gamma1*(AR*(k1-k2)/CG/PImax+k2/Prest/gamma1-k2/PImax/gamma2));
    VI1    = (1-PImax/Prest) / 
        (gamma1*(AST*(k2-k1)/CG/PImax+k1/Prest/gamma1-k1/PImax/gamma2));
    VI  = (VI0+VI1)/2;
    alpha  = gamma2/k1/k2;                    /* eq.A23,eq.A24 or eq.7 */
    beta   = -(k1+k2)*alpha;                  /* eq.A23 or eq.7 */
    theta1 = alpha*PImax/VI;
    theta2 = VI/PImax;
    theta3 = gamma2-1/PImax;

    PL = ((beta-theta2*theta3)/theta1-1)*PImax;  /* eq.4' */
    PG = 1/(theta3-1/PL);                 /* eq.5' */
    VL = theta1*PL*PG;                 /* eq.3' */
    CI = Asp/Prest;                          /* CI at rest, from eq.A3,eq.A12 */
    CL = CI*(Prest+PL)/PL;          /* CL at rest, from eq.1 */

    if (kslope >= 0) vsat = kslope+Prest;
    tmpst = log(2)*vsat/Prest;
    if(tmpst<400) synstrength = log(exp(tmpst)-1);
    else synstrength = tmpst;
    synslope = Prest/log(2)*synstrength;

    /* Outputs */
    (*synstrength_out) = synstrength;
    (*synslope_out) = synslope;
    (*VI_out) = VI;
    (*PG_out) = PG;
    (*VL_out) = VL;
    (*PL_out) = PL;
    (*CG_out) = CG;
    (*CI_out) = CI;
    (*CL_out) = CL;
}

/**
 * apply_ws1988_adaptation
 * 
 * Applies W&S three-store adaptation to vector of data sample by sample
 * 
 * Calculates outputs given inputs, paramters, and state variables for Westerman and Smith
 * (1988) three-store diffusion model of auditory-nerve synaptic adaptation.
 * 
 * In this model, neurotransmitter material that is present in the synaptic cleft (and thus
 * driving auditory-nerve spiking activity) is assumed to be the end result of a multistage 
 * process of chemcical diffusion between several store of neurotransmitter material. These
 * stages are the global stage (containing a constant concentration of material, 
 * independent of stimulus intensity and time) and the local and immediate stages 
 * (with concentrations that may vary with stimulus intensity and time). The permeabilities
 * between stages and the volumes in stages are assumed to be functions of stimulus 
 * intensity. The variable names are as follows:
 * C_g - concentration global
 * C_l - concentration local
 * C_i - concentration global -> local
 * P_g - permeability global
 * P_l - permability local -> immediate
 * P_i - permeability immediate -> cleft?
 * V_l - volume local
 * V_i - volume immediate
 * 
 * @param x Input data (nominally IHC potentials, a.u.)
 * @param n Current sample index
 * @param synstrength ???
 * @param synslope ???
 * @param VI Volume of the immediate store/synaptic cleft
 * @param PG Permeability from the global store to the local store
 * @param VL Volume of the local store
 * @param PL Permeability from local store to immediate store/synaptic cleft
 * @param CG Global concentration 
 * @param CI_state State variable holding current value of CI, the concentration in the 
 *   immediate store/synaptic cleft
 * @param CL_state State variable holding current value of CL, the concentration in the 
 *   local store
 * @param CIlast_state State variable holding previous value of CI
 * @param y Output vector 
 */
void apply_ws1988_adaptation(double *x, int n, double synstrength, double synslope, 
                             double VI, double PG, double VL, double PL, double CG, 
                             double tdres, double *CI_state, double *CL_state, 
                             double *CIlast_state, double *y) {
    /* Declare variables */
    double tmp, PPI, CI, CL;

    /* Compute ??? */
    tmp = synstrength*x[n];
    if (tmp < 400) tmp = log(1 + exp(tmp));
    PPI = synslope/synstrength*tmp;

    /* Set state */
    (*CIlast_state) = (*CI_state);
    CI = (*CI_state);
    CL = (*CL_state);

    /* Compute ???*/
    CI = CI + (tdres/VI)*(-PPI*CI + PL*(CL-CI));
    CL = CL + (tdres/VL)*(-PL*(CL-(*CIlast_state)) + PG*(CG-CL));
    if (CI < 0) {
        CI = CG/(PPI*(1/PG + 1/PL + 1/PPI));
        CL = CI*(PPI+PL)/PL;
    }

    /* Store state and output */
    (*CI_state) = CI;
    (*CL_state) = CL;
    y[n] = CI*PPI;
}

/**
 * apply_powerlaw_adaptation
 * 
 * Applies slow and fast powerlaw adaptation to signal.
 * 
 * @param x Input data 
 * @param randNums Input fractional Gaussian noise
 * @param I1 Pointer to state variable for path #1
 * @param I2 Pointer to state variable for path #1
 * @param n Current sample index
 * @param alpha1
 * @param beta1
 * @param alpha2
 * @param beta2
 * @param tdres
 * @param sout1 Output vector for path #1
 * @param sout2 Output vector for path #2
 */
void apply_powerlaw_adaptation(double *x, double *randNums, double *I1, double *I2, int n,
                               double alpha1, double beta1, double alpha2, 
                               double beta2, double tdres, double *sout1, double *sout2) {
    sout1[n]  = __max(0, x[n] + randNums[n] - alpha1*(*I1));
    sout2[n] = __max(0, x[n] - alpha2*(*I2));
    (*I1) = 0.0; (*I2) = 0.0;
    for (int j = 0; j < n+1; ++j) {
        (*I1) += (sout1[j])*tdres/((n-j)*tdres + beta1);
        (*I2) += (sout2[j])*tdres/((n-j)*tdres + beta2);
    }
}

/**
 * apply_powerlaw_adaptation_iir
 * 
 * Applies slow and fast powerlaw adaptation to signal using fast IIR approximation
 * 
 * @param x Input data 
 * @param randNums Input fractional Gaussian noise
 * @param I1 Pointer to integrator value state variable for path #1
 * @param I2 Pointer to integrator value state variable for path #2
 * @param E1 Pointer to parallel exponential process state variables for path #1
 * @param E2 Pointer to parallel exponential process state variables for path #2
 * @param W1 Pointer to parallel exponential weights for path #1
 * @param W2 Pointer to parallel exponential weights for path #2
 * @param d Pointer to decay-coefficient vector, size (n_process, )
 * @param n_process Number of parallel exponential processes used to approximate PLA
 * @param n Current sample index
 * @param alpha1
 * @param alpha2
 * @param tdres
 * @param sout1 Output vector for path #1
 * @param sout2 Output vector for path #2
 */
void apply_powerlaw_adaptation_iir(double *x, double *randNums, double *I1, double *I2, 
                               double *E1, double *E2, double *D1, double *D2, 
                               double *W1, double *W2, int n_process, int n,
                               double alpha1, double alpha2, 
                               double tdres, double *sout1, double *sout2) {
    // Apply power-law adaptation
    sout1[n]  = __max(0, x[n] + randNums[n] - alpha1*(*I1));
    sout2[n] = __max(0, x[n] - alpha2*(*I2));

    // Update values for I1/I2 based on approximation via parallel exponential processes
    (*I1) = 0.0; (*I2) = 0.0;
    for (int i = 0; i < n_process; i++) {
        if (n == 0) {
            E1[i] = W1[i] * sout1[n];
            E2[i] = W2[i] * sout2[n];
        } else {
            E1[i] = W1[i] * sout1[n] + (1-D1[i]) * E1[i];
            E2[i] = W2[i] * sout2[n] + (1-D2[i]) * E2[i];
        }
        (*I1) += E1[i];
        (*I2) += E2[i];
    }
}