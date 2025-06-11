#include "complex.hpp"

/* Declare primary model() function 
 * Because this function has so many arguments, they are unrolled below, grouped with empty
 * lines, and annotated with the correspond argument names in the definition of model
 */
void model(
    // Principal inputs
    double *,   // px
    double **,  // randNums_hsr
    double **,  // randNums_lsr
    double *,   // cf
    // IHC/AN parameters
    int,        // n_chan
    double,     // tdres
    int,        // totalstim
    double *,   // cohc
    double *,   // cihc
    int,        // species
    double,     // spont
    int,        // powerlaw_mode
    // CN parameters
    double,     // cn_tau_e
    double,     // cn_tau_i
    double,     // cn_delay
    double,     // cn_amp
    double,     // cn_inh
    // IC parameters
    double,     // ic_tau_e
    double,     // ic_tau_i
    double,     // ic_delay
    double,     // ic_amp
    double,     // ic_inh
    // MOC parameters
    double,     // moc_cutoff
    double *,   // moc_beta
    double *,   // moc_offset
    double,     // moc_minval
    double,     // moc_maxval
    double *,   // moc_weight
    double,     // moc_width
    double,     // dur_settle
    double,     // moc_delay
    int,        // moc_fix_gain
    // BM/IHC outputs
    double **,  // controlout
    double **,  // c1out
    double **,  // c2out
    double **,  // ihcout
    // HSR outputs
    double **,  // expout_hsr
    double **,  // sout1_hsr
    double **,  // sout2_hsr
    double **,  // synout_hsr
    // LSR outputs
    double **,  // expout_lsr
    double **,  // sout1_lsr
    double **,  // sout2_lsr
    double **,  // synout_lsr
    // AN outputs
    double **,  // anrateout_hsr
    double **,  // anrateout_lsr
    // CN/IC outputs
    double **,  // cnout
    double **,  // icout
    // MOC outputs
    double **,  // mocwdr
    double **,  // mocic
    double **,  // gain
    double **   // gainpostmix
);

/* Declare other functions */
// model.c
double normal_pdf(double, double, double);
void middle_ear(double *, double, int, int, double *);
void Get_tauwb(int, double *, int, int, double *, double *);
void Get_taubm(int, double *, int, double *, double *, double *, double *);
double gain_groupdelay(double, double, double, double, int *);
double WbGammaTone(double, double, double, int, double, double, int, double *, 
                   COMPLEX *, COMPLEX *);
double Boltzman(double, double, double, double, double);
double NLafterohc(double, double, double, double);
double OhcLowPass(double, double, double, int, double, int, double *, double *);
double C1ChirpFilt(double, double,double, int, double, double, double *, double *, 
                   double **, double **);
double C2ChirpFilt(double, double,double, int, double, double, double *, double *, 
                   double **, double **);
double NLogarithm(double, double, double, double);
double IhcLowPass(double, double, double, int, double, int, double *, double *);
double delay_cat(double);
double delay_human(double);
void delay_signal(double *, int, int, double*);
void filter_lowpass_iir(double *, int, double, double *);
double moc_nonlinearity(double, double, double, double, double);
// adaptation.c
void initialize_ws1988_adaptation(double, double, double *, double *, double *,
                                  double *, double *, double *, double *, double *,
                                  double *);
void apply_ws1988_adaptation(double *, int, double, double, double, double, double,
                             double, double, double, double *, double *, double *,
                             double *);
void apply_powerlaw_adaptation(double *, double *, double *, double *, int, double,
                               double, double, double, double, double *, double *);
void apply_powerlaw_adaptation_iir(double *, double *, double *, double *, double *,
                               double *, double *, double *, double *, double *, int, int, 
                               double, double, double, double *, double *);
// sfie.c
void get_alpha_norm(double, double, double, double *, double *);
void filter_alpha(double *, int, double, double *, double *, double *);
double hw_rectify(double);