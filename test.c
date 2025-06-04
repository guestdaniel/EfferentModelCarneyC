#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

int main() {
    // Check time
    clock_t start_time = clock();

    // Print progress
    printf("Testing initializing...\n");

    // Set params
    int totalstim = 500000;
    int n_chan = 1;

    // Allocate matrices and arrays
    double *px = calloc(totalstim, sizeof(double));
    for (int i = 0; i < totalstim; i++) {
        px[i] = pow(10.0, 60.0 / 20.0) * 20e-6 * sin(2*3.14159*1e3 * (float) i/100e3);
    }

    double *cf = calloc(n_chan, sizeof(double));
    cf[0] = 1e3;

    double *cohc = calloc(n_chan, sizeof(double));
    cohc[0] = 1.0;

    double *cihc = calloc(n_chan, sizeof(double));
    cihc[0] = 1.0;

    double *moc_weight = calloc(n_chan, sizeof(double));
    moc_weight[0] = 10.0;

    double **ffGn_hsr = malloc(n_chan * sizeof(double*));
    ffGn_hsr[0] = calloc(totalstim, sizeof(double));

    double **ffGn_lsr = malloc(n_chan * sizeof(double*));
    ffGn_lsr[0] = calloc(totalstim, sizeof(double));

    double **controlout;
    controlout = malloc(n_chan * sizeof(double*));
    controlout[0] = calloc(totalstim, sizeof(double));

    double **c1out;
    c1out = malloc(n_chan * sizeof(double*));
    c1out[0] = calloc(totalstim, sizeof(double));

    double **c2out;
    c2out = malloc(n_chan * sizeof(double*));
    c2out[0] = calloc(totalstim, sizeof(double));

    double **ihcout;
    ihcout = malloc(n_chan * sizeof(double*));
    ihcout[0] = calloc(totalstim, sizeof(double));

    double **expout_hsr;
    expout_hsr = malloc(n_chan * sizeof(double*));
    expout_hsr[0] = calloc(totalstim, sizeof(double));

    double **sout1_hsr;
    sout1_hsr = malloc(n_chan * sizeof(double*));
    sout1_hsr[0] = calloc(totalstim, sizeof(double));

    double **sout2_hsr;
    sout2_hsr = malloc(n_chan * sizeof(double*));
    sout2_hsr[0] = calloc(totalstim, sizeof(double));

    double **synout_hsr;
    synout_hsr = malloc(n_chan * sizeof(double*));
    synout_hsr[0] = calloc(totalstim, sizeof(double));

    double **expout_lsr;
    expout_lsr = malloc(n_chan * sizeof(double*));
    expout_lsr[0] = calloc(totalstim, sizeof(double));

    double **sout1_lsr;
    sout1_lsr = malloc(n_chan * sizeof(double*));
    sout1_lsr[0] = calloc(totalstim, sizeof(double));

    double **sout2_lsr;
    sout2_lsr = malloc(n_chan * sizeof(double*));
    sout2_lsr[0] = calloc(totalstim, sizeof(double));

    double **synout_lsr;
    synout_lsr = malloc(n_chan * sizeof(double*));
    synout_lsr[0] = calloc(totalstim, sizeof(double));

    double **anrateout_hsr;
    anrateout_hsr = malloc(n_chan * sizeof(double*));
    anrateout_hsr[0] = calloc(totalstim, sizeof(double));

    double **anrateout_lsr;
    anrateout_lsr = malloc(n_chan * sizeof(double*));
    anrateout_lsr[0] = calloc(totalstim, sizeof(double));

    double **cnout;
    cnout = malloc(n_chan * sizeof(double*));
    cnout[0] = calloc(totalstim, sizeof(double));

    double **icout;
    icout = malloc(n_chan * sizeof(double*));
    icout[0] = calloc(totalstim, sizeof(double));

    double **mocwdr;
    mocwdr = malloc(n_chan * sizeof(double*));
    mocwdr[0] = calloc(totalstim, sizeof(double));

    double **mocic;
    mocic = malloc(n_chan * sizeof(double*));
    mocic[0] = calloc(totalstim, sizeof(double));

    double **gain;
    gain = malloc(n_chan * sizeof(double*));
    gain[0] = calloc(totalstim, sizeof(double));

    double **gainpostmix;
    gainpostmix = malloc(n_chan * sizeof(double*));
    gainpostmix[0] = calloc(totalstim, sizeof(double));

    model(
        px,
        ffGn_hsr,
        ffGn_lsr,
        cf,
        (int) 1,
        (double) 1/100e3,
        totalstim,
        cohc,
        cihc,
        (int) 1,
        (double) 100.0,
        (int) 2,
        (double) 1e-3,
        (double) 1e-3,
        (double) 1e-3,
        (double) 1.0,
        (double) 1.0,
        (double) 1e-3,
        (double) 1e-3,
        (double) 1e-3,
        (double) 1.0,
        (double) 1.0,
        (double) 0.64,
        (double) 0.015,
        (double) 1.0,
        (double) 0.0,
        (double) 1.0,
        moc_weight,
        (double) 1.0,
        (double) 0.1,
        (double) 0.025,
        (int) 0,
        controlout,
        c1out,
        c2out,
        ihcout,
        expout_hsr,
        sout1_hsr,
        sout2_hsr,
        synout_hsr,
        expout_lsr,
        sout1_lsr,
        sout2_lsr,
        synout_lsr,
        anrateout_hsr,
        anrateout_lsr,
        cnout,
        icout,
        mocwdr,
        mocic,
        gain,
        gainpostmix
    );

    printf("Made it out of the function!\n");

    // Print a few random values from an output to make sure it's real
    printf("Val: %f\n", gain[0][40000]);

    // Free single-dimensional arrays
    free(px);
    free(cf);
    free(cohc);
    free(cihc);
    free(moc_weight);

    // Free two-dimensional arrays
    for (int i = 0; i < n_chan; i++) {
        free(ffGn_hsr[i]);
        free(ffGn_lsr[i]);
        free(controlout[i]);
        free(c1out[i]);
        free(c2out[i]);
        free(ihcout[i]);
        free(expout_hsr[i]);
        free(sout1_hsr[i]);
        free(sout2_hsr[i]);
        free(synout_hsr[i]);
        free(expout_lsr[i]);
        free(sout1_lsr[i]);
        free(sout2_lsr[i]);
        free(synout_lsr[i]);
        free(anrateout_hsr[i]);
        free(anrateout_lsr[i]);
        free(cnout[i]);
        free(icout[i]);
        free(mocwdr[i]);
        free(mocic[i]);
        free(gain[i]);
        free(gainpostmix[i]);
    }

    // Free the pointers to the rows of the two-dimensional arrays
    free(ffGn_hsr);
    free(ffGn_lsr);
    free(controlout);
    free(c1out);
    free(c2out);
    free(ihcout);
    free(expout_hsr);
    free(sout1_hsr);
    free(sout2_hsr);
    free(synout_hsr);
    free(expout_lsr);
    free(sout1_lsr);
    free(sout2_lsr);
    free(synout_lsr);
    free(anrateout_hsr);
    free(anrateout_lsr);
    free(cnout);
    free(icout);
    free(mocwdr);
    free(mocic);
    free(gain);
    free(gainpostmix);

    // Print progress
    printf("... Testing complete!\n");

    // Check time
    clock_t stop_time = clock();
    double elapsed_time = (double)(stop_time - start_time) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.6f seconds\n", elapsed_time);
}