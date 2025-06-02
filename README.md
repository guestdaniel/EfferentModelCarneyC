# Introduction
This is the C source code repository for the Guest, Farhadi, and Carney (2025) auditory-nerve model including a medial olivocochlear efferent pathway.
The paper describing this model is located at [[link]].
This code can be found online at [Github](https://github.com/guestdaniel/EfferentModelCarneyC).
This code is licensed under the GNU AGPLv3 license — see the license file `LICENSE` in the top-level folder of this code for more information.

# Model architecture
## Description
This model is a model that accepts parameters and a sound-pressure waveform as input and provides predictions of neural responses at several different stages of the auditory system as output.
Large parts of the model are based on model components and work from many prior publications (see below for full list; primarily Zilany et al., 2009, 2014 and Farhadi et al., 2023).

Previous models from this lineage (e.g., Heinz et al., 2001, Zilany et al., 2009) were single-channel models simulating responses up to the level of the auditory nerve.
This version of the model is multi-channel in nature and includes dynamic changes in cochlear gain based on feedback signals from a simple model of the medial olivocochlear (MOC) efferent pathway.
See the model parper for more detail.

The model is written in the C programming language.
This code and supporting documentation is intended to be used principally by developers and not by end-users.
Individuals who hope to use this code in their own work should instead utilize a suitable "wrapper" in a high-level programming language, such as those for Julia, Python, and MATLAB located at [GitHub](https://github.com/guestdaniel/EfferentModelCarneyWrappers).

## Interface
- Users can generate model responses using a single function, `model`. 
The function `model` accepts many inputs and returns many outputs.
These inputs are documented in more detail inside the docstring at the head of the `model` function in the source code.
- Generally, the first few inputs to `model` are vector- or matrix-valued inputs of temporal signals (e.g., time-pressure waveform, synaptic noise) sampled with the same temporal resolution at which the model is evaluated.
That is, these inputs are of size `(n_samp, )` or `(n_chan, n_samp)`.
- Most other inputs are scalar- or vector-valued inputs of size `(n_chan, )` governing the parameterization of the model.
- Remaining inputs are pointers to pre-allocated storage for outputs, keeping in convention with C style for function signatures.
These outputs are all of the size `(n_chan, n_samp)`.
- It is important to note that the function `model` *is* modifying, and is not a pure function (i.e., it has side effects).

## Standards
Certain standards are assumed throughout the model code; these are documented here for consistency.
- All floating-point numbers are assumed to be double-precision 64-bit floating point numbers (`double` data type in C).
- All integers are assumed to be 16-bit signed integers (`int` data type in C).
- Variable-length arrays (VLAs) are used in the C source code; therefore, users must use a C compiler that supports VLAs.
Such support is included in the C99 (1999) standard but is optional in the more recent C11 (2011) standard, and different compilers handle the issue in different ways.
- Except where otherwise noted, frequencies are specified in units of Hertz (Hz), durations are specified in units of seconds (s), and pressures (e.g., input sound pressure vs time) are specified in units of Pascals (Pa)
- Some vector-valued variables are dynamically allocated at the beginning of the model function using `calloc`; some are allocated using variable-length or static-length arrays.
- Higher-dimensional variables (e.g., 2D, 3D arrays) are allocated at the beginning of the model function using `calloc` and referenced as pointers of pointers.
Almost all such variables have a temporal dimension, and such variables are arranged to obey the row-major standards of C with the long/hot dimension being the last dimension.
- CFs are assumed to be arranged from left to right in ascending order and to be log-spaced.

### Input options/flags
 Most input parameters are numerical quantities that are directly used in computations, but a few integer-type parameters are instead used as switching parameters to provide one of several choices for a given behavior.
 Options that should be treated as defaults are bolded.
- `species`   
1 → cat  
**2 → human with Shera and Oxenham tuning**  
3 → human with Glasberg and Moore tuning
- `powerlaw_mode`  
1 → true powerlaw adaptation  
**2 → approximate powerlaw adaptation based on Guest and Carney (2024)**  
- `moc_fix_gain`  
**0 → allow gain-factor calculations to be dynamic**  
1 → freeze gain factor calculations 

# Files
- `adaptation.c` Source code for adaptation functions (double-exponential, powerlaw, etc.
- `changelog.txt` Text-format changelog documenting changes to the code over time, intended to supplement the Git version history
- `complex.c` Utility file defining a complex-valued number type and corresponding mathematical operations
- `model.c` Source code for the main logic of the model
- `sfie.c` An implementation of the "same-frequency inhibition-excitation" model of Carney and colleagues in C

# Notes

# Literature
The efferent model has been developed with a variety of sources in the scientific literature in mind. 

## Prior Zilany, Bruce, and Carney model papers
Bruce, I. C., Erfani, Y., and Zilany, M. S. A. (2018). “A phenomenological model of the synapse between the inner hair cell and auditory nerve: Implications of limited neurotransmitter release sites,” Hearing Research, 360, 40–54. doi:10.1016/j.heares.2017.12.016

Zilany, M. S. A., Bruce, I. C., and Carney, L. H. (2014). “Updated parameters and expanded simulation options for a model of the auditory periphery,” The Journal of the Acoustical Society of America, 135, 283–286. doi:10.1121/1.4837815

Zilany, M. S. A., Bruce, I. C., Nelson, P. C., and Carney, L. H. (2009). “A phenomenological model of the synapse between the inner hair cell and auditory nerve: Long-term adaptation with power-law dynamics,” The Journal of the Acoustical Society of America, 126, 2390–2412. doi:10.1121/1.3238250

## Efferent physiology
Kawase, T., Delgutte, B., and Liberman, M. C. (1993). “Antimasking effects of the olivocochlear reflex. II. Enhancement of auditory-nerve response to masked tones,” Journal of Neurophysiology, 70, 2533–2549. doi:10.1152/jn.1993.70.6.2533

Warren, E. H., and Liberman, M. C. (1989a). “Effects of contralateral sound on auditory-nerve responses. I. Contributions of cochlear efferents,” Hearing Research, 37, 89–104. doi:10.1016/0378-5955(89)90032-4

Warren, E. H., and Liberman, M. C. (1989b). “Effects of contralateral sound on auditory-nerve responses. II. Dependence on stimulus variables,” Hearing Research, 37, 105–121. doi:10.1016/0378-5955(89)90033-6

Guinan, J. J., and Gifford, M. L. (1988a). “Effects of electrical stimulation of efferent olivocochlear neurons on cat auditory-nerve fibers. I. Rate-level functions,” Hearing Research, 33, 97–113. doi:10.1016/0378-5955(88)90023-8

Guinan, J. J., and Gifford, M. L. (1988b). “Effects of electrical stimulation of efferent olivocochlear neurons on cat auditory-nerve fibers. II. Spontaneous rate,” Hearing Research, 33, 115–127. doi:10.1016/0378-5955(88)90024-X

Guinan, J. J., and Gifford, M. L. (1988c). “Effects of electrical stimulation of efferent olivocochlear neurons on cat auditory-nerve fibers. III. Tuning curves and thresholds at CF,” Hearing Research, 37, 29–45. doi:10.1016/0378-5955(88)90075-5

## Efferent anatomy
Brown, M. C. (2014). “Single-unit labeling of medial olivocochlear neurons: the cochlear frequency map for efferent axons,” Journal of Neurophysiology, 111, 2177–2186. doi:10.1152/jn.00045.2014

Liberman, M. C., Dodds, L. W., and Pierce, S. (1990). “Afferent and efferent innervation of the cat cochlea: Quantitative analysis with light and electron microscopy,” J of Comparative Neurology, 301, 443–460. doi:10.1002/cne.903010309


