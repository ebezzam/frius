# Sampling at the rate of innovation for ultrasound imaging and localization (EPFL Master Thesis 2018)

Author: [Eric Bezzam](https://ebezzam.github.io/)  
Supervisors: [Adrien Besson](https://adribesson.github.io/)<sup>1</sup>, 
[Hanjie Pan](https://lcav.epfl.ch/people/hanjie.pan),<sup>2</sup>
[Dimitris Perdios](https://people.epfl.ch/dimitris.perdios)<sup>1</sup>, 
[Prof. Jean-Philippe Thiran](https://lts5www.epfl.ch/thiran.html)<sup>1</sup>

<sup>1</sup>Signal Processing Laboratory 5([LTS5](https://lts5www.epfl.ch/)) at [EPFL](http://www.epfl.ch).  
<sup>2</sup>Audiovisual Communications Laboratory ([LCAV](http://lcav.epfl.ch)) at [EPFL](http://www.epfl.ch).


Questions/comments can be directed to: `eric[dot]bezzam[at]epfl.ch`

Or feel free to create an issue up top!

## Summary

The initial goal of this Master thesis was to reduce the sampling and thus data rate necessary for medical ultrasound imaging.
To this end, we investigate how recent signal processing approach with regards to _Finite Rate of Innovation_ (FRI) and
_Euclidean Distance Matrices_ (EDM) can be applied to the task of ultrasound imaging and localization. 

For medical imaging, we identify a few practical limitations (e.g. sparsity under noise, zero-finding) that 
complicate the application of FRI techniques, in particular to achieve real-time imaging at sampling rates lower than 
the conventional approaches.

For the task of localization, we identify a suitable candidate for FRI and EDM techniques, most notably for the task of 
non-destructive evaluation (NDE) where we can identify a small amount of desired features at rates lower than 
conventional methods.

## Citation

If you use any code or results from here, please cite:

    E. Bezzam, (2018). "Sampling at the rate of innovation for ultrasound imaging and localization," 2018
    Unpublished Master’s thesis, EPFL, Lausanne, Switzerland.
     
    
## Software requirements

This software has been tested with a MacBook Pro running macOS Sierra (Version 10.12.6).

If you are using Anaconda, you can create the 
environment used in this project with the following command:

```
conda env create -f frius_env.yml
```

For activating the environment:
* Windows: `activate frius`
* macOS and Linux: `source activate frius`

If you are not using Anaconda, you can open the `'frius_env.yml'` file to 
see which libraries were used (but perhaps not necessary). The essentials are:

1. Python 3.
2. `numpy`, `scipy`, `matplotlib` for standard scientific computing and plotting.
3. `joblib` for parallelizing some of the tests.
4.  `h5py` for opening certain datasets.

## About this software

In `'notebooks'`, we provide a couple tutorials that we hope will help the interested 
reader and hacker understand the core topics in this thesis. If you are not opting
to download this repository, we recommend viewing the notebooks with 
[nbviewer](http://nbviewer.jupyter.org/) by entering the GitHub link to the 
corresponding notebook. 

In `'frius'` are the main utility functions for the work in this thesis:
1. `'us_utils'`: utilities for synthesizing ultrasound measurements and performing
delay-and-sum (DAS) beamforming.
2. `'fri_utils'`: utilities for performing pulse stream recovery and evaluating the
performance.
3. `'edm_utils'`: utilities for using EDMs to perform echo/time-of-flight matching.

In `'report_results'` are various scripts for reproducing figures in the report. See
below for more information.

Documentation is admittedly rough due to the short time frame of this project, but feel
free to contact the author with any questions/comments!

## Reproducing report figures (and modifying parameters)

All the figures in the report can be created by running:

```
python generate_figures.py
```

The PDF (and some PNG) files will be written to this directory.

For a particular figure, e.g. 'Figure 1.6', the image can be generated
(and with modified parameters) by running the corresponding script 
`'report_results/figXpX*.py'`, e.g. `'report_results/fig1p6_pulse_shape.py''`.

_Note: this is only true for those figures that were generated with Python, e.g.
simulations._

## License

The source code is released under the [MIT](https://opensource.org/licenses/MIT) license.



