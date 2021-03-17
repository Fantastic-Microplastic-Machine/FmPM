[![Build Status](https://travis-ci.com/Fantastic-Microplastic-Machine/FmPM.svg?branch=main)](https://travis-ci.com/github/Fantastic-Microplastic-Machine/FmPM)
[![Coverage Status](https://coveralls.io/repos/github/Fantastic-Microplastic-Machine/FmPM/badge.svg?branch=main)](https://coveralls.io/github/Fantastic-Microplastic-Machine/FmPM?branch=main)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

# FmPM - Fantastic μPlastic Machine
**FmPM** is a Data Science Project that was developed by a group of graduate students from the University of Washington in Seattle.

## Description
Microplastic pollution is an emerging global concern and there is an urgent need to understand the accumulation and transport of microplastics smaller than 1 mm. With the current rate of plastic consumption and the higher demands caused by the pandemic, it is urgent and critical to develop a computer vision and machine learning tool for quick and accurate analysis. One limitation in the field of microplastics is the ability to quickly and accurately categorize and label the large data sets of particles found in the environment. Currently, categorization and imaging are time-intensive and completed by human experts with ample experience in microplastic analysis. The understanding of physical features of microplastics can aid current computer vision programs in particle selection before chemical identification. With the application of machine learning techniques, it may be possible to identify specific features that are common in all microplastics and visually distinguish microplastics from non-microplastics. 

In this project, `FmPM` uses chemically analyzed and labeled Raman microscopy images of microparticles (microplastic and non-micropliastic particles) to develop a model that predicts whether a particle is microplastic or non-microplastic. 


## Getting started
### Clone the repository
```
https://github.com/Fantastic-Microplastic-Machine/FmPM.git
```

### Install the environment

```
conda env create -f environment.yml
```
### Load the environment
```
conda activate fmpm
```

### Using the code
1. Data Prep 
- All images in a single directory
- Pandas DataFrame with ‘isPlastic’ and ‘File’ columns (can use prep.py functions to help with this)

2. Load data into custom pytorch class tenX_dataset (prep.py)

3. Train the default model (construct.py- train();  specify dataset, epochs, and batch size)

4. Get predictions (construct.py- get_predictions())

Next Steps: Override defaults with custom options, k-fold cross validation (kfold.py), save/load models to file (construct.py)

See [examples.ipynb](https://github.com/Fantastic-Microplastic-Machine/FmPM/blob/main/examples.ipynb) for more details.

## Authors
Will Ballengee - Chemical engineering, MS

Nida Janulaitis - Chemical engineering, PhD

Samantha Phan - Chemistry, PhD

Liwen Xing - Molecular Engineering & Sciences, PhD


## License

This project is licensed under the MIT License - see `LICENSE` for details.
