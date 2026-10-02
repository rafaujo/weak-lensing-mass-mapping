# Glimpse

Glimpse is a sparsity based mass-mapping algorithm. See the
[Glimpse page](http://www.cosmostat.org/software/glimpse) on the
CosmoStat website for more information.

## Installation

### Requirements

Glimpse depends on the following software:

* **FFTW** version 3.3 or later
* **NFFT** version 3.2 or later
* **cfitsio** and **CCFits**
* **GSL**
* **CMake** version 2.8 or later

These dependencies can easily be installed using a package manager:

* Setting up requirements on **Linux**:
  Simply use the package manager provided by your Linux distribution, for instance on Ubuntu Linux:

    $ sudo apt-get install cmake libgsl0-dev libfftw3-3  libccfits-dev libnfft3-dev

* Setting up requirements on **MacOs X**:
  The preferred installation method for the dependencies is through [MacPorts](https://www.macports.org):
    
    $ sudo port install cmake libgsl0-dev pkgconfig gsl fftw-3 nfft-3
  
  CCFits needs to be installed manually, the sources can be found [here](http://heasarc.gsfc.nasa.gov/fitsio/ccfits/).

### Compilation

The Glimpse source can be retrieved from GitHub with the following command:

    $ git clone https://github.com/CosmoStat/Glimpse.git

This will clone the current development version of Glimpse into a local Glimpse folder. Once the requirements are installed, Glimpse can be compiled by running the  following commands:

    $ cd Glimpse
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make

This will create a Glimpse executable in the build directory.

## Usage

Glimpse expects input shear and/or flexion data as columns in a FITS file. A simple python script for converting .txt files to FITS is provided in the *utils* folder.

All the options of the reconstruction algorithm can be specified in a *config.ini* file such as the one provided in the *example* directory.

Glimpse can be run with the following command line:

    $ glimpse config.ini cat_3_0.fits kappa.fits

Where *kappa.fits* is the reconstructed convergence map (scaled for sources at infinite redshift) and *cat_3_0.fits* is the input data file.
