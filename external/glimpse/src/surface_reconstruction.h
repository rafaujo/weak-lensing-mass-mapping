/*! Copyright CEA, 2015-2016
 * author : Francois Lanusse < francois.lanusse@gmail.com >
 * 
 * This software is a computer program whose purpose is to reconstruct mass maps
 * from weak gravitational lensing.
 * 
 * This software is governed by the CeCILL license under French law and
 * abiding by the rules of distribution of free software.  You can  use, 
 * modify and/ or redistribute the software under the terms of the CeCILL
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info". 
 * 
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability. 
 * 
 * In this respect, the user's attention is drawn to the risks associated
 * with loading,  using,  modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean  that it is complicated to manipulate,  and  that  also
 * therefore means  that it is reserved for developers  and  experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or 
 * data to be ensured and,  more generally, to use and operate it in the 
 * same conditions as regards security. 
 * 
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
 * 
 */

#ifndef SURFACE_RECONSTRUCTION_H
#define SURFACE_RECONSTRUCTION_H

#include <boost/property_tree/ptree.hpp>

#include "field.h"
#include "wavelet_transform.h"

class surface_reconstruction
{
    // Reconstruction parameters
    int    nscales;                     /*!< Number of dyadic wavelet scales. */
    int    nRecIter;                    /*!< Number of main iterations. */
    double lambda;                      /*!< Regularisation parameter. */
    int    nreweights;                  /*!< Number of reweighted l1 iterations.*/
    bool   positivity;                  /*!< Apply positivity constraint on the reconstruction. */
    
    // Internal parameters
    int npix;                           /*!< Number of pixels. */
    int ncoeff;                         /*!< Number of reconstruction coefficients. */
    int nframes;                        /*!< Total number of wavelet frames. */
    int nwavcoeff;                      /*!< Number of wavelet coefficients. */
    int nrandom;                        /*!< Number of noise randomisations for building thresholds. */
    double fftFactor;                   /*!< Normalisation factor for the FFT. */
    double * sigma_thr;                 /*!< Array storing the regularisation parameter for each wavelet frame. */
    double mu1, mu2, sig, tau;          /*!< Hyper-parameters for the algorithm. */
    wavelet_transform *wav;             /*!< Wavelet transform operator. */
    field *f;                           /*!< Lensing field to use for the reconstruction.*/
    
    // Internal reconstruction arrays
    fftw_complex * kappa;
    fftw_complex * kappa_u;
    fftw_complex * kappa_rec;
    fftw_complex * kappa_old;
    fftw_complex * kappa_grad;
    fftw_complex * kappa_tmp;
    fftw_complex * kappa_trans;
    fftw_complex * fft_frame;
    double * alpha;                     
    double * alpha_u;
    double * alpha_res;
    double * alpha_tmp;
    double * thresholds;
    double * support;
    double * weights;
    
    fftw_plan plan_backward;
    fftw_plan plan_forward;

    double get_spectral_norm_prox(int niter, double tol);
    
public:
    
    /*! Initialise surface mass density reconstruction algorithm.
     * 
     */
    surface_reconstruction(boost::property_tree::ptree config, field *f);
    
    /*! Destructor.
     * 
     */
    ~surface_reconstruction();
    
    /*! Run the main iteration of the reconstruction algorithm for a number of iterations.
     * 
     */
    void run_main_iteration(long niter, bool debias=false);
    
    /*! Performs the reconstruction
     * 
     */
    void reconstruct();
    
    /*! Compute the noise thresholds.
     * 
     */
    void compute_thresholds(int niter);
    
    /*! Update weights for reweighted-l1 based on current solution.
     * 
     */
    void compute_weights();
    
    /*! Get current reconstructed convergence map
     * 
     */
    void get_convergence_map(double *kap);
    
};

#endif // SURFACE_RECONSTRUCTION_H
