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

#ifndef FIELD_H
#define FIELD_H

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <nfft3.h>
#include <nicaea/cosmo.h>

#include "survey.h"

// Reference redshift used to compute the 2D convergence maps
#define Z_INF 1000


/*! Class representing a lensing field in 2D or 3D.
 * 
 * Implements the transforms from and to convergence
 * 
 */
class field
{
  
  // Field geometry
  int      npix;		      	/*!< Number of pixels */
  double   pixel_size;		      	/*!< Size of the pixels, in radians */
  std::valarray<double> grid_ra;      	/*!< Right Ascension coordinates of the pixel centers, in radians */
  std::valarray<double> grid_dec;     	/*!< Declination coordinates of the pixel centers, in radians */
  int      padding_size;	      	/*!< Number of pixels included in the zero padding */
  double   convert_coordinates_unit;  	/*!< Factor to convert the provided coordinates in radians */

  int      nlp;			      	/*!< Number of lens planes */
  double   zlens;			/*!< Redshift of the lens if reconstructing a single lens plane */
  std::valarray<double> zlp_low;	/*!< Lower bound of redshift bin */
  std::valarray<double> zlp_up;    	/*!< Upper bound of redshift bin */
  
  // FFTs
  int NpixFFT;
  double fftFactor;
  nfft_plan** ps;
  fftw_complex* fft_frame;
  
  // Survey data
  long     ngal;                /*!< Number of galaxies */
  double * shear_gamma1;        /*!< 1D Array for storing the shear for each galaxy in the survey.*/
  double * shear_gamma2;        /*!< 1D Array for storing the shear for each galaxy in the survey.*/
  double * flexion_f1;          /*!< 1D Array for storing the first flexion for each galaxy in the survey.*/
  double * flexion_f2;          /*!< 1D Array for storing the first lexion for each galaxy in the survey.*/
  double * w_f; 
  double * w_e;
  double   sig_frac;            /*!< Ratio of variance between shear and flexion */
  bool     include_flexion;     /*!< Flag indicating whether flexion measurements are included in the reconstruction.*/
  
  // Auxiliary arrays
  double * res_gamma1;          /*!< Array storing the gamma residuals for each galaxy.*/
  double * res_gamma2;          /*!< Array storing the gamma residuals for each galaxy.*/
  double * res_f1;              /*!< Array storing the gamma residuals for each galaxy.*/
  double * res_f2;              /*!< Array storing the gamma residuals for each galaxy.*/
  double * res_conv;            /*!< Array storing the convergence for each galaxy.*/
  double * cov;                 /*!< Array to store the covariance matrix resulting from the reduced shear.*/
  double * lensKernel;          /*!< Array storing the lensing efficiency kernel for each galaxy.*/  
  
  survey *surv;                 /*!< Reference to the survey object */
  
  gsl_rng *rng;                 /*!< Random number generator */
  
  nicaea::error **err;          /*!< NICAEA error structure.*/
  nicaea::cosmo *model;         /*!< NICAEA cosmology used for the mapping.*/

  /*! Computes lensing kernel for each galaxy, to reconstruct a surface mass density
   * 
   */
  void compute_surface_lensing_kernel();
  
  /*! Computes lensing kernel for each galaxy, to reconstruct a 3D density contrast.
   * 
   */
  void compute_3D_lensing_kernel();

  /*! Computes the forward lensing transform from density to shear.
   * 
   */
  void forward_operator(fftw_complex *delta);
  
  /*! Compute the adjoint operation.
   * 
   */
  void adjoint_operator(fftw_complex *delta);
  
public:
  /*! Constructor from configuration file and survey
   * 
   */
  field(boost::property_tree::ptree config, survey *surv);

  /*! Destructor */
  ~field();
  
  /*! Return the number of pixels for a field of size npix x npix.
   * 
   */
  int get_npix() {
        return npix;
  }

  /*! Computes the gradient of the chi_2 for a given delta field.
   * 
   */
  void gradient(fftw_complex *delta);

  /*! Computes the gradient of the chi_2 for a given delta field and randomized measurements.
   * 
   */
  void gradient_noise(fftw_complex *delta);
  
  /*! Updates the non-linear correction factor in the covariance matrix.
   * 
   */
  void update_covariance(fftw_complex *kappa);

  /*! Computes the spectral norm of the lensing operator
   * 
   */
  double get_spectral_norm(int niter, double tol);

  /*! Checks that the adjoint_operator is indeed the adjoint of the forward operator.
   * 
   */
  bool check_adjoint();
  
  /*! Combines shear and flexion components applying a minimum variance filter
   * 
   */
  void combine_components(fftw_complex * delta, fftw_complex * delta_comb);
  
  /*! Copies the same combined field into shear and flexion commponents
   * 
   */
  void combine_components_inverse(fftw_complex * delta_comb, fftw_complex * delta);

};

#endif // FIELD_H
