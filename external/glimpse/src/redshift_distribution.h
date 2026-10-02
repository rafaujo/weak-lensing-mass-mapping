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

#ifndef REDSHIFT_DISTRIBUTION_H
#define REDSHIFT_DISTRIBUTION_H

#include <cmath>
#include <valarray>
#include <gsl/gsl_interp.h>

#define EPSILON_SPECTRO 1e-3

/*! Base class storing individual redshift information. 
 * 
 * Stores available redshift information and provides an estimate of the
 * probability density function.
 * 
 */
class redshift_distribution
{
protected:
  double normalizationFactor;	/*!< Normalisation factor of the PDF */
  
public:
  redshift_distribution() : normalizationFactor(1) { } 

  /*! Normalised redshift probability density function */
  virtual double pdf(double z) = 0;
};


/*! Spectroscopic redshift measurement
 * 
 */
class spectroscopic_redshift: public redshift_distribution
{
  double zSpec;
  
public:
  spectroscopic_redshift(double zSpec) : zSpec(zSpec) { }
  
  double get_redshift(){ return zSpec; }
  
  double pdf(double z){ return fabs(z - zSpec) <= EPSILON_SPECTRO ? 1.0 : 0.0; }
};


/*! Photometric redshift measurement
 * 
 */
class photometric_redshift: public redshift_distribution
{
  double zPhot;
  double zSigMin;
  double zSigMax;

public:
  
  photometric_redshift(double zPhot, double zSigMin, double zSigMax):
  zPhot(zPhot), zSigMin(zSigMin), zSigMax(zSigMax)
  {
    // Normalisation of the asymetric Gaussian PDF between 0 and infty
    normalizationFactor = 1.0/( 0.5 *sqrt(2.0 * M_PI)* (zSigMin * erf(zPhot/(sqrt(2.0) * zSigMin)) + zSigMax));
  }
  
  double pdf(double z){
      return normalizationFactor * exp( - 0.5 * pow(z - zPhot, 2.0)/pow( z < zPhot ? zSigMin : zSigMax,2.0));
  }
  
};


/*! Global redshift distribution
 * 
 * When no individual redshift measurements are available, the PDF 
 * is computed using a global Smail distribution of the type:
 * pdf(z) = z^a exp( - (z/z0)^b) 
 */
class smail_redshift : public redshift_distribution
{
  double a;
  double b;
  double z0;
  
public:
  
  smail_redshift(double z0, double a, double b):
  z0(z0), a(a), b(b)
  {
    //Normalisation of Smail distribution between 0 and infty
    normalizationFactor = 1.0 / ( 1./b * pow(z0, 1.0 + a)  * tgamma((1+a)/b) );
  }
  
  double pdf(double z){
      return normalizationFactor * pow(z, a) * exp( - pow(z/z0, b));
  }
};


/*! Full redshift pdf
 * 
 * 
 */
class pdf_redshift : public redshift_distribution
{
  gsl_interp *interpolator;
  gsl_interp_accel *accelerator;
  double *x;
  double *y;
  size_t nPoints;
  
public:
  pdf_redshift(std::valarray<double> &zSampling, std::valarray<double> &pdz);
  ~pdf_redshift();
  
  double pdf(double z){
    return normalizationFactor*gsl_interp_eval(interpolator, x, y, z, accelerator);
  }

};

#endif // REDSHIFT_MEASUREMENT_H
