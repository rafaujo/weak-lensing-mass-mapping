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

#include "redshift_distribution.h"


pdf_redshift::pdf_redshift(std::valarray< double >& zSampling, std::valarray< double >& pdz)
{
  
  // Allocate arrays to store pdf samples
  nPoints = zSampling.size();
  x = (double*) malloc(sizeof(double) * nPoints);
  y = (double*) malloc(sizeof(double) * nPoints);
  
  // Copy array data
  for(int i =0; i < nPoints; i++){
      x[i] = zSampling[i];
      y[i] = pdz[i];
  }
  
  interpolator = gsl_interp_alloc(gsl_interp_cspline, nPoints);
  accelerator = gsl_interp_accel_alloc();
  
  gsl_interp_init(interpolator, x, y, nPoints);
  
  // Compute normalization factor
  normalizationFactor = 1.0 / gsl_interp_eval_integ(interpolator, x, y, x[0], x[nPoints-1], accelerator);
}

pdf_redshift::~pdf_redshift()
{
  gsl_interp_accel_free(accelerator);
  gsl_interp_free(interpolator);
  free(x);
  free(y);
}
