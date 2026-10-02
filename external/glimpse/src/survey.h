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

#ifndef SURVEY_H
#define SURVEY_H

#include <vector>
#include <map>
#include <boost/property_tree/ptree.hpp>

#include "redshift_distribution.h"

/*! Class representing a lensing survey.
 * 
 * Stores all survey information and shear data for each galaxy.
 * 
 */
class survey
{
  // Utility structures and declarations
  
  /*! Structure storing shape measurements. */
  struct shape_measurement{
    float pos[2];			/*!< Position of the measurement in projected coordinate plane */
    float e[2];				/*!< Ellipticity */
    float f[2];				/*!< Flexion */
    float w_e;				/*!< Inverse variance weight for Ellipticity*/
    float w_f;				/*!< Inverse variance weight for Flexion*/
  };
  
  typedef enum {
    RA,
    DEC,
    E1,
    E2,
    F1, 
    F2,
    Z,
    PZ,
    MAG,
    ZSIG_MIN,
    ZSIG_MAX,
    W_E,
    W_F
  } data_fields;
  
  typedef enum { 
    DIRAC,
    GAUSSIAN,
    DISTRIBUTION
  } redshift_types;
  
  
  // Survey geometry
  double   size; 		      	/*!< Angular size of the field, in radians */
  double   center_ra;		      	/*!< Right Ascension of the center of the field, in radians */
  double   center_dec;		      	/*!< Declination of the center of the field, in radians */
  
  bool flexion_available;		/*!< Flag indicates whether flexion is available */
  bool flip_e2;                         /*!< Flag indicates if the second shear component should be flipped : e2' = -e2 */
  
  // Data format
  double convert_coordinates_unit;	/*!< Factor to convert the provided coordinates to radians */
  std::map<int,std::string> column_map;	/*!< Maps the fields to their columns in FITS files */
  int hdu_number;

  
  /*! Vectors storing the shapes and redshifts */
  std::vector<std::pair<shape_measurement *, redshift_distribution *> > shape_catalogue;
 
public:
  /*! Constructor from configuration file.
   * 
   * Creates the class based on the parameters specified in the configuration
   * file.
   * 
   * \param config: Boost property tree built from configuration file
   * \param filename: FITS file to load
   * 
   */
  survey(boost::property_tree::ptree config);
  
  /*! Destructor */
  ~survey();
  
   /*! Loads survey data from a file
    * If a file has already been loaded, adds the new data to the previously loaded data.
    * The format of the file to load is assumed to be the one specified for the survey.
    * 
    * \param fileName: Path to the data file.
    * 
    */
    void load(std::string fileName);

    /*! Returns the number of galaxies in the survey */
    long get_ngal() {
        return shape_catalogue.size();
    }
    
    /*! Returns flexion availability in the lensing survey */
    bool get_flexion_availability(){
	return flexion_available;
    }
    
    /*! Returns the angular size covered by the survey, in radians */
    double get_size() {
        return size;
    }

    /*! Returns the Right Ascension of the center of the survey field, in radians */
    double get_center_ra() {
        return center_ra;
    }

    /*! Returns the Declination of the center of the survey field, in radians */
    double get_center_dec() {
        return center_dec;
    }
    
    /*! Returns the first component of the shear of the specified galaxy
    *  \a gal_index   : index of the galaxy in the catalogue */
    double get_gamma1(long gal_index) {
        return shape_catalogue[gal_index].first->e[0];
    }

    /*! Returns the second component of the shear of the specified galaxy
     *  \a gal_index   : index of the galaxy in the catalogue */
    double get_gamma2(long gal_index) {
        return shape_catalogue[gal_index].first->e[1];
    }

    /*! Returns the first component of the shear of the specified galaxy
     *  \a gal_index   : index of the galaxy in the catalogue */
    double get_F1(long gal_index) {
        return shape_catalogue[gal_index].first->f[0];
    }

    /*! Returns the second component of the shear of the specified galaxy
     *  \a gal_index   : index of the galaxy in the catalogue */
    double get_F2(long gal_index) {
        return shape_catalogue[gal_index].first->f[1];
    }

    /*! Returns the Right Ascension of the specified galaxy
     *  \a gal_index   : index of the galaxy in the catalogue */
    double get_ra(long gal_index) {
        return shape_catalogue[gal_index].first->pos[0];
    }

    /*! Returns the Declination of the specified galaxy
     *  \a gal_index   : index of the galaxy in the catalogue */
    double get_dec(long gal_index) {
        return shape_catalogue[gal_index].first->pos[1];
    }
    
    /*! Returns the inverse variance shear weight of the specified galaxy
     *  \a gal_index   : index of the galaxy in the catalogue */
    double get_shear_weight(long gal_index) {
        return shape_catalogue[gal_index].first->w_e;
    }
    
    /*! Returns the inverse variance flexion weight of the specified galaxy
     *  \a gal_index   : index of the galaxy in the catalogue */
    double get_flexion_weight(long gal_index) {
        return shape_catalogue[gal_index].first->w_f;
    }
    
    
    /*! Returns the redshift distribution of the specified galaxy
     *  \a gal_index   : index of the galaxy in the catalogue */
    redshift_distribution *get_redshift(long gal_index){
        return shape_catalogue[gal_index].second;
    }
    
};

#endif // SURVEY_H
