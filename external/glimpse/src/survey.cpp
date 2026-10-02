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

#include <CCfits/CCfits>
#include "survey.h"


using namespace std;
using namespace CCfits;

survey::survey(boost::property_tree::ptree config)
{
    string coordinates_unit_str;

    // Retrieving general configuration
    coordinates_unit_str = config.get("survey.units", "radian");

    convert_coordinates_unit = 1.0;
    if (coordinates_unit_str.find("radian") != string::npos) {
        convert_coordinates_unit = 1.0;
    } else if (coordinates_unit_str.find("arcsec") != string::npos) {
        convert_coordinates_unit = M_PI / 180.0 / 60.0 / 60.0;
    } else if (coordinates_unit_str.find("arcmin") != string::npos) {
        convert_coordinates_unit = M_PI / 180.0 / 60.0;
    } else if (coordinates_unit_str.find("degree") != string::npos) {
        convert_coordinates_unit = M_PI / 180.0;
    } else {
        std::cout << "Unknown coordinates units, assuming radians." << std::endl;
    }
    center_ra  = config.get<double>("survey.center_ra")  * convert_coordinates_unit;
    center_dec = config.get<double>("survey.center_dec") * convert_coordinates_unit;
    size       = config.get<double>("survey.size")       * convert_coordinates_unit;

    // Retrieving FITS column names, in case they differ from standards
    hdu_number           = config.get<int>("survey.hdu",        1);
    column_map[RA]       = config.get<string>("survey.ra",      "ra");
    column_map[DEC]      = config.get<string>("survey.dec",     "dec");
    column_map[E1]       = config.get<string>("survey.e1",      "e1");
    column_map[E2]       = config.get<string>("survey.e2",      "e2");
    column_map[F1]       = config.get<string>("survey.f1",      "f1");
    column_map[F2]       = config.get<string>("survey.f2",      "f2");
    column_map[W_E]      = config.get<string>("survey.w_e",     "w");
    column_map[W_F]      = config.get<string>("survey.w_f",     "w_f");
    column_map[MAG]      = config.get<string>("survey.mag",     "mag");
    column_map[Z]        = config.get<string>("survey.z",       "z");
    column_map[ZSIG_MIN] = config.get<string>("survey.zsig_min", "zsig_min");
    column_map[ZSIG_MAX] = config.get<string>("survey.zsig_max", "zsig_max");
    column_map[PZ]       = config.get<string>("survey.pz",      "pz");
    
    flexion_available = false;
    
    flip_e2 = config.get<bool>("survey.flip_e2", false);
}

survey::~survey()
{
    // TODO: Deallocate arrays
}

void survey::load(string fileName)
{
    std::vector<string> hdus;
    std::valarray < double > ra, dec, e1, e2, f1, f2, w_e, w_f;
    std::valarray < double > z, zsig, zsig_min, zsig_max, mag;
    std::vector< std::valarray < double > > pz;
    std::valarray < double > zsamp;
    redshift_types ztype;
    bool has_flexion = false;
    bool has_weights = false;

    // Reading the data file
    std::auto_ptr<FITS> pInfile(new FITS(fileName, Read));
    
    ExtHDU &table = pInfile->extension(hdu_number);

    // Identify the type of data contained in this HDU
    
    // Check coordinates
    if (table.column().find(column_map[RA]) == table.column().end() ||
        table.column().find(column_map[DEC]) == table.column().end()) {
        std::cout << "Could not find the coordinates columns in data file" << std::endl;
        exit(-1);
    }

    // Check for ellipticity
    if (table.column().find(column_map[E1]) == table.column().end() ||
        table.column().find(column_map[E2]) == table.column().end()) {
        std::cout << "Could not find ellipticity measurements in data file" << std::endl;
        exit(-1);
    }

    // Check for flexion
    if (table.column().find(column_map[F1]) != table.column().end() && 
        table.column().find(column_map[F2]) != table.column().end()) {
        has_flexion = true;
	flexion_available = true;
    }

    // Check for weights
    if (table.column().find(column_map[W_E]) != table.column().end() ||
        table.column().find(column_map[W_F]) != table.column().end()) {
        has_weights = true;
    }

    // Check for different redshift information
    if (table.column().find(column_map[PZ])             != table.column().end()) {
        ztype = DISTRIBUTION;
    } else if (table.column().find(column_map[Z])        != table.column().end() &&
               table.column().find(column_map[ZSIG_MIN]) == table.column().end() &&
               table.column().find(column_map[ZSIG_MAX]) == table.column().end()) {
        ztype = DIRAC;
    } else if (table.column().find(column_map[Z])        != table.column().end() &&
               table.column().find(column_map[ZSIG_MIN]) != table.column().end() &&
               table.column().find(column_map[ZSIG_MAX]) != table.column().end()) {
        ztype = GAUSSIAN;
    } else {
        std::cout << "Could not find redshift information in data file" << std::endl;
        exit(-1);
    }

    long nrows = table.column(column_map[RA]).rows();

    // Reading data from file
    table.column(column_map[RA]).read(ra, 1, nrows);
    table.column(column_map[DEC]).read(dec, 1, nrows);
    table.column(column_map[E1]).read(e1, 1, nrows);
    table.column(column_map[E2]).read(e2, 1, nrows);
    if (has_weights) {
        table.column(column_map[W_E]).read(w_e, 1, nrows);
    }

    if (has_flexion) {
        table.column(column_map[F1]).read(f1, 1, nrows);
        table.column(column_map[F2]).read(f2, 1, nrows);
        if (has_weights) {
            table.column(column_map[W_F]).read(w_f, 1, nrows);
        }
    }

    if (ztype == DISTRIBUTION) {
        table.column(column_map[PZ]).readArrays(pz, 1, nrows);
    }
    if (ztype == DIRAC || ztype == GAUSSIAN) {
        table.column(column_map[Z]).read(z, 1, nrows);
    }
    if (ztype == GAUSSIAN) {
        table.column(column_map[ZSIG_MIN]).read(zsig_min, 1, nrows);
        table.column(column_map[ZSIG_MAX]).read(zsig_max, 1, nrows);
    }

    for (long i = 0 ; i < nrows; i++) {
        double ra_rad  = ra[i]  * convert_coordinates_unit;
        double dec_rad = dec[i] * convert_coordinates_unit;

	// Test if galaxy falls within the survey footprint, otherwise skip it
	if( ((ra_rad  - center_ra)  > size/2 ) &&
	    ((dec_rad - center_dec) > size/2 ))
	    continue;

        shape_measurement *shape = new shape_measurement;
        shape->pos[0]  = ra_rad;
        shape->pos[1] = dec_rad;
        shape->e[0]  = e1[i];
        shape->e[1]  = flip_e2 ? - e2[i] : e2[i];
        if (has_weights) {
            shape->w_e = w_e[i];
        }else
	    shape->w_e = 1.0;

        if (has_flexion) {
            shape->f[0] = f1[i];
            shape->f[1] = f2[i];
            if (has_weights) {
                shape->w_f = w_f[i];
            }else
		shape->w_f = 1.0;
        }else{
	    shape->f[0] = 0;
	    shape->f[1] = 0;
	    shape->w_f  = 0.0;
	}

        redshift_distribution *redshift;
        if (ztype == DIRAC) {
            redshift = new spectroscopic_redshift(z[i]);
        }
        if (ztype == GAUSSIAN) {
            redshift = new photometric_redshift(z[i], zsig_min[i], zsig_max[i]);
        }
        if (ztype == DISTRIBUTION) {
            redshift = new pdf_redshift(zsamp, pz[i]);
        }

        shape_catalogue.push_back(std::make_pair(shape, redshift));
    }

    std::cout << "Successfully loaded " << shape_catalogue.size() << " galaxies." << std::endl;
}


