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
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/program_options.hpp>
#include <sparse2d/IM_IO.h>

#include "survey.h"
#include "field.h"
#include "surface_reconstruction.h"


namespace po = boost::program_options;

int main(int argc, char *argv[])
{

    gsl_rng_env_setup();
    boost::property_tree::ptree pt;

    // Read command line arguments
    po::options_description desc("Allowed options");
    desc.add_options()
    ("config", po::value< std::string >()->required(), "configuration file")
    ("data",  po::value< std::string >()->required(), "survey data file")
    ("output", po::value< std::string >()->required(), "output file");

    po::positional_options_description positionalOptions;
    positionalOptions.add("config", 1);
    positionalOptions.add("data", 1);
    positionalOptions.add("output", 1);
    po::variables_map vm;

    try {
        po::store(po::command_line_parser(argc, argv)
                  .options(desc)
                  .positional(positionalOptions)
                  .run(), vm);
        po::notify(vm);
    } catch (po::error &e) {
        std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
        std::cerr << desc << std::endl;
        return 1;
    }

    // Load the configuration file
    try {
        boost::property_tree::ini_parser::read_ini(vm["config"].as<std::string>(), pt);
    } catch (boost::property_tree::ini_parser_error e) {
        std::cout << e.what() << std::endl;
        exit(-1);
    }

    // Create survey object and load data
    survey *surv = new survey(pt);
    surv->load(vm["data"].as<std::string>());

    // Initialize lensing field
    field *f = new field(pt, surv);
    
    // Initialize reconstruction object
    surface_reconstruction rec(pt, f);
    
    rec.reconstruct();
    
    // Extracts the reconstructed array
    dblarray kappa(f->get_npix(),f->get_npix());
    rec.get_convergence_map(kappa.buffer());
    fits_write_dblarr(vm["output"].as<std::string>().c_str(), kappa);
    
    delete f;
    delete surv;
    
    return 0;
}