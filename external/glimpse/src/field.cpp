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

#include <iostream>
#include <fstream>
#include <string>
#include <gsl/gsl_integration.h>

#include "field.h"

field::field(boost::property_tree::ptree config, survey *su)
{
    surv = su;
    // Reading configuration
    double Omega_m = config.get<double>("cosmology.Omega_m", 0.25);
    double h = config.get<double>("cosmology.h", 0.70);
    padding_size = config.get<int>("field.padding", 0);
    std::string coordinates_unit_str = config.get("field.units", "radian");
    convert_coordinates_unit = 1.0;
    if (coordinates_unit_str.find("radian") != std::string::npos) {
        convert_coordinates_unit = 1.0;
    } else if (coordinates_unit_str.find("arcsec") != std::string::npos) {
        convert_coordinates_unit = M_PI / 180.0 / 60.0 / 60.0;
    } else if (coordinates_unit_str.find("arcmin") != std::string::npos) {
        convert_coordinates_unit = M_PI / 180.0 / 60.0;
    } else if (coordinates_unit_str.find("degree") != std::string::npos) {
        convert_coordinates_unit = M_PI / 180.0;
    } else {
        std::cout << "Unknown coordinates units, assuming radians." << std::endl;
    }
    pixel_size = config.get<double>("field.pixel_size") * convert_coordinates_unit;

    // Load  the  redshift range of the reconstruction
    nlp   = config.get<double>("field.nlp",    1);
    if (nlp == 1) {
        zlens = config.get<double>("field.zlens", -1);
        if(zlens <= 0){
            std::cout << "Warning: No redshift specified for the lens, ignoring source redsfhits" << std::endl;
        }
    } else {
        zlp_low.resize(nlp);
        zlp_up.resize(nlp);
        double zlp_min = config.get<double>("field.zlp_min");
        double zlp_max = config.get<double>("field.zlp_max");
        // Creates the lensplanes, for now just regularly spaced between zmin and zmax
        for (int i = 0; i < nlp; i++) {
            zlp_low[i] = zlp_min + (zlp_max - zlp_min) * ((double) i) / ((double) nlp);
            zlp_up[i]  = zlp_min + (zlp_max - zlp_min) * ((double)(i + 1.0)) / ((double) nlp);
        }
        //For the last bin we go up to 10
        //TODO: Is it necessary ?
        zlp_up[nlp - 1] = 10;
    }

    // Here we increase the size of the field to avoid border effects
    double size       = surv->get_size();
    double center_ra  = surv->get_center_ra();
    double center_dec = surv->get_center_dec();
    npix = size / pixel_size;
    npix = npix + (npix % 2) + 2 * padding_size;
    size = npix * pixel_size;
    std::cout << "Number of pixels : " << npix << "Pixel size : " << pixel_size << std::endl;

    // Check whether flexion measurements are available
    include_flexion = config.get<bool>("field.include_flexion", false);
    if (include_flexion) {
        if (! surv->get_flexion_availability()) {
            std::cout << "No flexion measurements provided, reconstructing from shear alone" << std::endl;
            include_flexion = false;
        }
    }

    // Initialize the random number generator
    const gsl_rng_type *T;
    T = gsl_rng_default;
    rng = gsl_rng_alloc(T);

    // Initialize NICAEA
    err = new nicaea::error*;
    *err = NULL;
    model = nicaea::init_parameters(Omega_m, 1.0 - Omega_m, -1.0, 0.0, NULL, 0, h, 0.044, 0.0, 0.0, 0.80, 0.96,
                                    nicaea::smith03, nicaea::eisenhu, nicaea::growth_de, nicaea::linder,
                                    nicaea::norm_s8, 0.0, err);

    // Load data from the survey
    ngal = surv->get_ngal();

    // Allocate data arrays
    shear_gamma1 = (double *) malloc(sizeof(double) * ngal);
    shear_gamma2 = (double *) malloc(sizeof(double) * ngal);
    w_e          = (double *) malloc(sizeof(double) * ngal);

    if (include_flexion) {
        flexion_f1 = (double *) malloc(sizeof(double) * ngal);
        flexion_f2 = (double *) malloc(sizeof(double) * ngal);
        res_f1     = (double *) malloc(sizeof(double) * ngal);
        res_f2     = (double *) malloc(sizeof(double) * ngal);
        w_f        = (double *) malloc(sizeof(double) * ngal);
    }

    // Allocate  auxiliary arrays
    res_gamma1 = (double *) malloc(sizeof(double) * ngal);
    res_gamma2 = (double *) malloc(sizeof(double) * ngal);
    res_conv   = (double *) malloc(sizeof(double) * ngal);
    cov        = (double *) malloc(sizeof(double) * ngal);

    // Loading data from the survey and initializing arrays
    for (long i = 0; i < ngal; i++) {
        shear_gamma1[i] = surv->get_gamma1(i);
        shear_gamma2[i] = surv->get_gamma2(i);
        w_e[i]          = surv->get_shear_weight(i);
        res_gamma1[i] = 0;
        res_gamma2[i] = 0;

        if (include_flexion) {
            flexion_f1[i] = surv->get_F1(i);
            flexion_f2[i] = surv->get_F2(i);
            w_f[i]        = surv->get_flexion_weight(i);
            res_f1[i] = 0;
            res_f2[i] = 0;
        }

        res_conv[i]   = 0.;
        cov[i]        = 1.;
    }

    // Initialize the lensing planes, with one nfft per plane
    ps = (nfft_plan **) malloc((nlp) * sizeof(nfft_plan *));
    fft_frame = fftw_alloc_complex(npix * npix * nlp);
    
    // Normalization factor for the fft
    fftFactor =1.0/(((double)npix)*npix);

    for (int i = 0; i < nlp; i++) {
        // Create the nfft plan
        ps[i] = new nfft_plan;
        nfft_init_2d(ps[i], npix, npix, ngal);
        // Set up the nodes at the galaxy positions
        for (long ind = 0; ind < ngal;  ind++) {
            double ra  = surv->get_ra(ind);
            double dec = surv->get_dec(ind);
            double denom = cos(center_dec) * cos(dec) * cos(ra  - center_ra) + sin(center_dec) * sin(dec);
            double X =  cos(dec) * sin(ra  - center_ra) / denom;
            double Y = (cos(center_dec) * sin(dec) - cos(dec) * sin(center_dec) * cos(ra - center_ra)) / denom;

            double val = -0.5 + ((X) / size);
            val = val < -0.5 ? val + 1.0 : val;
            ps[i]->x[2 * ind]  = val ;
            val = -0.5 + ((Y) / size);
            val = val < -0.5 ? val + 1.0 : val;
            ps[i]->x[2 * ind + 1] = val;
        }
        /** precompute psi, the entries of the matrix B */
        nfft_precompute_one_psi(ps[i]);
        if (nfft_check(ps[i])) {
            std::cout << "Problem " << nfft_check(ps[i]) << std::endl;
        }
    }

    // Initialize the lensing kernel for each galaxy
    lensKernel = (double *) malloc(sizeof(double) * ngal * nlp);
    if (nlp == 1) {
        
        // If the lens redshift wasn't provided, use unit weights
        if(zlens <= 0){
            for(long ind =0; ind < ngal*nlp; ind++){lensKernel[ind] = 1. ;}
        }else{
            // In the 2D case, the lensing kernel is just a lensing weight based on the
            // critical surface mass density.
            compute_surface_lensing_kernel();
        }
    } else {
        // Compute the full 3D lensing kernel, to reconstruct the 3D density contrast
        compute_3D_lensing_kernel();
    }

    // Compute the ratio of shear and flexion variance if necessary
    sig_frac = 1.0;
    if (include_flexion) {
        double shear_mean    = 0;
        double flexion_mean  = 0;
        double shear_sigma   = 0;
        double flexion_sigma = 0;
        for (int i = 0; i < ngal; i++) {
            shear_mean   += shear_gamma1[i];
            flexion_mean += flexion_f1[i];
        }
        shear_mean /= ngal;
        flexion_mean /= ngal;
        for (int i = 0; i < ngal; i++) {
            shear_sigma   += pow(shear_gamma1[i] - shear_mean, 2.0);
            flexion_sigma += pow(flexion_f1[i] - flexion_mean, 2.0);
        }
        shear_sigma   /= (ngal - 1.0);
        flexion_sigma /= (ngal - 1.0);

        sig_frac = flexion_sigma / shear_sigma;
   }
    
    

}

field::~field()
{
    gsl_rng_free(rng);
    //TODO: free nicaea

    // Free data arrays
    free(shear_gamma1);
    free(shear_gamma2);
    free(w_e);
    free(res_gamma1);
    free(res_gamma2);
    free(res_conv);
    free(cov);
    free(lensKernel);

    if (include_flexion) {
        free(res_f1);
        free(res_f2);
        free(flexion_f1);
        free(flexion_f2);
        free(w_f);
    }

    // Deallocate nfft plans
    for (int i = 0; i < nlp; i++) {
        nfft_finalize(ps[i]);
    }
    free(ps);
    fftw_free(fft_frame);
}

void field::gradient(fftw_complex *delta)
{
    forward_operator(delta);

    // Compute residuals, note that we only differentiate for the second term here.
    #pragma omp parallel for
    for (int i = 0; i < ngal ; i++) {
        double factor = std::max(1. - res_conv[i], 0.);
        res_gamma1[i] = cov[i] * w_e[i] * (factor * shear_gamma1[i] - res_gamma1[i]);
        res_gamma2[i] = cov[i] * w_e[i] * (factor * shear_gamma2[i] - res_gamma2[i]);
        if (include_flexion) {
            res_f1[i] = cov[i] * w_f[i] * (factor * flexion_f1[i] - res_f1[i]);
            res_f2[i] = cov[i] * w_f[i] * (factor * flexion_f2[i] - res_f2[i]);
        }
    }

    adjoint_operator(delta);
}

void field::gradient_noise(fftw_complex *delta)
{
    for (long ind = 0; ind < ngal; ind++) {
        double theta1 = gsl_ran_flat(rng, 0, 2.0 * M_PI);
        double theta2 = gsl_ran_flat(rng, 0, 2.0 * M_PI);

        res_gamma1[ind] = sqrt(cov[ind]) * w_e[ind] * (shear_gamma1[ind] * cos(theta1) - shear_gamma2[ind] * sin(theta1));
        res_gamma2[ind] = sqrt(cov[ind]) * w_e[ind] * (shear_gamma2[ind] * cos(theta1) + shear_gamma1[ind] * sin(theta1));

        if (include_flexion) {
            res_f1[ind] = sqrt(cov[ind]) * w_f[ind] * (flexion_f1[ind] * cos(theta2) - flexion_f2[ind] * sin(theta2));
            res_f2[ind] = sqrt(cov[ind]) * w_f[ind] * (flexion_f2[ind] * cos(theta2) + flexion_f1[ind] * sin(theta2));
        }

    }

    adjoint_operator(delta);
}

void field::forward_operator(fftw_complex *delta)
{
    double freqFactor = 2.0 * M_PI / pixel_size / ((double) npix);

    fftw_complex *deltaFlex = delta + nlp * npix * npix;

    // First apply the kappa to gamma transform
    #pragma omp parallel for
    for (int z = 0; z < nlp; z++) {

        double k1, k2, k1k1, k2k2, k1k2, ksqr;
        double denom;

        // Compute residuals (1 - Ax)y - Bx
        for (int y = 0; y < npix ; y++) {
            k2 = (y - npix / 2) * freqFactor;
            int ky  = (y < npix / 2 ? y + npix / 2 : y - npix / 2);

            for (int x = 0; x < npix ; x++) {
                k1 = (x - npix / 2) * freqFactor;
                int kx  = (x < npix / 2 ? x + npix / 2 : x - npix / 2);

                long pos = ky * npix + kx + z * (npix * npix);
                if (k1 == 0 && k2 == 0) {
                    ps[z]->f_hat[y * npix + x][0] = 0;
                    ps[z]->f_hat[y * npix + x][1] = 0;
                    continue;
                }

                k1k1 = k1 * k1;
                k2k2 = k2 * k2;
                k1k2 = k1 * k2;
                ksqr = k1k1 + k2k2;

                denom = 1.0 / ksqr;
                ps[z]->f_hat[y * npix + x][0] = denom * (delta[pos][0] * (k2k2 - k1k1) + delta[pos][1] * (2.0 * k1k2));
                ps[z]->f_hat[y * npix + x][1] = denom * (delta[pos][1] * (k2k2 - k1k1) - delta[pos][0] * (2.0 * k1k2));
            }
        }

        nfft_trafo_2d(ps[z]);
    }

    // Apply the lensing efficiency kernel
    #pragma omp parallel for
    for (int i = 0; i < ngal ; i++) {
        res_gamma1[i] = 0;
        res_gamma2[i] = 0;
        for (int z = 0; z < nlp ; z++) {
            double q = lensKernel[i * nlp + z];
            res_gamma1[i] += q * ps[z]->f[i][0] * fftFactor;
            res_gamma2[i] += q * ps[z]->f[i][1] * fftFactor;
        }
    }

    if (include_flexion) {
        #pragma omp parallel for
        for (int z = 0; z < nlp; z++) {

            double k1, k2;

            for (int y = 0; y < npix ; y++) {
                k2 = (y - npix / 2) * freqFactor;
                int ky  = (y < npix / 2 ? y + npix / 2 : y - npix / 2);

                for (int x = 0; x < npix ; x++) {
                    k1 = (x - npix / 2) * freqFactor;
                    int kx  = (x < npix / 2 ? x + npix / 2 : x - npix / 2);

                    long pos = ky * npix + kx + z * (npix * npix);
                    
                    ps[z]->f_hat[y * npix + x][0] = deltaFlex[pos][0];
                    ps[z]->f_hat[y * npix + x][1] = deltaFlex[pos][1];
                }
            }

            nfft_trafo_2d(ps[z]);
        }

        // Apply the lensing efficiency kernel
        #pragma omp parallel for
        for (int i = 0; i < ngal ; i++) {
            res_f1[i] = 0;
            res_f2[i] = 0;
            for (int z = 0; z < nlp ; z++) {
                double q = lensKernel[i * nlp + z];
                res_f1[i] += q * ps[z]->f[i][0] * fftFactor;
                res_f2[i] += q * ps[z]->f[i][1] * fftFactor;
            }
        }
    }


    // Compute the value of the field evaluated at each galaxy position
    combine_components(delta, fft_frame);
    #pragma omp parallel for
    for (int z = 0; z < nlp; z++) {

        double k1, k2, k1k1, k2k2, k1k2, ksqr;
        double denom;

        //Compute reduced shear correction factor
        for (int y = 0; y < npix ; y++) {
            k2 = (y - npix / 2) * freqFactor;
            int ky  = (y < npix / 2 ? y + npix / 2 : y - npix / 2);

            for (int x = 0; x < npix ; x++) {
                k1 = (x - npix / 2) * freqFactor;
                int kx  = (x < npix / 2 ? x + npix / 2 : x - npix / 2);

                long pos = ky * npix + kx + z * (npix * npix);
                ps[z]->f_hat[y * npix + x][0] = fft_frame[pos][0];
                ps[z]->f_hat[y * npix + x][1] = fft_frame[pos][1];
            }
        }

        nfft_trafo_2d(ps[z]);
    }

    #pragma omp parallel for
    for (int i = 0; i < ngal ; i++) {
        res_conv[i] = 0;
        for (int z = 0; z < nlp ; z++) {
            double q = lensKernel[i * nlp + z];
            res_conv[i] += q * ps[z]->f[i][0] * fftFactor;
        }
    }
}


void field::adjoint_operator(fftw_complex *delta)
{
    double freqFactor = 2.0 * M_PI / pixel_size / ((double) npix);

    fftw_complex *deltaFlex = delta + nlp * npix * npix;

    #pragma omp parallel for
    for (int z = 0; z < nlp; z++) {
        double k1, k2, k1k1, k2k2, k1k2, ksqr;
        double denom;

        for (long i = 0; i < ngal ; i++) {
            double q = lensKernel[i * nlp + z];
            ps[z]->f[i][0] = res_gamma1[i] * q;
            ps[z]->f[i][1] = res_gamma2[i] * q;
        }

        nfft_adjoint_2d(ps[z]);

        for (int y = 0; y < npix ; y++) {
            k2 = (y - npix / 2) * freqFactor;
            int ky  = (y < npix / 2 ? y + npix / 2 : y - npix / 2);

            for (int x = 0; x < npix ; x++) {
                k1 = (x - npix / 2) * freqFactor;
                int kx  = (x < npix / 2 ? x + npix / 2 : x - npix / 2);

                long pos = ky * npix + kx + z * (npix * npix);
                if (k1 == 0 && k2 == 0) {
                    continue;
                }

                k1k1 = k1 * k1;
                k2k2 = k2 * k2;
                k1k2 = k1 * k2;
                ksqr = k1k1 + k2k2;

                delta[pos][0] = (ps[z]->f_hat[y * npix + x][0] * (k2k2 - k1k1) - ps[z]->f_hat[y * npix + x][1] * (2.0 * k1k2)) / ksqr;
                delta[pos][1] = (ps[z]->f_hat[y * npix + x][1] * (k2k2 - k1k1) + ps[z]->f_hat[y * npix + x][0] * (2.0 * k1k2)) / ksqr;
            }
        }
        delta[z * (npix * npix)][0] = 0;
        delta[z * (npix * npix)][1] = 0;

        if (include_flexion) {
            for (long i = 0; i < ngal ; i++) {
                double q = lensKernel[i * nlp + z];
                ps[z]->f[i][0] = res_f1[i] * q;
                ps[z]->f[i][1] = res_f2[i] * q;
            }

            nfft_adjoint_2d(ps[z]);

            for (int y = 0; y < npix ; y++) {
                k2 = (y - npix / 2) * freqFactor;
                int ky  = (y < npix / 2 ? y + npix / 2 : y - npix / 2);

                for (int x = 0; x < npix ; x++) {
                    k1 = (x - npix / 2) * freqFactor;
                    int kx  = (x < npix / 2 ? x + npix / 2 : x - npix / 2);

                    long pos = ky * npix + kx + z * (npix * npix);

                    deltaFlex[pos][0] = ps[z]->f_hat[y * npix + x][0];
                    deltaFlex[pos][1] = ps[z]->f_hat[y * npix + x][1];
                }
            }
        }
    }
}


// TODO: Check indices and convention for x and y
void field::combine_components(fftw_complex *delta, fftw_complex *delta_comb)
{

    double freqFactor = 2.0 * M_PI / pixel_size / ((double) npix);
    double k1, k2, k1k1, k2k2, k1k2, ksqr;
    double denom;

    fftw_complex *deltaFlex = delta + nlp * npix * npix;

    for (int z = 0; z < nlp; z++) {

        // Computes the convergence at the position of the
        for (int y = 0; y < npix ; y++) {
            
            k2 = (y - npix / 2) * freqFactor;
            int ky  = (y < npix / 2 ? y + npix / 2 : y - npix / 2);
            
            for (int x = 0; x < npix ; x++) {
                k1 = (x - npix / 2) * freqFactor;
                int kx  = (x < npix / 2 ? x + npix / 2 : x - npix / 2);
                    
                long pos = ky * npix + kx + z * (npix * npix);
                
                ksqr = k1 * k1 + k2 * k2;

                denom = 1.0 / (ksqr + sig_frac);
                if (include_flexion) {
                    delta_comb[pos][0] = (deltaFlex[pos][0] * k2 - deltaFlex[pos][1] * k1) * denom;
                    delta_comb[pos][1] = (deltaFlex[pos][0] * k1 + deltaFlex[pos][1] * k2) * denom;
                    delta_comb[pos][0] += denom * sig_frac * delta[pos][0];
                    delta_comb[pos][1] += denom * sig_frac * delta[pos][1];
                } else {
                    delta_comb[pos][0] = delta[pos][0];
                    delta_comb[pos][1] = delta[pos][1];
                }
            }
        }
        delta_comb[0][0] = 0;
        delta_comb[0][1] = 0;
    }
}

void field::combine_components_inverse(fftw_complex *delta_comb, fftw_complex *delta)
{
    double freqFactor = 2.0 * M_PI / pixel_size / ((double) npix);
    double k1, k2, k1k1, k2k2, k1k2, ksqr;
    double denom;

    fftw_complex * deltaFlex = delta + nlp * npix * npix;

    for (int z = 0; z < nlp; z++) {

        // Computes the convergence at the position of the galaxy
        for (int y = 0; y < npix ; y++) {
            k2 = (y - npix / 2) * freqFactor;
            int ky  = (y < npix / 2 ? y + npix / 2 : y - npix / 2);
            
            for (int x = 0; x < npix ; x++) {
                k1 = (x - npix / 2) * freqFactor;
                int kx  = (x < npix / 2 ? x + npix / 2 : x - npix / 2);
                long pos = ky * npix + kx + z * (npix * npix);
                
                if (include_flexion) {
                    deltaFlex[pos][0] = (delta_comb[pos][0] * k2 + delta_comb[pos][1] * k1);
                    deltaFlex[pos][1] = (-delta_comb[pos][0] * k1 + delta_comb[pos][1] * k2);
                    delta[pos][0] = delta_comb[pos][0];
                    delta[pos][1] = delta_comb[pos][1];
                } else {
                    delta[pos][0] = delta_comb[pos][0];
                    delta[pos][1] = delta_comb[pos][1];
                }
            }
        }
    }
}


bool field::check_adjoint()
{
    if (include_flexion) {
        fftw_complex *delta1 = fftw_alloc_complex(2 * npix * npix * nlp);
        fftw_complex *delta2 = fftw_alloc_complex(2 * npix * npix * nlp);
        double *test_g1 = (double *) malloc(sizeof(double) * ngal);
        double *test_g2 = (double *) malloc(sizeof(double) * ngal);
        double *test_f1 = (double *) malloc(sizeof(double) * ngal);
        double *test_f2 = (double *) malloc(sizeof(double) * ngal);

        for (long ind = 0; ind < nlp * npix * npix * 2 ; ind++) {
            delta1[ind][0] =  gsl_ran_gaussian(rng, 1.0);
            delta1[ind][1] =  gsl_ran_gaussian(rng, 1.0);
        }

        double result_forward   = 0;
        double result_forward2  = 0;
        double result_backward  = 0;
        double result_backward2 = 0;

        forward_operator(delta1);
        
        for (long ind = 0; ind < ngal ; ind++) {
            test_g1[ind] = res_gamma1[ind];
            test_g2[ind] = res_gamma2[ind];
            test_f1[ind] = res_f1[ind];
            test_f2[ind] = res_f2[ind];
            res_gamma1[ind] = gsl_ran_gaussian(rng, 1.0);
            res_gamma2[ind] = gsl_ran_gaussian(rng, 1.0);
            res_f1[ind]     = gsl_ran_gaussian(rng, 1.0);
            res_f2[ind]     = gsl_ran_gaussian(rng, 1.0);
        }
        adjoint_operator(delta2);

        for (long ind = 0; ind < npix * npix * nlp * 2 ; ind++) {
            result_forward  +=  delta1[ind][0] * delta2[ind][0] + delta1[ind][1] * delta2[ind][1];
            result_forward2 += -delta1[ind][1] * delta2[ind][0] + delta1[ind][0] * delta2[ind][1];
        }

        for (long ind = 0; ind < ngal; ind++) {
            result_backward  += res_gamma1[ind] * test_g1[ind] + res_gamma2[ind] * test_g2[ind] + res_f1[ind] * test_f1[ind] + res_f2[ind] * test_f2[ind];
            result_backward2 += res_gamma2[ind] * test_g1[ind] - res_gamma1[ind] * test_g2[ind] + res_f2[ind] * test_f1[ind] - res_f1[ind] * test_f2[ind];
        }
        std::cout << " Results  of check: " << result_forward << " against " << result_backward  << std::endl;
        std::cout << " Results  of check: " << result_forward2 << " against " << result_backward2  << std::endl;
    }else{
        fftw_complex *delta1 = fftw_alloc_complex( npix * npix * nlp);
        fftw_complex *delta2 = fftw_alloc_complex( npix * npix * nlp);
        
        double *test_g1 = (double *) malloc(sizeof(double) * ngal);
        double *test_g2 = (double *) malloc(sizeof(double) * ngal);

        for (long ind = 0; ind < nlp * npix * npix ; ind++) {
            delta1[ind][0] =  gsl_ran_gaussian(rng, 1.0);
            delta1[ind][1] =  gsl_ran_gaussian(rng, 1.0);
        }

        double result_forward   = 0;
        double result_forward2  = 0;
        double result_backward  = 0;
        double result_backward2 = 0;

        forward_operator(delta1);
        
        for (long ind = 0; ind < ngal ; ind++) {
            test_g1[ind] = res_gamma1[ind];
            test_g2[ind] = res_gamma2[ind];
            res_gamma1[ind] = gsl_ran_gaussian(rng, 1.0);
            res_gamma2[ind] = gsl_ran_gaussian(rng, 1.0);
        }
        
        adjoint_operator(delta2);

        for (long ind = 0; ind < npix * npix * nlp ; ind++) {
            result_forward  +=  delta1[ind][0] * delta2[ind][0]*fftFactor + delta1[ind][1] * delta2[ind][1]*fftFactor;
            result_forward2 += -delta1[ind][1] * delta2[ind][0]*fftFactor + delta1[ind][0] * delta2[ind][1]*fftFactor;
        }

        for (long ind = 0; ind < ngal; ind++) {
            result_backward  += res_gamma1[ind] * test_g1[ind] + res_gamma2[ind] * test_g2[ind];
            result_backward2 += res_gamma2[ind] * test_g1[ind] - res_gamma1[ind] * test_g2[ind];
        }
        std::cout << " Results  of check: " << result_forward << " against " << result_backward  << std::endl;
        std::cout << " Results  of check: " << result_forward2 << " against " << result_backward2  << std::endl;            
    }
    return true;
}

void field::compute_3D_lensing_kernel()
{
    //TODO: Implement the 3D lensing kernel
    for (int i = 0; i < ngal * nlp; i++) {
        lensKernel[i] = 1.0;
    }
}

typedef struct {
    double w_l;
    double w_inf;
    redshift_distribution *redshift;
    nicaea::error **err;
    nicaea::cosmo *model;
} int_for_sigma_params;

double int_for_sigma(double a_s, void *intpar)
{
    int_for_sigma_params *params    = (int_for_sigma_params *) intpar ;
    nicaea::cosmo *self             = params->model;
    nicaea::error **err             = params->err;
    redshift_distribution *redshift = params->redshift;
    double w_l                      = params->w_l;
    double w_inf                    = params->w_inf;

    double w_s = nicaea::w(self, a_s, 0, err);
    if (w_s - w_l <= 0) {
        return 0;
    }

    double p = 1.0 / a_s / a_s * redshift->pdf(1.0 / a_s - 1.0);

    return  p * ((w_s - w_l) * w_inf) / ((w_inf - w_l) * w_s);
}

void field::compute_surface_lensing_kernel()
{
    double result;
    double abserr;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(2048);

    gsl_function F;
    F.function = &int_for_sigma;

    double a_inf  = 1.0 / (1.0 + Z_INF);
    double a_lens = 1.0 / (1.0 + zlens);

    double w_l   = nicaea::w(model, a_lens, 0, err);
    double w_inf = nicaea::w(model, a_inf, 0, err);

    int_for_sigma_params params;
    params.model = model;
    params.err   = err;
    params.w_l   = w_l;
    params.w_inf = w_inf;

    for (long i = 0; i < surv->get_ngal(); i++) {

        redshift_distribution *redshift = surv->get_redshift(i);

        // Treat the case of spectroscopic redshifts
        if (spectroscopic_redshift *specz = dynamic_cast<spectroscopic_redshift *>(redshift)) {

            double afit = 1.0 / (1.0 + specz->get_redshift());
            double w_s = nicaea::w(model, afit, 0, err);

            if (afit >= a_lens) {
                lensKernel[i] = 0;
            } else {
                lensKernel[i] = ((w_s - w_l) * w_inf) / ((w_inf - w_l) * w_s);
            }
        } else {
            params.redshift = redshift;
            F.params = (void *) &params;
            gsl_integration_qags(&F, std::max(model->a_min, a_inf), a_lens, 0, 1.0e-5, 1024, w, &result, &abserr);
            lensKernel[i] = result;
        }
    }

    gsl_integration_workspace_free(w);
}


double field::get_spectral_norm(int niter, double tol) {

    double norm=0;
    double norm_old=0;

    long ncoeff = npix*npix*nlp;
    if(include_flexion)
        ncoeff *= 2;

    fftw_complex* kap = fftw_alloc_complex(ncoeff);
    fftw_complex* kap_tmp = fftw_alloc_complex(ncoeff);

    norm= 0;
    for(long ind =0; ind< ncoeff; ind++) {
        kap[ind][0] = gsl_ran_gaussian(rng,1.0);
        kap[ind][1] = gsl_ran_gaussian(rng,1.0);
        norm += kap[ind][0] * kap[ind][0] + kap[ind][1] * kap[ind][1];
    }
    norm = sqrt(norm);

    for(long ind =0; ind< ncoeff; ind++) {
        kap[ind][0] /= norm;
        kap[ind][1] /= norm;
    }


    for(int k=0; k < niter; k++) {

        // Apply operator A^t A
        combine_components(kap, kap_tmp);
        forward_operator(kap_tmp);

        // Compute residuals, note that we only differentiate for the second term here
        for(int i=0; i < ngal ; i++) {
            res_gamma1[i] = cov[i] * w_e[i] * res_gamma1[i];
            res_gamma2[i] = cov[i] * w_e[i] * res_gamma2[i];
            if(include_flexion) {
                res_f1[i] = cov[i] * w_f[i] * res_f1[i];
                res_f2[i] = cov[i] * w_f[i] * res_f2[i];
            }
        }

        // Apply adjoint operator on normalized results
        adjoint_operator(kap_tmp);
        combine_components_inverse(kap_tmp,kap);
        

        // Compute norm
        norm= 0;
        for(long ind =0; ind< ncoeff; ind++) {
            norm += kap[ind][0] * kap[ind][0] + kap[ind][1] * kap[ind][1];
        }

        norm = sqrt(norm);

        if( fabs(norm - norm_old)/norm <= tol)
            break;

        for(long ind =0; ind< ncoeff; ind++) {
            kap[ind][0] /= norm;
            kap[ind][1] /= norm;
        }

        norm_old = norm;
        //std::cout <<  "Iter : " << k << " Value " << norm <<std::endl;
        if(k == niter -1 ) std::cout << "Warning, reached maximum number of iterations" << std::endl;
    }

    fftw_free(kap);
    fftw_free(kap_tmp);

    return norm*(1.0+tol);

}

void field::update_covariance(fftw_complex* delta)
{
    double freqFactor = 2.0 * M_PI / pixel_size / ((double) npix);
    // Compute the value of the field evaluated at each galaxy position
    combine_components(delta, fft_frame);
    #pragma omp parallel for
    for (int z = 0; z < nlp; z++) {

        double k1, k2, k1k1, k2k2, k1k2, ksqr;
        double denom;

        //Compute reduced shear correction factor
        for (int y = 0; y < npix ; y++) {
            k2 = (y - npix / 2) * freqFactor;
            int ky  = (y < npix / 2 ? y + npix / 2 : y - npix / 2);

            for (int x = 0; x < npix ; x++) {
                k1 = (x - npix / 2) * freqFactor;
                int kx  = (x < npix / 2 ? x + npix / 2 : x - npix / 2);

                long pos = ky * npix + kx + z * (npix * npix);
                ps[z]->f_hat[y * npix + x][0] = fft_frame[pos][0];
                ps[z]->f_hat[y * npix + x][1] = fft_frame[pos][1];
            }
        }

        nfft_trafo_2d(ps[z]);
    }

    #pragma omp parallel for
    for (int i = 0; i < ngal ; i++) {
        res_conv[i] = 0;
        for (int z = 0; z < nlp ; z++) {
            double q = lensKernel[i * nlp + z];
            res_conv[i] += q * ps[z]->f[i][0] * fftFactor;
        }
    }
    
    for(int i=0; i < ngal ; i++) {
        double factor = std::max(1.0 -  res_conv[i],0.3);
        cov[i] = 1.0/(factor*factor);
    }
}
