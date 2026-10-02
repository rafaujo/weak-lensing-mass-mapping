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
#include <cmath>

#include "surface_reconstruction.h"

using namespace std;

surface_reconstruction::surface_reconstruction(boost::property_tree::ptree config, field *fi)
{
    f = fi;
    // Get options from the configuration file.
    nRecIter      = config.get<int>("parameters.niter", 500);
    nscales       = config.get<int>("parameters.nscales", 4);
    lambda        = config.get<double>("parameters.lambda", 4.0);
    nrandom       = config.get<int>("parameters.nrandom", 1000.0);
    nreweights    = config.get<int>("parameters.nreweights", 5);
    positivity    = config.get<bool>("parameters.positivity", false);
    double bl_reg = config.get<double>("parameters.battle_lemarie_reg", 0.1);
    double ls_reg = config.get<double>("parameters.last_scale_reg", 0.1);


    cout << "Using following reconstruction parameters :" << std::endl;
    cout << "Number of scales         : " << nscales << endl;
    cout << "Regularisation parameter : " << lambda  << endl;

    // Get characteristics of the field
    npix = f->get_npix();

    // Allocate wavelet transform
    wav = new wavelet_transform(npix, nscales);

    // Effective number of wavelet frames
    nframes = wav->get_nframes();

    ncoeff = npix * npix * 2;   // The factor 2 is to include both shear and flexion
    nwavcoeff = npix * npix * nframes;

    // Allocating internal arrays
    kappa       = fftw_alloc_complex(ncoeff);
    kappa_u     = fftw_alloc_complex(ncoeff);
    kappa_rec   = fftw_alloc_complex(ncoeff);
    kappa_old   = fftw_alloc_complex(ncoeff);
    kappa_grad  = fftw_alloc_complex(ncoeff);
    kappa_tmp   = fftw_alloc_complex(ncoeff);
    kappa_trans = fftw_alloc_complex(ncoeff);
    alpha       = (double *) malloc(sizeof(double) * nwavcoeff);
    alpha_u     = (double *) malloc(sizeof(double) * nwavcoeff);
    alpha_res   = (double *) malloc(sizeof(double) * nwavcoeff);
    alpha_tmp   = (double *) malloc(sizeof(double) * nwavcoeff);
    thresholds  = (double *) malloc(sizeof(double) * nwavcoeff);
    weights     = (double *) malloc(sizeof(double) * nwavcoeff);
    support     = (double *) malloc(sizeof(double) * nwavcoeff);

    // Initialise internal arrays
    for (long ind = 0; ind < ncoeff; ind++) {
        kappa[ind][0]     = 0;
        kappa[ind][1]     = 0;
        kappa_u[ind][0]   = 0;
        kappa_u[ind][1]   = 0;
        kappa_old[ind][0] = 0;
        kappa_old[ind][1] = 0;
        kappa_rec[ind][0] = 0;
        kappa_rec[ind][1] = 0;
        kappa_grad[ind][0] = 0;
        kappa_grad[ind][1] = 0;
        kappa_tmp[ind][0] = 0;
        kappa_tmp[ind][1] = 0;
    }

    for (long ind = 0; ind < nwavcoeff; ind++) {
        alpha[ind]     = 0;
        alpha_u[ind]   = 0;
        alpha_res[ind] = 0;
        alpha_tmp[ind] = 0;
        thresholds[ind] = 0;
        weights[ind]   = 1;
        support[ind]   = 1;
    }

    // Normalization factor for the fft
    fftFactor     = 1.0 / (((double)npix) * npix);
    fft_frame     = fftw_alloc_complex(ncoeff);
    plan_forward  = fftw_plan_dft_2d(npix, npix, fft_frame, fft_frame, FFTW_FORWARD,  FFTW_MEASURE);
    plan_backward = fftw_plan_dft_2d(npix, npix, fft_frame, fft_frame, FFTW_BACKWARD, FFTW_MEASURE);

    // Initialize the threshold levels, with lower thresholds on larger scales
    sigma_thr = (double *) malloc(sizeof(double) * nframes);
    for (int i = 0; i < nscales - 1; i++) {
        sigma_thr[i] = lambda * sqrt(2 * log(npix / pow(2.0, i) * npix / pow(2.0, i))) / sqrt(2 * log(npix * npix));
    }

    // Special regularisation for the smooth approximation
    sigma_thr[nscales - 1] = ls_reg;

    // Additional regularisation for additional BL frames
    for (int i = nscales; i < nscales + 3; i++) {
        sigma_thr[i] = bl_reg;
    }

    // Using only the first scale of the BL transform
    for (int i = nscales + 3; i < nframes ; i++) {
        sigma_thr[i] = 0;
    }

    mu1 = get_spectral_norm_prox(100, 1e-7);
    // mu2 = f->get_spectral_norm(200, 1e-7);
    sig = 1.0 / mu1;
    // tau = 0.9 / (mu2 / 2.0 + sig * mu1);

}

surface_reconstruction::~surface_reconstruction()
{
    fftw_free(kappa);
    fftw_free(kappa_u);
    fftw_free(kappa_rec);
    fftw_free(kappa_old);
    fftw_free(kappa_grad);
    fftw_free(kappa_tmp);
    fftw_free(kappa_trans);
    fftw_free(fft_frame);
    free(sigma_thr);
    free(alpha);
    free(alpha_u);
    free(alpha_res);
    free(alpha_tmp);
    free(thresholds);
    free(weights);

    // TODO: delete fftw plans ?

    delete wav;
}

void surface_reconstruction::run_main_iteration(long int niter, bool debias)
{
    mu2 = f->get_spectral_norm(200, 1e-7);
    tau = 0.9 / (mu2 / 2.0 + sig * mu1);

    std::cout << "Step size : " << tau << std::endl;

    for (long iter = 0; iter < niter; iter++) {
        if (iter % 100 == 0) {
            std::cout << "Iteration :" << iter << std::endl;
        }

        // Copy kappa for computing gradient step
        for (long ind = 0; ind < ncoeff; ind++) {
            kappa_grad[ind][0] = kappa[ind][0];
            kappa_grad[ind][1] = kappa[ind][1];
            kappa_old[ind][0]  = kappa[ind][0];
            kappa_old[ind][1]  = kappa[ind][1];
        }

        f->gradient(kappa_grad);

        // Reconstructing from wavelet coefficients
        wav->trans_adjoint(alpha, kappa_trans);
        f->combine_components_inverse(kappa_trans, kappa_u);

        // Updating kappa
        for (long ind = 0; ind < ncoeff; ind++) {
            kappa[ind][0] += tau * (kappa_grad[ind][0] - kappa_u[ind][0]) ;
            kappa[ind][1] += tau * (kappa_grad[ind][1] - kappa_u[ind][1]) ;
        }

        // Here is the place to compute the prox of the E mode constraint
        f->combine_components(kappa, kappa_tmp);
        for (long ind = 0; ind < npix * npix; ind++) {
            fft_frame[ind][0] = kappa_tmp[ind][0] * fftFactor;
            fft_frame[ind][1] = kappa_tmp[ind][1] * fftFactor;
        }
        fftw_execute(plan_backward);

        if (positivity) {
            for (long ind = 0; ind < npix * npix; ind++) {
                fft_frame[ind][0] = max(fft_frame[ind][0], 0.);
                fft_frame[ind][1] = 0;
            }
        } else {
            for (long ind = 0; ind < npix * npix; ind++) {
                fft_frame[ind][1] = 0;
            }
        }

        fftw_execute(plan_forward);
        for (long ind = 0; ind < npix * npix; ind++) {
            kappa_tmp[ind][0] = fft_frame[ind][0];
            kappa_tmp[ind][1] = fft_frame[ind][1];
        }
        f->combine_components_inverse(kappa_tmp, kappa);
        /////////////////////////////////////////////////////////

        for (long ind = 0; ind < ncoeff; ind++) {
            kappa_tmp[ind][0] = 2 * kappa[ind][0] - kappa_old[ind][0];
            kappa_tmp[ind][1] = 2 * kappa[ind][1] - kappa_old[ind][1];
        }

        f->combine_components(kappa_tmp, kappa_trans);
        wav->transform(kappa_trans, alpha_u);

        for (int j = 0; j < nframes; j++) {
            for (long ind = 0; ind < npix * npix; ind++) {
                double dum = alpha[j * npix * npix + ind] + sig * alpha_u[j * npix * npix + ind];

                if (debias) {
                    if (j == (nscales - 1)) {
                        alpha[j * npix * npix + ind] = 0;
                    } else {
                        alpha[j * npix * npix + ind] = dum - dum * support[j * npix * npix + ind] ;
                    }
                } else {
                    double val = dum - copysign(max(fabs(dum) - sigma_thr[j] * thresholds[j * npix * npix + ind] * weights[j * npix * npix + ind], 0.0), dum);
                    support[j * npix * npix + ind] = fabs(val) < fabs(dum) ? 1 : 0;
                    alpha[j * npix * npix + ind] = val;
                }
            }
        }
    }

}

void surface_reconstruction::reconstruct()
{
    std::cout << "Computing thresholds" << std::endl;
    compute_thresholds(nrandom);

    std::cout << "Running main iteration" << std::endl;
    run_main_iteration(nRecIter);

    // Reweighted l1 loop
    for (int i = 0; i < nreweights ; i++) {
        f->update_covariance(kappa);
        compute_thresholds(nrandom / 2);
        compute_weights();
        run_main_iteration(nRecIter / 2);
    }

    std::cout  << "Starting debiasing " << std::endl;
    // Final debiasing step
    f->update_covariance(kappa);
    run_main_iteration(nRecIter * 5, true);
}

void surface_reconstruction::compute_thresholds(int niter)
{

    for (long ind = 0; ind < npix * npix * nframes; ind++) {
        thresholds[ind] = 0;
    }

    for (int i = 0; i < niter ; i++) {
        f->gradient_noise(kappa_rec);
        f->combine_components(kappa_rec, kappa_trans);
        wav->transform(kappa_trans, alpha_tmp);

        // Compute gradient step
        for (long ind = 0; ind < npix * npix * nframes; ind++) {

            thresholds[ind] += pow(alpha_tmp[ind], 2.0);
        }
    }

    for (long n = 0; n < nframes; n++) {
        double maxThr = 0;
        for (long ind = 0; ind < npix * npix; ind++) {
            thresholds[n * npix * npix + ind] = sqrt(1.0 / ((double) niter) * thresholds[n * npix * npix + ind]);
            maxThr = thresholds[n * npix * npix + ind] > maxThr ? thresholds[n * npix * npix + ind] : maxThr;
        }
        for (long ind = 0; ind < npix * npix; ind++) {
            thresholds[n * npix * npix + ind] = max(thresholds[n * npix * npix + ind], maxThr * 0.1);
        }
    }
}

double surface_reconstruction::get_spectral_norm_prox(int niter, double tol)
{

    double *pt_vec = (double *) malloc(sizeof(double) * npix * npix * nframes);

    double norm = 0;
    double norm_old = 0;

    // Initialise array with random numbers
    norm = 0;
    f->gradient_noise(kappa_tmp);
    for (long ind = 0; ind < ncoeff; ind++) {
        norm += kappa_tmp[ind][0] * kappa_tmp[ind][0] + kappa_tmp[ind][1] * kappa_tmp[ind][1];
    }
    norm = sqrt(norm);

    // Normalise the input
    for (long ind = 0; ind < ncoeff; ind++) {
        kappa_tmp[ind][0] /= norm;
        kappa_tmp[ind][1] /= norm;
    }

    for (int k = 0; k < niter; k++) {
        wav->transform(kappa_rec, alpha_u);
        wav->trans_adjoint(alpha_u, kappa_rec);

        // Compute norm
        for (long ind = 0; ind < ncoeff; ind++) {
            norm += kappa_tmp[ind][0] * kappa_tmp[ind][0] + kappa_tmp[ind][1] * kappa_tmp[ind][1];
        }
        norm = sqrt(norm);


        if (fabs(norm - norm_old) / norm <= tol) {
            break;
        }

        for (long ind = 0; ind < ncoeff; ind++) {
            kappa_tmp[ind][0] /= norm;
            kappa_tmp[ind][1] /= norm;
        }

        norm_old = norm;
    }


    free(pt_vec);

    return norm * (1.0 + tol);
}



void surface_reconstruction::compute_weights()
{
    for (long ind = 0; ind < npix * npix; ind++) {
        fft_frame[ind][0] = kappa[ind][0] * fftFactor;
        fft_frame[ind][1] = kappa[ind][1] * fftFactor;
    }
    fftw_execute(plan_backward);
    for (long ind = 0; ind < npix * npix; ind++) {
        fft_frame[ind][0] = max(fft_frame[ind][0], 0.0);
        fft_frame[ind][1] = 0;
    }

    fftw_execute(plan_forward);
    for (long ind = 0; ind < npix * npix; ind++) {
        kappa_tmp[ind][0] = fft_frame[ind][0];
        kappa_tmp[ind][1] = fft_frame[ind][1];
    }

    wav->transform(kappa_tmp, alpha_res);

    for (long ind = 0; ind < npix * npix * nframes; ind++) {
        if (fabs(alpha_res[ind]) < lambda * thresholds[ind]) {
            weights[ind] = 1.0;
        } else {
            weights[ind] = lambda * thresholds[ind] / fabs(alpha_res[ind]);
        }
    }
}

void surface_reconstruction::get_convergence_map(double *kap)
{

    for (long ind = 0; ind < npix * npix; ind++) {
        fft_frame[ind][0] = kappa[ind][0] * fftFactor;
        fft_frame[ind][1] = kappa[ind][1] * fftFactor;
    }

    fftw_execute(plan_backward);

    for (int y = 0; y < npix ; y++) {
        for (int x = 0; x < npix ; x++) {
            long pos = (npix - y - 1) * npix + (npix - x - 1);
            kap[x * npix + y] = fft_frame[pos][0];
        }

    }
}


