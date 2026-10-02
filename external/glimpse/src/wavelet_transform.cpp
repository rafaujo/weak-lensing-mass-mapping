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

#include <sparse2d/MR_Obj.h>
#include "wavelet_transform.h"
#include "starlet_2d.h"

wavelet_transform::wavelet_transform(int npix, int nscale):
    npix(npix), nscale(nscale)
{

    frame1 = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * npix * npix);
    frame2 = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * npix * npix);

    plan_forward   = fftw_plan_dft_2d(npix, npix, frame1, frame1, FFTW_FORWARD,   FFTW_MEASURE);
    plan_backward  = fftw_plan_dft_2d(npix, npix, frame2, frame2, FFTW_BACKWARD,  FFTW_MEASURE);

    // We begin with starlets combined with battle_lemarie wavelets
    nframes = 4 * (nscale - 1) + 2; // 3 directional + 1 isotropic at each scale plus the 2 smooth planes

    frames = new fftw_complex*[nframes];
    for (int i = 0; i < nframes; i++) {
        frames[i] = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * npix * npix);
    }

    // We extract the atom for each frame
    starlet_2d star(npix, npix, nscale);

    dblarray image(npix, npix);
    image.init(0);
    image(npix / 2, npix / 2) = 1.0;
    dblarray alphaStar(npix, npix, nscale);
    star.transform_gen1(image.buffer(), alphaStar.buffer());
    
    for (int i = 0; i < nscale; i++) {
        for (long ind = 0; ind < npix * npix; ind++) {
            frame1[ind][0] = alphaStar.buffer()[i * npix * npix + ind];
            frame1[ind][1] = 0;
        }

        fftw_execute(plan_forward);

        for (long ind = 0; ind < npix * npix; ind++) {
            frames[i][ind][0] = sqrt(frame1[ind][0] * frame1[ind][0] + frame1[ind][1] * frame1[ind][1]);
            frames[i][ind][1] = 0;
        }
    }

    MultiResol mr;
    FilterAnaSynt FAS;
    FilterAnaSynt *PtrFAS = NULL;
    FAS.alloc(F_LEMARIE_5);
    PtrFAS = &FAS;
    mr.alloc(npix, npix, nscale, TO_UNDECIMATED_MALLAT, PtrFAS);

    Ifloat  im(npix, npix);
    im(npix / 2, npix / 2) = 1.0;
    mr.transform(im);
    for (int i = 0; i < mr.nbr_band(); i++) {
        im = mr.band(i);
        for (long x = 0; x < npix; x++) {
            int k1 = x - npix / 2;
            k1 = k1 < 0 ? npix + k1 : k1;
            for (long y = 0; y < npix; y++) {
                int k2 = y - npix / 2;
                k2 = k2 < 0 ? npix + k2 : k2;
                frame1[x + npix * y][0] = im.buffer()[k1 + npix * k2];
                frame1[x + npix * y][1] = 0;
            }
        }


        fftw_execute(plan_forward);

        for (long ind = 0; ind < npix * npix ; ind++) {
            frames[i + nscale][ind][0] = sqrt(frame1[ind][0] * frame1[ind][0] + frame1[ind][1] * frame1[ind][1]);
            frames[i + nscale][ind][1] = 0;
        }
    }
}

wavelet_transform::~wavelet_transform()
{
    fftw_free(frame1);
    fftw_free(frame2);
    for(int i=0; i < nframes; i++){
        fftw_free(frames[i]);
    }
    delete[] frames;
}

void wavelet_transform::transform(fftw_complex *image, double *alpha)
{
    for (int i = 0; i < nframes; i++) {

        for (long ind = 0; ind < npix * npix; ind++) {
            frame2[ind][0] = image[ind][0] * frames[i][ind][0];
            frame2[ind][1] = image[ind][1] * frames[i][ind][0];
        }

        fftw_execute(plan_backward);

        for (long ind = 0; ind < npix * npix; ind++) {
            alpha[ind + i * npix * npix] = frame2[ind][0] / npix;
        }
    }
}

void wavelet_transform::trans_adjoint(double *alpha, fftw_complex *image)
{
    for (long ind = 0; ind < npix * npix; ind++) {
        image[ind][0] = 0;
        image[ind][1] = 0;
    }

    for (int i = 0; i < nframes; i++) {

        for (long ind = 0; ind < npix * npix; ind++) {
            frame1[ind][0] = alpha[ind + i * (npix * npix)];
            frame1[ind][1] = 0;
        }

        fftw_execute(plan_forward);

        for (long ind = 0; ind < npix * npix; ind++) {
            image[ind][0] += frame1[ind][0] * frames[i][ind][0] / npix;
            image[ind][1] += frame1[ind][1] * frames[i][ind][0] / npix;
        }
    }
}
