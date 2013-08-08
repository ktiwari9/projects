/* 
 *  Copyright (c) 2008-2010  Noah Snavely (snavely (at) cs.cornell.edu)
 *    and the University of Washington
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 */

/* BundlerApp.cpp */

#include "getopt.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "BundlerApp.h"

#include "Epipolar.h"
#include "Register.h"
#include "SifterUtil.h"

#ifndef __USE_ANN__
#include "BruteForceSearch.h"
#endif

#include "defines.h"
#include "fit.h"
#include "fmatrix.h"
#include "image.h"
#include "matrix.h"
#include "qsort.h"
#include "util.h"

#define SIFT_COMMAND "/homes/gws/snavely/code/bin/sift"

#define SUBSAMPLE_LEVEL 1

#define MIN_MATCHES 10

extern int optind;

static void ReadFisheyeParameters(char *filename, 
				  double &fCx, double &fCy, 
				  double &fRad, double &fAngle, 
                                  double &fFocal)
{
    fCx = 0.0;
    fCy = 0.0;

    /* Read fisheye params */
    FILE *f = fopen(filename, "r");

    if (f == NULL) {
	printf("Error opening fisheye parameters file %s for reading\n", 
	       filename);
	exit(1);
    }

    char buf[256];
    while (fgets(buf, 256, f) != NULL) {
	/* Split the command into tokens */
    std::string str(buf);
    
	std::vector<std::string> toks;
	
    Tokenize(str, toks, " ");

	if (strcmp(toks[0].c_str(), "FisheyeCenter:") == 0) {
	    if (toks.size() != 3) {
		printf("FisheyeCenter needs two arguments!\n");
		exit(1);
	    } else {
		fCx = atof(toks[1].c_str());
		fCy = atof(toks[2].c_str());
	    }
	} else if (strcmp(toks[0].c_str(), "FisheyeRadius:") == 0) {
	    if (toks.size() != 2) {
		printf("FisheyeRadius needs one argument!\n");
		exit(1);
	    } else {
		fRad = atof(toks[1].c_str());
	    }
	} else if (strcmp(toks[0].c_str(), "FisheyeAngle:") == 0) {
	    if (toks.size() != 2) {
		printf("FisheyeAngle needs one argument!\n");
		exit(1);
	    } else {
		fAngle = atof(toks[1].c_str());
	    }	    
	} else if (strcmp(toks[0].c_str(), "FisheyeFocal:") == 0) {
	    if (toks.size() != 2) {
		printf("FisheyeFocal needs one argument!\n");
		exit(1);
	    } else {
		fFocal = atof(toks[1].c_str());
	    }
	} else {
	    printf("Unrecognized fisheye field %s\n", toks[0].c_str());
	}
    }

    fclose(f);
}

bool BundlerApp::OnInit()
{
    printf("[OnInit] Running program %s\n", argv[0]);

    char *imageList;
    
    bool load_file = false;

    if (argc >= 2) {
	printf("Loading images from file '%s'\n", argv[1]);
	imageList = argv[1];
	load_file = true;
    } else {
	PrintUsage();
	exit(0);
    }

    printf("[BundlerApp::OnInit] Processing options...\n");
    ProcessOptions(argc - 1, argv + 1);

    if (m_use_intrinsics && m_estimate_distortion) {
        printf("Error: --intrinsics and --estimate_distortion "
               "are incompatible\n");
        exit(1);
    }

    if (m_fixed_focal_length && m_estimate_distortion) {
        printf("Error: --fixed_focal_length and --estimate_distortion "
               "are currently incompatible\n");
        exit(1);
    }

    printf("[BundlerApp::OnInit] Loading frame...\n");

    printf("[BundlerApp::OnInit] Loading images...\n");
    fflush(stdout);
    if (load_file) {
	FILE *f = fopen(imageList, "r");

	if (f == NULL) {
	    printf("[BundlerApp::OnInit] Error opening file %s for reading\n",
		   imageList);
	    exit(1);
	}

        LoadImageNamesFromFile(f);

	int num_images = GetNumImages();

	if (m_fisheye) {
	    double fCx = 0.0, fCy = 0.0, fRad = 0.0, fAngle = 0.0, fFocal = 0.0;
	    ReadFisheyeParameters(m_fisheye_params, 
				             fCx, fCy, fRad, fAngle, fFocal);
	    
	    for (int i = 0; i < num_images; i++) {
            if (m_image_data[i].m_fisheye) {
                m_image_data[i].m_fCx = fCx;
                m_image_data[i].m_fCy = fCy;
                m_image_data[i].m_fRad = fRad;
                m_image_data[i].m_fAngle = fAngle;
                m_image_data[i].m_fFocal = fFocal;
            }                
	    }
	}

	fclose(f);
    }

    if (m_rerun_bundle) {
        // ReadCameraConstraints();
        assert(m_bundle_provided);

        printf("[BundlerApp::OnInit] Reading bundle file...\n");
        ReadBundleFile(m_bundle_file);

        if (m_bundle_version < 0.3) {
            printf("[BundlerApp::OnInit] Reflecting scene...\n");
            FixReflectionBug();
        }

        ReRunSFM();
        exit(0);
    }

    if (m_use_constraints) {
        ReadCameraConstraints();
    }

    if (m_ignore_file != NULL) {
        printf("[BundlerApp::OnInit] Reading ignore file...\n");
        ReadIgnoreFile();
    }
    
    if (m_ignore_file != NULL) {
        printf("[BundlerApp::OnInit] Reading ignore file...\n");
        ReadIgnoreFile();
    }

    /* Do bundle adjustment (or read from file if provided) */
    // ParseCommand("UndistortAll", NULL);
    if (m_bundle_provided) {
        printf("[BundlerApp::OnInit] Reading bundle file...\n");
        ReadBundleFile(m_bundle_file);

        if (m_bundle_version < 0.3) {
            printf("[BundlerApp::OnInit] Reflecting scene...\n");
            FixReflectionBug();
        }

        if (m_compress_list) {
            OutputCompressed();
            return 0;
        }

        if (m_reposition_scene) {
            double center[3], R[9], scale;
            RepositionScene(center, R, scale);
            OutputCompressed("reposition");
            return 0;
        }

        if (m_prune_bad_points) {
            SetupImagePoints(3);
            RemoveBadImages(24);
            PruneBadPoints();
            OutputCompressed("pruned");
            return 0;
        }

        if (m_scale_focal != 1.0) {
            ScaleFocalLengths(m_scale_focal);
            return 0;
        }

        if (m_scale_focal_file != NULL) {
            ScaleFocalLengths(m_scale_focal_file);
            return 0;
        }

        if (m_rotate_cameras_file != NULL) {
            RotateCameras(m_rotate_cameras_file);
        }



        if (m_track_file != NULL) {
            CreateTracksFromPoints();
            WriteTracks(m_track_file);
        }

        if (m_zero_distortion_params) {
            ZeroDistortionParams();
            OutputCompressed("nord");
            return 0;
        }


        
        if (m_output_relposes) {
            double center[3], R[9], scale;
            RepositionScene(center, R, scale);
            RepositionScene(center, R, scale);
            // OutputRelativePoses2D(m_output_relposes_file);
            OutputRelativePoses3D(m_output_relposes_file);
            return 0;
        }

        if (m_compute_covariance) {
            ComputeCameraCovariance();
            return 0;
        }
        
#define MIN_POINT_VIEWS 3 // 0 // 2
        if (!m_run_bundle) {
            SetMatchesFromPoints(MIN_POINT_VIEWS);
            // WriteMatchTableDrew(".final");            

            printf("[BundlerApp::OnInit] "
                   "Setting up image points and lines...\n");
            SetupImagePoints(/*2*/ MIN_POINT_VIEWS);
            RemoveBadImages(6);

            if (m_point_constraint_file != NULL) {
                printf("[BundlerApp::OnInit] Reading point constraints...\n");
                m_use_point_constraints = true;
                ReadPointConstraints();
            }

            printf("[BundlerApp::OnInit] Scaling world...\n");

            printf("[BundlerApp::OnInit] Computing camera orientations...\n");
            ComputeImageRotations();

            double center[3], R[9], scale;
            RepositionScene(center, R, scale);

            if (m_rerun_bundle) {
                ReRunSFM();
            }
        }

        if (m_add_image_file != NULL) {
            printf("[BundlerApp::OnInit] Adding additional images...\n");
            FILE *f = fopen(m_add_image_file, "r");

            if (f == NULL) {
                printf("[BundlerApp::OnInit] Error opening file %s for "
                       "reading\n",
                       m_add_image_file);
            } else {
                BundleImagesFromFile(f);

                /* Write the output */
                OutputCompressed("added");

                if (m_bundle_version < 0.3)
                    FixReflectionBug();

                // RunSFMWithNewImages(4);
                fclose(f);
            }
        }
    }

    if (m_run_bundle) {
        if (!m_fast_bundle)
            BundleAdjust();
        else
            BundleAdjustFast();
        
        if (m_bundle_version < 0.3)
            FixReflectionBug();

	exit(0);
    }

    return true;
}

static BundlerApp *bundler_app = NULL;

int main(int argc, char **argv) 
{
    // mtrace();

    bundler_app = new BundlerApp();
    bundler_app->argc = argc;
    bundler_app->argv = argv;

    bundler_app->OnInit();
}
