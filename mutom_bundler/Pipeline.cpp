#include "Pipeline.h"

using namespace std;
using namespace cv;

int Pipeline::extract_features (Frame &frm)
{
  cout << "---------------- Extracting features ----------------" << endl;
  m_start = clock();
  if ( frm.m_kpts.size () != 0 || frm.m_descs.cols != 0)
  {
    cout << "Warning: Pipeline::extract_features: kpts or descs already there." << endl;
    return 0;
  }
  else
  {
    M.extract(frm.m_img, frm.m_kpts, frm.m_descs);
    m_ends = clock();
    cout << "Extracted " << frm.m_kpts.size() << " keypoints." << endl;
  }

  cout << "Extraction time : " << (double) (m_ends - m_start) / CLOCKS_PER_SEC << endl;
  return frm.m_kpts.size();
}

int Pipeline::match_frames (Frame &frm1, Frame &frm2)
{
  // Make sure features have been extracted in both frames
  if (frm1.m_kpts.size() == 0)
  {
    M.extract(frm1.m_img, frm1.m_kpts, frm1.m_descs);
    cout << "Warning: Pipeline::match_frames: Frame " << frm1.m_id << " had not features extracted" << endl;
  }
  if (frm2.m_kpts.size() == 0)
  {
    M.extract(frm2.m_img, frm2.m_kpts, frm2.m_descs);
    cout << "Warning: Pipeline::match_frames: Frame " << frm2.m_id << " had not features extracted" << endl;
  }

  cout << "---------------- Matching frames ----------------" << endl;
  m_start = clock();

  vector<int> idx1;
  vector<int> idx2;

  M.match (frm1.m_kpts,	 frm1.m_descs,
           frm2.m_kpts,	frm2.m_descs,
           idx1, idx2);
  // Save matches
  frm1.add_matches(frm2.id (), idx1);
  frm2.add_matches(frm1.id (), idx2);

  m_ends = clock();
  cout << "Matched " << idx1.size() << " keypoints." << endl;
  cout << "Matching time : " << (double) (m_ends - m_start) / CLOCKS_PER_SEC << endl;

  return idx1.size();
}

void Pipeline::get_pose_matches(Frame &frm1, Frame &frm2,
                                 vector<int> &out_pose_2d,
                                 vector<int> &out_pose_3d)
{
  if (frm1.m_3d_matches.size () == 0)
  {
    cout << "Warning: Pipeline::get_pose_matches: no index for 3d points." << endl;
  }

	// For each index of frame 1, matching with an index in frame 2
	// see if it has a corresponding 3D pts.
	int idx_3d = -1;

	for (size_t i=0; i<frm1.m_matches[frm2.id ()].size (); ++i)
	{
	  idx_3d = -1;
	  // if there is a corresponding 3D points saves its corresponding index
	  // and its 3d point index in frame 2.
	  if ( (idx_3d = frm1.get_corr_3d_pts (frm1.m_matches[frm2.id ()][i])) >= 0)
	  {
	    out_pose_2d.push_back (frm2.m_matches[frm1.id ()][i]);
	    out_pose_3d.push_back (idx_3d);
	  }
	}

	cout << "Matches for pose estimation: "<< out_pose_2d.size() << "/" << frm1.m_matches[frm2.id ()].size() << endl;
}

void Pipeline::get_triang_matches(Frame &frm1, Frame &frm2,
                                 vector<int> &out_triang1,
                                 vector<int> &out_triang2)
{
  if (frm1.m_3d_matches.size () == 0)
  {
    cout << "Warning: Pipeline::get_triang_matches: no index for 3d points." << endl;
  }

	// For each index of frame 1, matching with an index in frame 2
	// see if it has a corresponding 3D pts.
	int idx_3d = -1;

	for (size_t i=0; i<frm1.m_matches[frm2.id ()].size (); ++i)
	{
	  idx_3d = -1;
	  // if there is a corresponding 3D points saves its corresponding index
	  // and its 3d point index in frame 2.
	  if ( (idx_3d = frm1.get_corr_3d_pts (frm1.m_matches[frm2.id ()][i])) < 0)
	  {
		  out_triang1.push_back (frm1.m_matches[frm2.id ()][i]);
		  out_triang2.push_back (frm2.m_matches[frm1.id ()][i]);
		}
	}

	cout << "Matches for points triangulation:" << out_triang1.size () << "/" << frm1.m_matches[frm2.id ()].size () << endl;
}

double Pipeline::estimate_pose_init (Frame &frm1, Frame &frm2)
{
  int n = frm1.m_matches[frm2.id ()].size ();

  // Check if there are enough matches
  if (n < 5)
  {
    cout << "Warning: Pipeline::estimate_pose_init: not enough matches. (got only " << n << " matches)" << endl;
    return (-1);
  }

  cout << "---------------- Estimate pose 5point ----------------" << endl;
  m_start = clock ();

  int n_in = 0;
  double scale = 1.0;
  // Putting triang points (to triangulate), pose is not needed here so
  // we pass null vectors
  vector<int> pose_2d;
  vector<int> pose_3d;
  get_pose_matches(frm1, frm2, pose_2d, pose_3d);
  n_in = P.estimate_pose_5point (frm1.m_kpts, frm1.m_matches[frm2.id ()],
                                  frm2.m_kpts, frm2.m_matches[frm1.id ()],
                                  scale, m_K,
                                  frm2.m_R, frm2.m_t);
  m_ends = clock();

  cout << n_in << "/" << n << " inliers." << endl;
  cout << "5points time : " << (double) (m_ends - m_start) / CLOCKS_PER_SEC << endl;
}

double Pipeline::estimate_pose_run (Frame &frm1, Frame &frm2, vector<Point3d> p3d)
{
  int n = frm1.m_matches[frm2.m_id].size();

  //Check if there are enough matches
  if (n < 3)
  {
    cout << "Warning: Pipeline::estimate_pose_init: not enough matches. (got only " << n << " matches)" << endl;
    return (-1);
  }

  cout << "---------------- Estimate pose 3point ----------------" << endl;
  int n_in = 0;
  double scale = 1.0;
  // Get intersection between points that see 3D points from Frame1 and points
  // that have been matched with Frame2
  vector<int> pose_2d;
  vector<int> pose_3d;
  get_pose_matches(frm1, frm2, pose_2d, pose_3d);
  n_in = P.estimate_pose_3point_cv (p3d, pose_3d,
                                 frm2.m_kpts, pose_2d,
                                 m_K,
                                 frm2.m_R, frm2.m_t);

  cout << n_in << "/" << pose_2d.size() << " inliers." << endl;
}

int Pipeline::triangulate_frames (Frame &frm1, Frame &frm2,
                                  vector<Point3d> &p3d)
{
  cout << "---------------- Triangulate frames ----------------" << endl;
  vector<int> triang1;
  vector<int> triang2;
  get_triang_matches(frm1, frm2, triang1, triang2);
  int n = triang1.size ();
  m_start = clock();
  int n_pos = 0;

  vector<Point3d> pointcloud;
  vector<int> idx;

  n_pos = T.triangulate_bundler (frm1.m_kpts, triang1,
                                frm2.m_kpts, triang2,
                                frm1.m_R,
                                frm1.m_t,
                                frm2.m_R,
                                frm2.m_t,
                                m_K,
                                pointcloud);

  // Put all computed variables in place
  for (size_t i=0; i<pointcloud.size(); ++i)
	{
	  p3d.push_back (pointcloud[i]);
		idx.push_back (p3d.size() - 1 );
  }

  frm1.add_3d_matches(frm2.id (), idx);
  frm2.add_3d_matches(frm1.id (), idx);
  frm1.add_triang_matches(frm2.id (), triang1);
  frm2.add_triang_matches(frm1.id (), triang2);

  m_ends = clock();
  print_to_file("robust_pipeline.txt", pointcloud);
  cout << n_pos << "/" << frm1.m_matches[frm2.m_id].size() << " positive depth points." << endl;

  cout << "Triangulation time : " << (double) (m_ends - m_start) / CLOCKS_PER_SEC << endl;

  return n_pos;
}

void Pipeline::refine_frames (Frame frm1, Frame &frm2, vector<Point3d> p3d)
{
  cout << "---------------- Refining Frame 2 ----------------" << endl;
  m_start = clock();
  int num_points = frm2.m_triang_matches[frm1.id ()].size();
  v3_t *points = (v3_t*) malloc (num_points*sizeof(v3_t));
  v2_t *projs = (v2_t*) malloc (num_points*sizeof(v2_t));
  cv2bd ( idx2pts (p3d, frm2.m_3d_matches[frm1.id ()]), points);
  cv2bd ( idx2pts (frm2.m_kpts, frm2.m_triang_matches[frm1.id ()]), projs);

  double *R = (double*) malloc (9*sizeof(double));
  double *t = (double*) malloc (9*sizeof(double));
  double *K = (double*) malloc (9*sizeof(double));
  mat_cv2bd(frm2.m_R, R);
  mat_cv2bd(frm2.m_t, t);
  mat_cv2bd(m_K, K);
  camera_params_t params = camera_params_new (R, t, K);

  camera_refine(num_points, points,
                projs, &params, 0, 0);

  m_ends = clock();
  cout << "Refine time : " << (double) (m_ends - m_start) / CLOCKS_PER_SEC << endl;
  for(int i=0; i<3; ++i)
  {
    for(int j=0; j<3; ++j)
    {
      cout << params.R[i*3+j] << " ";
    }
    cout << endl;
  }

  for(int j=0; j<3; ++j)
  {
    cout << params.t[j] << " ";
  }
  cout << endl;

  frm2.m_R = Matx33d (R[0], R[1], R[2],
                      R[3], R[4], R[5],
                      R[6], R[7], R[8]);
  frm2.m_t = Matx31d (t[0], t[1], t[2]);
}

vector<Point3d> Pipeline::local_bundle_adjustement (vector<Frame> &frames,
                                                    vector<Point3d> p3d)
{
  int num_cameras = frames.size ();
  int num_pts = p3d.size ();
  int ncons = 0;

  // Setup visibility mask and measured projections of 3D points
  /*
  int num_projections = 0;
  map< int,vector<int> >::iterator it;
  for (size_t i=0; i<frames.size(); ++i)
    for (it= frames[i].m_2d_matches.begin(); it!= frames[i].m_2d_matches.end(); ++it)
      num_projections += it->second.size();

  char *vmask = new char[num_pts * num_cameras];
  double* projections = new double[2 * num_projections];

  for (int i = 0; i < num_pts * num_cameras; i++)
    vmask[i] = 0;
  */
  /*
  for (int i = 0; i < num_cameras; i++)
  {
    for (size_t j = 0; j < frames[i].m_3d_matches.size(); ++j)
    {
      for (size_t k = 0; k < frames[i].m_3d_matches[j].size(); ++k)
      {
        // For lines frames[i].m_3d_matches, set all the column i to 1.
        int vmask_idx = matAt(num_pts, num_cameras, frames[i].m_3d_matches[j][k], i);
        vmask [vmask_idx] = 1;

        int proj_idx = matAt(num_pts, num_cameras, frames[i].m_3d_matches[j][k], i);
        projections [2 * proj_idx + 0] = idx2pts(frames[i].m_2d_matches[i][k], frames[i].m_kpts).x;
        projections[2 * proj_idx + 1] = idx2pts(frames[i].m_2d_matches[i][k], frames[i].m_kpts).y;
      }
    }
  }
  */
  /*
  // Set focal length and undistort useless stuff
  int est_focal_lenth = 0;
  int const_focal_lenth = 0;
  int undistort = 0;
  int explicit_camera_centers = 1;

  // Setup init params and init pts
  camera_params_t *init_camera_params = new camera_params_t[num_cameras];


  v3_t *init_pts = new v3_t[num_pts];
  for(size_t i=0; i<p3d.size(); ++i)
    init_pts = new v3_t(p3d[i].x, p3d[i].y, p3d[i].z);

  // Set constraints
  int use_constraints = 0;
  int use_point_constraints = 0;
  v3_t *points_constraints = ;
  double point_constraint_weight = ;

  // Set fisheye params (we doooon't caaaaare !)
  int fix_points = 0;
  int optimize_for_fisheye = 0;
  double eps2 = 1.0e-20;

  // WTF !?
  double *Vout= NULL;
  double *Sout= NULL;
  double *Uout= NULL;
  double *Wout= NULL;

  void run_sfm (num_pts, num_cameras, int ncons,
               vmask,
               projections,
               est_focal_length,
               const_focal_length,
               undistort,
               explicit_camera_centers,
               init_camera_params,
               init_pts,
               use_constraints,
               use_point_constraints,
               points_constraints,
               point_constraint_weight,
               fix_points,
               optimize_for_fisheye,
               eps2,
               Vout,
               Sout,
               Uout, Wout);
  */
}

void Pipeline::setK(Mat K)
{
  m_K = K;
  m_Kinv = K.inv();
}

void Pipeline::show_matches(Frame frm1, Frame frm2)
{
  Mat img = M.get_image_matches (frm1.m_img, frm2.m_img,
                                 frm1.m_kpts, frm1.m_matches[frm2.id ()],
                                 frm2.m_kpts, frm2.m_matches[frm1.id ()]);

  namedWindow("Matches");
  imshow("Matches", img);

  waitKey(0);
}

void Pipeline::show_matches_triang(Frame frm1, Frame frm2)
{
  cout << "Showing " << frm1.m_triang_matches[frm2.id ()].size () << " matches." << endl;
  Mat img = M.get_image_matches (frm1.m_img, frm2.m_img,
                                 frm1.m_kpts, frm1.m_triang_matches[frm2.id ()],
                                 frm2.m_kpts, frm2.m_triang_matches[frm1.id ()]);

  namedWindow("Matches");
  imshow("Matches", img);

  waitKey(0);
}

Pipeline::Pipeline()
{}

Pipeline::~Pipeline()
{}
