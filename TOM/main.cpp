#include "test_pipeline.h"

// TODO
  // Revoir à quoi servent les p2d (int id).
  // Test triangulation
  // Remplir avec les bonnes info le bundle adjustement

int main (int argc, char** argv) {
  //test_matches ();
  //test_pose2D2D ();
  //test_pose3D2D ();
  test_sfm_two_images ();

  /*
  // open camera opencv

  while (1) {
    frame << Camera;
    pl << frame;
  }

  pl.build_object ();
  pl.save_object (argv[2]);
  */
}
