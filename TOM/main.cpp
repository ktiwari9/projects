#include "test_pipeline.h"

// TODO
  // Faire des fonctions dans Image.cpp qui prennent en argument une autre
  // image et pas juste un id.
  // Revoir à quoi servent les p2d (int id).
  // Tester également le pruning avec F.

int main (int argc, char** argv) {
  //test_matches ();
  //test_pose2D2D ();
  test_pose3D2D ();
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
