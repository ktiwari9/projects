#include "test_pipeline.h"

// TODO
  // Créer la nouvelle structure globale
  // P3D = point 3d
  //        + vecteur des descripteurs
  //        + vecteur des point 2D et ID des images associées
  //        + erreur de reprojection
  // Matcher contre le modèle en création ET contre l'images précédente
  //
  // Retoucher matching pour utiliser descripteurs globaux.
  // Retoucher pose3D2D pour utiliser points 3D globaux.
  // Retoucher triangulation pour ajouter points 3D trouvés à liste globale.
  // Retoucher triangulation pour faire triangulation avec le nouveau points 2D
  // et un ou plusieurs points des 2D qui l'ont triangulé auparavant.
  //
  // Faire marcher bundle !
  //
  // Faire fonction de creation de CloudPoint pour backward compatibilité avec
  // code de mastering opencv
  //
  //
  //
  // Revoir à quoi servent les p2d (int id).
  // Remplir avec les bonnes info le bundle adjustement


int main (int argc, char** argv) {
  //test_matches ();
  setup_test_data ();
  //test_pose2D2D ();
  //test_pose3D2D ();
  //test_triangulation ();
  test_bundle ();
  //test_sfm_two_images ();

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
