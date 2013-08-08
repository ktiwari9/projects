#ifndef DEF_MUTOM_BUNDLER_TOOLS
#define DEF_MUTOM_BUNDLER_TOOLS

#include <iostream>

void printMatx33(double* in)
{
  for(int i=0; i<3; ++i)
  {
    for(int j=0; j<3; ++j)
    {
      cout << in[i*3+j];
    }
    cout << endl;
  }
}

#endif // DEF_MUTOM_BUNDLER_TOOLS

