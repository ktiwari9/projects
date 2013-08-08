#include "Frame.h"

using namespace std;
using namespace cv;

int Frame::id()
{
  return m_id;
}

void Frame::clear ()
{
  m_id = 0;
  m_img = Scalar (0);
	m_kpts.clear ();
	m_descs = Scalar (0);
	m_matches.clear ();
	m_R = Matx33d (0, 0, 0,
                 0, 0, 0,
                 0, 0, 0);
  m_t = Matx31d (0, 0, 0);
  m_matches.clear();
  m_triang_matches.clear();
  m_3d_matches.clear();
}

void Frame::remove_from_neighbours(int frame_id)
{
  m_matches.erase(frame_id);
}

vector<Point2d> Frame::get_matched_points(int id)
{
  return idx2pts(m_kpts, m_matches[id]);
}

int Frame::get_corr_3d_pts (int match_idx)
{
  // ... check if it is in triangulated index of keyframe 1 ...
  map< int, vector<int> >::iterator it;
  for (it=m_triang_matches.begin(); it!=m_triang_matches.end(); ++it)
  {
    for (size_t i=0; i<it->second.size(); ++i)
    {
      if ( match_idx == it->second[i] )
      {
        return m_3d_matches[it->first][i];
      }
    }
  }

  return -1;
}

void Frame::add_matches (int id, vector<int>idx)
{
  if (id >= 0 && idx.size () != 0)
  {
    m_matches.insert(pair<int, vector<int> >(id, idx) );
  }
  else
    cout << "Warning: Frame::add_matches: Trying to add matches with nulle size or frame index" << endl;
}

void Frame::add_triang_matches (int id, vector<int>idx)
{
  if (id >= 0 && idx.size () != 0)
  {
    map< int,vector<int> >::iterator it;
    // If frame index already exists, juste push back more values in vector
    it = m_triang_matches.find (id);
    if ( it != m_triang_matches.end ())
    {
      for (size_t i=0; i<idx.size(); ++i)
        it->second.push_back (idx[i]);
    }
    // else insert new pair
    else
      m_triang_matches.insert(pair<int, vector<int> >(id, idx) );
  }
  else
    cout << "Warning: Frame::add_triang_matches: Trying to add matches with nulle size or frame index" << endl;
}

void Frame::add_3d_matches (int id, vector<int> idx)
{
  if (id >= 0 && idx.size () != 0)
  {
    map< int,vector<int> >::iterator it;
    // If frame index already exists, juste push back more values in vector
    it = m_3d_matches.find (id);
    if ( it != m_3d_matches.end ())
    {
      for (size_t i=0; i<idx.size(); ++i)
        it->second.push_back (idx[i]);
    }
    // else insert new pair
    else
      m_3d_matches.insert(pair<int, vector<int> >(id, idx) );
  }
  else
    cout << "Warning: Frame::add_3d_matches: Trying to add matches with nulle size or frame index" << endl;
}

Frame::Frame()
{
  m_R = Matx33d (1, 0, 0,
                 0, 1, 0,
                 0, 0, 1);
  m_t = Matx31d (0, 0, 0);
}

Frame::~Frame()
{}
