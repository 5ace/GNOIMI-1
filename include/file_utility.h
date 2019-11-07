#ifndef NovuGNO_IMI_INCLUDE_FILE_UTILITY_H_
#define NovuGNO_IMI_INCLUDE_FILE_UTILITY_H_

#include <iostream>
#include <iosfwd>
#include <assert.h>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <type_traits>
#include <iomanip>
#include <set>


using namespace std;
/* --------------------------------------------------------------------------*/
/**
 * load features from a file into a vector<T>. The features are listed in one
 * col
 *
 * @param[in] f_name	file name
 *
 * @return		features in vector<T>  
 */
/* ----------------------------------------------------------------------------*/
void split(std::string& s, std::string delim,std::vector< std::string >* ret)
{
 size_t last = 0;
 size_t index=s.find_first_of(delim,last);
 while (index!=std::string::npos)
 {
  ret->push_back(s.substr(last,index-last));
  last=index+1;
  index=s.find_first_of(delim,last);
 }
 if (index-last>0)
 {
  ret->push_back(s.substr(last,index-last));
 }
}
template<typename T>
inline vector<T> load_feature(const string& f_name) {
	ifstream in_f(f_name.c_str());
	assert(in_f.is_open()); 

	vector<T> v_res;
	while(in_f.good()) {
		string cur_line;
		getline(in_f, cur_line);
		if(in_f.eof()) break;

		v_res.push_back(stof(cur_line));
	}
	in_f.close();

	return v_res;
}

template <typename FType, typename IDType>
inline void load_feature_ID_fromFiles(const string& f_list, vector<IDType>& v_ID, vector<vector<FType> >& vv_feature) {
	ifstream in_f(f_list.c_str());
	assert(in_f.is_open()); 

	while(in_f.good()) {
		string cur_line;
		getline(in_f, cur_line);
		if(in_f.eof()) break;

		vector<string> v_temp;
		split(cur_line, " ", &v_temp);

		vector<FType> v_cur_feature = load_feature<FType>(v_temp[0]);
		vv_feature.push_back(v_cur_feature);

		v_ID.push_back(stoi(v_temp[1]));
	}
	in_f.close();
}

inline void load_ID_fromFile(const string& f_list, vector<int>& v_ID) {

    ifstream in_f(f_list.c_str());
    assert(in_f.is_open()); 
    v_ID.clear();

    while(in_f.good()) {
	string cur_line;
	getline(in_f, cur_line);
	if(in_f.eof()) break;

	v_ID.push_back(stoi(cur_line));
    }
    in_f.close();
}

#endif
