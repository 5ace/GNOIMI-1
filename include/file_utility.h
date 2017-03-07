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


#include <boost/algorithm/string.hpp>
#include <boost/tokenizer.hpp>

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
template<typename T>
inline vector<T> load_feature(const string& f_name) {
	ifstream in_f(f_name.c_str());
	assert(in_f.is_open()); 

	vector<T> v_res;
	while(in_f.good()) {
		string cur_line;
		getline(in_f, cur_line);
		boost::algorithm::trim(cur_line);
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
		boost::algorithm::trim(cur_line);
		if(in_f.eof()) break;

		vector<string> v_temp;
		boost::split(v_temp, cur_line, boost::is_any_of(" "));

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
	boost::algorithm::trim(cur_line);
	if(in_f.eof()) break;

	v_ID.push_back(stoi(cur_line));
    }
    in_f.close();
}

#endif
