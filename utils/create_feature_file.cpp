//Second step,creates the vocabulary from the set of features. It can be slow
#include <iostream>
#include <fstream>
#include <vector>
#include <agent_brain.pb.h>
//
#include "vocabulary_creator.h"
// OpenCV
#include <opencv2/core/core.hpp>
#include <unistd.h>
#include <bitset>
#include <stdlib.h>

using namespace std;

class CmdLineParser{int argc; char **argv; public: CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}  bool operator[] ( string param ) {int idx=-1;  for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i;    return ( idx!=-1 ) ;    } string operator()(string param,string defvalue="-1"){int idx=-1;    for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue;   else  return ( argv[  idx+1] ); }};
void exit_message(const char * message) {
    std::cerr << message << std::endl;
    exit(1);
}
vector<cv::Mat> readFeatures(){
    vector<cv::Mat> features;
    agent_brain::slam_data slam_data;
    std::vector<uchar> data_vec;
    uint32_t size;
    unsigned char size_bits[32];
    while (1) {
        // cerr << "hello1" << endl;
        if (read(0,&size_bits, sizeof(size_bits)) < 0) exit_message("Error reading protobuf size.");
        
        // cerr << "hello2" << endl;
        size = stoi(( char * ) size_bits, nullptr, 2);
        cerr << "size: " << size << endl;
        if (size == 0) {
            cerr << "Done reading protobuf data" << endl;
            break;
        }
        uchar data_arr[size];
        if (read(0,&data_arr, size) < 0) exit_message("Error reading protobuf data.");
        
        slam_data.ParseFromArray(data_arr, size);
        // cerr << "hello3" << endl;
        size_t data_size = slam_data.descriptions_size();
        uchar data[data_size][32];
        int row = 0;
        for (string description256: slam_data.descriptions()) {
            for (int i = 0; i < description256.size(); i += 8) {
                string description8 = description256.substr(i,8);
                bitset<8> b(description8);
                data[row][i/8] = ( b.to_ulong() & 0xFF);
            }
            row++;
        }
        // cerr << "hello4" << endl;
        cv::Mat mat(data_size,32,0,&data);
        features.push_back(mat);
        
    }
    
    return features;
}

// ----------------------------------------------------------------------------

int main(int argc,char **argv)
{

    try{
        CmdLineParser cml(argc,argv);
        if (cml["-h"] || argc<3){
            cerr<<"Usage:  features output.fbow [-k k] [-l L] [-t nthreads] [-maxIters <int>:0 default] [-v verbose on]. "<<endl;
            cerr<<"Creates the vocabylary of k^L"<<endl;
            cerr<<"By default, we employ a random selection center without runnning a single iteration of the k means.\n"
                  "As indicated by the authors of the flann library in their paper, the result is not very different from using k-means, but speed is much better\n";
            return -1;
        }


        string desc_name = "deepbit";
        auto features=readFeatures();

        cerr<<"DescName="<<desc_name<<endl;
        fbow::VocabularyCreator::Params params;
        params.k = stoi(cml("-k","10"));
        params.L = stoi(cml("-l","6"));
        params.nthreads=stoi(cml("-t","1"));
        params.maxIters=std::stoi (cml("-maxIters","0"));
        params.verbose=cml["-v"];
        srand(0);
        fbow::VocabularyCreator voc_creator;
        fbow::Vocabulary voc;
        cerr << "Creating a " << params.k << "^" << params.L << " vocabulary..." << endl;
        auto t_start=std::chrono::high_resolution_clock::now();
        voc_creator.create(voc,features,desc_name, params);
        auto t_end=std::chrono::high_resolution_clock::now();
        cerr<<"time="<<double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count())<<" msecs"<<endl;
        cerr<<"nblocks="<<voc.size()<<endl;
        cerr<<"Saving "<<argv[2]<<endl;
        voc.saveToFile(argv[2]);


    }catch(std::exception &ex){
        cerr<<ex.what()<<endl;
    }

    return 0;
}
