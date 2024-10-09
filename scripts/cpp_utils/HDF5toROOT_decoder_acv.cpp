#include "hdf5libs/HDF5RawDataFile.hpp"

#include "daqdataformats/Fragment.hpp"
#include "detdataformats/DetID.hpp"
#include "trgdataformats/TriggerObjectOverlay.hpp"
#include "fddetdataformats/DAPHNEFrame.hpp"
#include "fddetdataformats/DAPHNEStreamFrame.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <TROOT.h>
#include <TObject.h>
#include <TFile.h>
#include <TTree.h>

using namespace dunedaq::hdf5libs;
using namespace dunedaq::daqdataformats;
using namespace dunedaq::detdataformats;
using namespace dunedaq::trgdataformats;

using DAPHNEStreamFrame = dunedaq::fddetdataformats::DAPHNEStreamFrame;
using DAPHNEFrame = dunedaq::fddetdataformats::DAPHNEFrame;

using dunedaq::daqdataformats::FragmentHeader;

//****************************************************
class daphne_record{
  
public:

  daphne_record(){
    run = -1;
    record = -1;
    sequence = -1;
    daq_timestamp = -1;
    _Window_end = -1;
    _Window_begin = -1;
    // _NFrames = -1;
    is_fullstream=false;
    //    waveforms.clear();
    adcs.clear();
    integral.clear();
    //adcs=new std::vector<short>();
    channel    = -1;
    timestamp  = -1;
    baseline   = -1;
    trigger_sample_value = -1;     

  }
  virtual ~daphne_record(){
    //    delete adcs;
  }

  void Copy(const daphne_record& d){
    run             = d.run            ;
    record          = d.record         ;
    sequence          = d.sequence     ;
    daq_timestamp   = d.daq_timestamp  ;
    _Window_end     = d._Window_end    ;
    _Window_begin   = d._Window_begin  ;
    is_fullstream   = d.is_fullstream  ;
    //    waveforms       = d.waveforms      ;
    adcs            = d.adcs;
    integral        = d.integral;
    channel    = d.channel;
    timestamp  = d.timestamp;
    baseline   = d.baseline;
    trigger_sample_value   = d.trigger_sample_value;     
    
  }
  
  UInt_t run;
  UInt_t record;
  UInt_t sequence;
  ULong64_t daq_timestamp;
  ULong64_t _Window_end;
  ULong64_t _Window_begin;
  // int _NFrames;
  bool is_fullstream;
  //  std::vector<waveform> waveforms;
  std::vector<short> adcs;
  std::vector<int> integral; 
  short channel;
  ULong64_t timestamp;
  short trigger_sample_value;
  short baseline;
};
//****************************************************

void CreateTree(TTree& tree, daphne_record& record);
void FillTree(TTree& tree, daphne_record& record);
void FillTPs(DAPHNEFrame::Header& header, DAPHNEFrame::Trailer& trailer);

bool check_douplications = false;

void print_usage(){
  TLOG() << "Usage: HDF5LIBS_DumptoROOT <input_file_name> <nevents(optional)>";
}

std::vector<int>* tt = new std::vector<int>();
std::vector<int>* tt_stream = new std::vector<int>();


//****************************************************
//****************************************************
int main(int argc, char **argv)
{

  if (argc < 2){
    print_usage();
    return 1;
  }
  
  std::map<std::vector<short>, daphne_record> allval;
  const std::string ifile_name = std::string(argv[1]); 
  HDF5RawDataFile h5_raw_data_file(ifile_name);

  int nrecords_to_process = 1e10;
  if (argc==3) nrecords_to_process = std::atoi(argv[2]);

  TLOG() << "\nReading... " << h5_raw_data_file.get_file_name() << "\n" << std::endl;

  int run_number                 = h5_raw_data_file.get_attribute<unsigned int>("run_number");
  TString app_name               = h5_raw_data_file.get_attribute<std::string>("application_name");
  size_t file_index              = h5_raw_data_file.get_attribute<size_t>("file_index");
  std::string creation_timestamp = h5_raw_data_file.get_attribute<std::string>("creation_timestamp");

  std::cout << "index: " << run_number << " " << file_index << " " << app_name.Data() << " " << creation_timestamp << std::endl;
  
  // Create the root file with the tree
  TTree fWaveformTree("raw_waveforms", "raw_waveforms");
  daphne_record* record=NULL;
  TFile hf(Form("run_%i_%i_%s_decode_acv.root", run_number, (unsigned int)file_index, app_name.Data()), "recreate");
  CreateTree(fWaveformTree,*record);
  
  auto records = h5_raw_data_file.get_all_record_ids();

  std::cout << "Number of records: " << records.size() << std::endl;
  
  // ---------- Loop over records
  int irecord=0;
  for (auto const &record_id : records){
    if (irecord%10==0 || irecord == nrecords_to_process) 
      std::cout << "  records processed: " << irecord << std::endl;
    irecord++;
    if (irecord > nrecords_to_process) break;
  
    auto trh_ptr = h5_raw_data_file.get_trh_ptr(record_id);

    size_t timestamp = trh_ptr->get_header().trigger_timestamp;
    std::set<uint64_t> frag_sid_list = h5_raw_data_file.get_geo_ids_for_subdetector(record_id, "HD_PDS");

    // ---------- Loop over geo_ids in that record
    for (auto const &geo_id : frag_sid_list){

      auto frag_ptr = h5_raw_data_file.get_frag_ptr(record_id, geo_id);
      if (frag_ptr->get_data_size() == 0) continue;      
      if (DetID::subdetector_to_string(static_cast<DetID::Subdetector>(frag_ptr->get_detector_id())) != "HD_PDS") continue;

      auto datafrag = frag_ptr->get_data();
      ComponentRequest cr = trh_ptr->get_component_for_source_id(frag_ptr->get_element_id());

      //------- self-trigger fragments ----------
      if (fragment_type_to_string(frag_ptr->get_fragment_type()) == "DAPHNE"){
        const auto adcs_per_channel = DAPHNEFrame::s_num_adcs;

	int nframes = (frag_ptr->get_size() - sizeof(dunedaq::daqdataformats::FragmentHeader)) / sizeof(DAPHNEFrame);
	// ---------- Loop over frames
	for (size_t i = 0; i < (size_t)nframes; ++i){
          auto fr = reinterpret_cast<DAPHNEFrame *>(static_cast<char *>(datafrag) + i * sizeof(DAPHNEFrame));

	  daphne_record record;
          record.run           = run_number;
          record.record        = trh_ptr->get_header().trigger_number;
	  record.sequence      = trh_ptr->get_header().sequence_number;
	  record.daq_timestamp = timestamp;
          record._Window_end   = cr.window_end;
          record._Window_begin = cr.window_begin;
          record.channel       = 100*(100 + fr->daq_header.slot_id)+fr->get_channel();
	  record.baseline      = fr->header.baseline;
          record.timestamp     = fr->get_timestamp();
	  record.is_fullstream = false;
	  record.trigger_sample_value = fr->header.trigger_sample_value;

	  //	  FillTPs(fr->header,fr->trailer);
	  
          std::vector<short> adctemp;
	  adctemp.clear();
	  record.adcs.clear();
	  for (size_t j = 0; j < adcs_per_channel; ++j){
	    if (check_douplications)
	      if (j<20) adctemp.push_back(fr->get_adc(j));
	    record.adcs.push_back(fr->get_adc(j));
	    if (record.integral.size()==0)
	      record.integral.push_back(fr->get_adc(j));
	    else
	      record.integral.push_back(record.integral.back() + fr->get_adc(j)-record.baseline);
	  }
	  if (check_douplications){
	    if (allval.find(adctemp) != allval.end() && adctemp[0] != (pow(2,14)-1)){
	      std::cout << "duplication" << std::endl;
	      std::cout << " channel: " << record.channel << std::endl;
	      for (size_t j = 0; j < adcs_per_channel; ++j)
		std::cout << record.adcs[j] << " " ;
	      std::cout << std::endl;
	      for (size_t j = 0; j < adctemp.size(); ++j)
		std::cout << adctemp[j] << " " ;
	      std::cout << std::endl;
	    
	      continue;
	    }
	  }
	  // Create a map for filtering
	  allval[adctemp] = record;

	  // Add this waveform to the tree
	  FillTree(fWaveformTree, record);
	}
      }
      //------- full streaming gragments ----------
      if (fragment_type_to_string(frag_ptr->get_fragment_type()) == "DAPHNEStream"){
	
	bool first = true;
        short dpch[4] = {-1};

	std::map<short, std::vector<short>> adcstream;
        const auto channels_per_daphne     = DAPHNEStreamFrame::s_channels_per_frame;
        const auto adcs_per_channel_stream = DAPHNEStreamFrame::s_adcs_per_channel;

	daphne_record records[4]; 
	int nframes = (frag_ptr->get_size() - sizeof(dunedaq::daqdataformats::FragmentHeader)) / sizeof(DAPHNEStreamFrame);
	for (size_t i = 0; i < (size_t)nframes; ++i){
          auto fr = reinterpret_cast<DAPHNEStreamFrame *>(static_cast<char *>(datafrag)
							  + i * sizeof(DAPHNEStreamFrame));	  
	  if (first){ 

	    dpch[0] = (short)fr->get_channel0();
            dpch[1] = (short)fr->get_channel1();
            dpch[2] = (short)fr->get_channel2();
            dpch[3] = (short)fr->get_channel3();

	    for (int i=0;i<channels_per_daphne;i++){
	      records[i].run            = run_number;
	      records[i].record         = trh_ptr->get_header().trigger_number;
	      records[i].sequence       = trh_ptr->get_header().sequence_number;
	      records[i].daq_timestamp  = timestamp;
	      records[i]._Window_end    = cr.window_end;
	      records[i]._Window_begin  = cr.window_begin;
	      records[i].channel        = 100*(100 + fr->daq_header.slot_id)+dpch[i];
	      records[i].timestamp      = fr->get_timestamp();
	      //	      records[i].baseline       = fr->header.baseline;
	      records[i].is_fullstream  = true;
	      records[i].trigger_sample_value = 0;
	    }
            first = false;
	  }

          for (size_t k = 0; k < channels_per_daphne; ++k){
	    for (size_t j = 0; j < adcs_per_channel_stream; ++j){
	      if (j<10) adcstream[k].push_back(fr->get_adc(j, k));
	      records[k].adcs.push_back(fr->get_adc(j,k));
	      /*
	      if (record[k].integral.size()==0)
		record[k].integral.push_back(fr->get_adc(j,k));
	      else
		record[k].integral.push_back(record[k].integral.back() + fr->get_adc(j,k)-record[k].baseline);	      
	      */
	    }
          }
	}
      
	for (size_t k = 0; k < channels_per_daphne; ++k){
	  if (allval.find(adcstream[k]) != allval.end()){
	    std::cout << "duplication" << std::endl;
	    continue;
	  }
	  // Create a map for filtering
	  allval[adcstream[k]] = records[k];
	  
	  // Add those waveforms to the tree
	  FillTree(fWaveformTree, records[k]);
	}
      }
    }
  }

  std::cout << "\nWritting ROOT file " << std::endl;
  fWaveformTree.Write("", TObject::kWriteDelete);
  hf.Close();
  return 0;
}

//*****************************************
void FillTree(TTree& tree, daphne_record& record){
//*****************************************  

  std::vector<short>* adcs = &(record.adcs);
  std::vector<int>* integral = &(record.integral);

  tree.SetBranchAddress("is_fullstream", &record.is_fullstream); 
  tree.SetBranchAddress("run",           &record.run);
  tree.SetBranchAddress("record",        &record.record);
  tree.SetBranchAddress("seq",           &record.sequence);        
  tree.SetBranchAddress("daq_timestamp", &record.daq_timestamp); 
  tree.SetBranchAddress("window_begin",  &record._Window_begin);
  tree.SetBranchAddress("window_end",    &record._Window_end);   
  tree.SetBranchAddress("channel",       &record.channel);   
  tree.SetBranchAddress("adcs",          &adcs);
  tree.SetBranchAddress("integral",      &integral);
  tree.SetBranchAddress("timestamp",     &record.timestamp);     
  tree.SetBranchAddress("baseline",      &record.baseline);        
  tree.SetBranchAddress("trigger_sample_value",     &record.trigger_sample_value);

  if (record.is_fullstream){
    tree.SetBranchAddress("t", &tt_stream);
    static bool first=true;
    if (first){
      for (unsigned int i=0;i<adcs->size();i++){
	tt_stream->push_back(i);
      }
      first=false;
    }
  }else{
    tree.SetBranchAddress("t", &tt);
    static bool first=true;
    if (first){
      for (unsigned int i=0;i<adcs->size();i++){
	tt->push_back(i);
      }
      first=false;
    }
  }
  
  
  tree.Fill();
}

//*****************************************
void CreateTree(TTree& tree, daphne_record& record){
//*****************************************  

  std::vector<int> tt;
  tree.Branch("record",        &record.record,         "record/i");
  tree.Branch("seq",           &record.sequence,       "seq/i");
  tree.Branch("daq_timestamp", &record.daq_timestamp,  "daq_timestamp/l");
  tree.Branch("is_fullstream", &record.is_fullstream,  "is_fullstream/O");
  tree.Branch("run",           &record.run,            "run/i");
  tree.Branch("window_begin",  &record._Window_begin,  "Window_begin/l");
  tree.Branch("window_end",    &record._Window_end,    "Window_end/l");
  tree.Branch("channel",       &record.channel,        "channel/S");
  tree.Branch("adcs",          &record.adcs);
  tree.Branch("integral",      &record.integral);
  tree.Branch("baseline",      &record.baseline);
  tree.Branch("t",             &tt);  
  tree.Branch("timestamp",     &record.timestamp,      "timestamp/l");
  tree.Branch("trigger_sample_value", &record.trigger_sample_value, "trigger_sample_value/S"); // only for self-trigger
}

//*****************************************
void FillTPs(DAPHNEFrame::Header& header, DAPHNEFrame::Trailer& trailer){
//*****************************************  


  // Trailer word 1
  std::cout << trailer.num_peak_ub_0 << std::endl;
  std::cout << trailer.num_peak_ob_0 << std::endl;
  std::cout << trailer.charge_0 << std::endl;
  std::cout << trailer.da_0 << std::endl;
  std::cout << trailer.max_peak_0 << std::endl;
  std::cout << trailer.time_peak_0 << std::endl;
  std::cout << trailer.time_pulse_0 << std::endl;
  std::cout << "-------------" << std::endl;
  
}

//*****************************************
//void FillTPTree(TTree& tree, daphne_record& record){
//*****************************************  

/*
  tree.SetBranchAddress("num_peak_ub", &);
  tree.SetBranchAddress("num_peak_ob", &);
  tree.SetBranchAddress("charge", &);
  tree.SetBranchAddress("da", &);

    
  tree.Fill();
}
*/
