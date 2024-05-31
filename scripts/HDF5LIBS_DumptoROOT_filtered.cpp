/**
 * @file HDF5TestDumpRecord.cpp
 *
 * Demo of HDF5 file reader for TPC fragments: this example demonstrates
 * simple 'record-dump' functionality.
 *
 * This is part of the DUNE DAQ Software Suite, copyright 2020.
 * Licensing/copyright details are in the COPYING file that you should have
 * received with this code.
 */

#include "hdf5libs/HDF5RawDataFile.hpp"

#include "daqdataformats/Fragment.hpp"
#include "detdataformats/DetID.hpp"
#include "detdataformats/HSIFrame.hpp"
// #include "detdataformats/SourceID.hpp"

#include "logging/Logging.hpp"
#include "hdf5libs/hdf5rawdatafile/Structs.hpp"
#include "hdf5libs/hdf5rawdatafile/Nljs.hpp"
#include "trgdataformats/TriggerObjectOverlay.hpp"

#include "fddetdataformats/DAPHNEFrame.hpp"
#include "fddetdataformats/DAPHNEStreamFrame.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <TROOT.h>
#include <TObject.h>
#include <TChain.h>
#include <TFile.h>
#include <TTree.h>

using namespace dunedaq::hdf5libs;
using namespace dunedaq::daqdataformats;
using namespace dunedaq::detdataformats;
using namespace dunedaq::trgdataformats;

using DAPHNEStreamFrame = dunedaq::fddetdataformats::DAPHNEStreamFrame;
using DAPHNEFrame = dunedaq::fddetdataformats::DAPHNEFrame;

using dunedaq::daqdataformats::FragmentHeader;

// template <typename S>
// std::ostream &operator<<(std::ostream &os,
//                          const std::vector<S> &vector)
// {
//   for (auto element : vector)
//   {
//     os << element << " ";
//   }
//   return os;
// }

template <typename S>
std::ostream &operator<<(std::ostream &os,
                         const std::vector<S> &vector)
{
  // Printing all the elements
  // using <<
  // for (int i = 0; i < vector.size(); i++)
  for (int i = 0; i < 22; i++)
  {
    os << vector[i] << " ";
  }
  return os;
}

void print_usage()
{
  TLOG() << "Usage: HDF5LIBS_DumptoROOT <input_file_name> <channel_map_file>";
}

int main(int argc, char **argv)
{

  if (argc != 3)
  {
    print_usage();
    return 1;
  }

  const std::string string_map = std::string(argv[2]);
  std::ifstream file_map(string_map);
  size_t sl, lk, dpch, ch;
  std::stringstream ssmap;
  // vector<size_t> vlink;
  std::vector<uint16_t> vslot;

  // std::map<size_t, std::map<size_t, std::map<size_t, size_t>>> detmap;
  std::map<std::tuple<size_t, size_t, size_t>, size_t> detmap;

  std::map<std::vector<int>, std::tuple<int, int, int, long int, long int, long int, int, int, int, int, int, int, int, long int, int, int, int, long int>> allval;

  // String to store each line of the file.
  std::string line;

  if (file_map.is_open())
  {
    // Read each line from the file_map and store it in the
    // 'line' variable.
    while (getline(file_map, line))
    {
      ssmap.clear();
      ssmap.str(line);

      while (ssmap >> sl >> lk >> dpch >> ch)
      {
        // detmap[sl][lk][dpch] = ch;
        detmap[std::make_tuple(sl, lk, dpch)] = ch;
        vslot.push_back(sl);
      }
    }
    // Close the file_map stream once all lines have been
    // read.
    file_map.close();
  }
  else
  {
    // Print an error message to the standard error
    // stream if the file cannot be opened.
    std::cerr << "Unable to open file!" << std::endl;
    file_map.close();
  }

  std::sort(vslot.begin(), vslot.end());
  auto it = std::unique(vslot.begin(), vslot.end());
  vslot.erase(it, vslot.end());

  // PRINT MAP VALUES
  // for (auto &dm : detmap)
  // {
  //   std::cout << "MAP[" << std::get<0>(dm.first) << "," << std::get<1>(dm.first) << "," << std::get<2>(dm.first) << "] = " << dm.second << "\t";
  // }

  // std::cout << "\n"
  //           << std::endl;

  const std::string ifile_name = std::string(argv[1]);
  HDF5RawDataFile h5_raw_data_file(ifile_name);

  size_t b_slot, b_crate, b_link;
  bool b_is_stream;
  size_t b_channel_0, b_channel_1, b_channel_2, b_channel_3;

  bool fExportWaveformTree;
  // vars per event
  // clang complained -- commenting out
  // int _SubRun;
  int _Run;
  int _Event;
  int _TriggerNumber;
  long int _TimeStamp;
  long int _Window_end;
  long int _Window_begin;
  int _NFrames;
  int _Slot;
  int _Link;
  int _Crate;
  int _DaphneChannel;
  int _OfflineChannel;
  int _Deltatmst;
  long int _FrameTimestamp;
  short _adc_value[1024] = {-1};
  int _TriggerSampleValue;
  int _Threshold;
  int _Baseline;
  long int _TriggerTimeStamp;
  // open our file reading

  // std::ostringstream ss;

  TLOG() << "\nReading... " << h5_raw_data_file.get_file_name() << "\n"
         << std::endl;
  ;
  // get some file attributes
  auto run_number = h5_raw_data_file.get_attribute<unsigned int>("run_number");
  auto app_name = h5_raw_data_file.get_attribute<std::string>("application_name");
  auto file_index = h5_raw_data_file.get_attribute<unsigned int>("file_index");
  auto creation_timestamp = h5_raw_data_file.get_attribute<std::string>("creation_timestamp");

  // std::vector<std::string> attr = h5_raw_data_file.get_attribute_names();

  // std::cout << attr << std::endl;

  // std::cout << "\n\tCreation timestamp: " << creation_timestamp << std::endl;

  TString appn = app_name;
  int rn = run_number;
  int idxrn = file_index;

  TFile hf(Form("run_%i_%i_%s_decode.root", rn, idxrn, appn.Data()), "recreate");
  hf.mkdir("pdhddaphne");
  hf.cd("pdhddaphne");

  _Run = run_number;

  auto records = h5_raw_data_file.get_all_record_ids();
  size_t frag_header_size = sizeof(FragmentHeader);

  std::cout << "\nReading fragments and filling ROOT file... \n";
  size_t rep = 0;
  int counter = 0;
  for (auto const &record_id : records)
  {
    auto trh_ptr = h5_raw_data_file.get_trh_ptr(record_id);

    // if (rep == trh_ptr->get_header().trigger_timestamp)
    //   continue;

    counter++;
    rep = trh_ptr->get_header().trigger_timestamp;
    // if (counter % 100 == 0)
    // std::cout << counter << " Processing record (" << record_id.first << "," << record_id.second << "):" << std::endl;

    size_t tmstp = trh_ptr->get_header().trigger_timestamp;

    std::set<uint64_t> frag_sid_list = h5_raw_data_file.get_geo_ids_for_subdetector(record_id, "HD_PDS");
    int countergeoid = 0;
    for (auto const &geo_id : frag_sid_list)
    {
      countergeoid++;
      // std::cout << "\n\t\t\tGeo_id: " << countergeoid << " " << geo_id << std::endl;

      uint16_t slot_id = (geo_id >> 32) & 0xffff;
      uint16_t link_id = (geo_id >> 48) & 0xffff;

      std::vector<uint16_t>::iterator it2;
      it2 = std::find(vslot.begin(), vslot.end(), slot_id);

      if (it2 == vslot.end())
      {
        // std::cout << "Slot id " << slot_id << " not found!" << std::endl;
        continue;
      }

      auto frag_ptr = h5_raw_data_file.get_frag_ptr(record_id, geo_id);

      if (frag_ptr->get_data_size() == 0)
      {
        // std::cout << "\n\t\t" << "*** Empty fragment! Moving to next fragment. ***" << std::endl;
        continue;
      }

      if (DetID::subdetector_to_string(static_cast<DetID::Subdetector>(frag_ptr->get_detector_id())) != "HD_PDS")
        continue;

      int nframes = (frag_ptr->get_size() - sizeof(dunedaq::daqdataformats::FragmentHeader)) / sizeof(dunedaq::fddetdataformats::DAPHNEFrame);
      auto data = frag_ptr->get_data();
      // short adcval[1024];

      ComponentRequest cr = trh_ptr->get_component_for_source_id(frag_ptr->get_element_id());
      // std::cout << "\n\t\t"
      // << "Readout window before = " << (trh_ptr->get_trigger_timestamp() - cr.window_begin)
      // << ", after = " << (cr.window_end - trh_ptr->get_trigger_timestamp()) << std::endl;

      // std::cout << "\n\t\t\t\tNumber of frames: " << nframes << std::endl;

      for (size_t i = 0; i < (size_t)nframes; ++i)
      {
        auto fr = reinterpret_cast<dunedaq::fddetdataformats::DAPHNEFrame *>(static_cast<char *>(data) + i * sizeof(dunedaq::fddetdataformats::DAPHNEFrame));
        const auto adcs_per_channel = dunedaq::fddetdataformats::DAPHNEFrame::s_num_adcs;

        b_channel_0 = fr->get_channel();
        b_slot = (fr->daq_header.slot_id);
        b_crate = (fr->daq_header.crate_id);
        b_link = (fr->daq_header.link_id);
        std::tuple<size_t, size_t, size_t> slc = {b_slot, b_link, b_channel_0};

        size_t ofch = 9999;

        if (detmap.find(slc) != detmap.end())
        {
          ofch = detmap[slc];
        }

        _Event = -1;
        _TriggerNumber = -1;
        _TimeStamp = -1;
        _Window_end = -1;
        _Window_begin = -1;
        _NFrames = -1;
        _Slot = -1;
        _Link = -1;
        _Crate = -1;
        _DaphneChannel = -1;
        _OfflineChannel = -1;
        _Deltatmst = -1;
        _FrameTimestamp = -1;
        // _adc_value[1024] = {-1};
        _TriggerSampleValue = -1;
        _Threshold = -1;
        _Baseline = -1;
        _TriggerTimeStamp = -1;
        int deltatmstp = -1;

        if (ofch == 9999)
          continue;

        // std::cout << "\n\t\t\t\t\tNumber of ADC in frame: " << adcs_per_channel << std::endl;
        // std::cout << "\n\t\t\t\t\t\tFrame Timestamp: " << fr->get_timestamp() << std::endl;

        deltatmstp = fr->get_timestamp() - tmstp;

        // std::cout << "\n_OfflineChannel: " << _OfflineChannel << std::endl;
        _Event = _TriggerNumber;
        _TriggerNumber = trh_ptr->get_header().trigger_number;
        _TimeStamp = tmstp;
        _Window_end = cr.window_end;
        _Window_begin = cr.window_begin;
        _NFrames = nframes;
        _Slot = b_slot;
        _Link = b_link;
        _Crate = b_crate;
        _DaphneChannel = b_channel_0;
        _OfflineChannel = ofch;
        _Deltatmst = deltatmstp;
        _FrameTimestamp = fr->get_timestamp();
        _TriggerSampleValue = fr->header.trigger_sample_value;
        _Threshold = fr->header.threshold;
        _Baseline = fr->header.baseline;
        _TriggerTimeStamp = trh_ptr->get_trigger_timestamp();

        // std::cout << "\n\t\t\tVersion:" << unsigned(fr->daq_header.version) << " DetID:" << unsigned(fr->daq_header.det_id) << " CrateID:" << b_crate
        // << " SlotID:" << b_slot << " LinkID:" << b_link
        // << " Timestamp: " << fr->get_timestamp() << " Trigger timestamp: " << tmstp << " delta timestamp: " << deltatmstp << std::endl;
        // std::cout << "\nADC[" << b_channel_0 << " - " << _OfflineChannel << "] - {";
        std::vector<int> adctemp;
        for (size_t j = 0; j < adcs_per_channel; ++j)
        {
          _adc_value[j] = fr->get_adc(j);
          adctemp.push_back(_adc_value[j]);
          // std::cout << fr->get_adc(j) << ", ";
        }
        // std::cout << "}" << std::endl;

        allval[adctemp] = std::make_tuple(_Run, _Event, _TriggerNumber, _TimeStamp, _Window_end, _Window_begin, _NFrames, _Slot, _Link, _Crate, _DaphneChannel, _OfflineChannel, _Deltatmst, _FrameTimestamp, _TriggerSampleValue, _Threshold, _Baseline, _TriggerTimeStamp);
      }
    }
  }

  int _Run_a;
  int _Event_a;
  int _TriggerNumber_a;
  long int _TimeStamp_a;
  long int _Window_end_a;
  long int _Window_begin_a;
  int _NFrames_a;
  int _Slot_a;
  int _Link_a;
  int _Crate_a;
  int _DaphneChannel_a;
  int _OfflineChannel_a;
  int _Deltatmst_a;
  long int _FrameTimestamp_a;
  // short _adc_value_a[1024] = {-1};
  int _TriggerSampleValue_a;
  int _Threshold_a;
  int _Baseline_a;
  long int _TriggerTimeStamp_a;
  std::vector<int> adcvec;

  TTree fWaveformTree("waveforms", "waveforms");
  fWaveformTree.Branch("Run", &_Run_a, "Run/I");
  fWaveformTree.Branch("Event", &_Event_a, "Event/I");
  fWaveformTree.Branch("TriggerNumber", &_TriggerNumber_a, "TriggerNumber/I");
  fWaveformTree.Branch("TimeStamp", &_TimeStamp_a, "TimeStamp/l");
  fWaveformTree.Branch("Window_begin", &_Window_begin_a, "Window_begin/l");
  fWaveformTree.Branch("Window_end", &_Window_end_a, "Window_end/l");

  fWaveformTree.Branch("Slot", &_Slot_a, "Slot/I");
  fWaveformTree.Branch("Crate", &_Crate_a, "Crate/I");
  fWaveformTree.Branch("Link", &_Link_a, "Link/I");
  fWaveformTree.Branch("DaphneChannel", &_DaphneChannel_a, "DaphneChannel/I");
  fWaveformTree.Branch("OfflineChannel", &_OfflineChannel_a, "OfflineChannel/I");
  fWaveformTree.Branch("FrameTimestamp", &_FrameTimestamp_a, "FrameTimestamp/l");
  fWaveformTree.Branch("DeltaTimestamp", &_Deltatmst_a, "DeltaTimestamp/I");
  fWaveformTree.Branch("adc_channel", &adcvec);

  fWaveformTree.Branch("TriggerSampleValue", &_TriggerSampleValue_a, "TriggerSampleValue/I"); // only for self-trigger
  fWaveformTree.Branch("Threshold", &_Threshold_a, "Threshold/I");                            // only for self-trigger
  fWaveformTree.Branch("Baseline", &_Baseline_a, "Baseline/I");
  fWaveformTree.Branch("TriggerTimeStamp", &_TriggerTimeStamp_a, "TriggerTimeStamp/l");

  for (auto &v : allval)
  {
    adcvec = v.first;
    // std::cout << adcvec << "\n " << std::endl;

    _Run_a = std::get<0>(v.second);
    _Event_a = std::get<1>(v.second);
    _TriggerNumber_a = std::get<2>(v.second);
    _TimeStamp_a = std::get<3>(v.second);
    _Window_end_a = std::get<4>(v.second);
    _Window_begin_a = std::get<5>(v.second);
    _NFrames_a = std::get<6>(v.second);
    _Slot_a = std::get<7>(v.second);
    _Link_a = std::get<8>(v.second);
    _Crate_a = std::get<9>(v.second);
    _DaphneChannel_a = std::get<10>(v.second);
    _OfflineChannel_a = std::get<11>(v.second);
    _Deltatmst_a = std::get<12>(v.second);
    _FrameTimestamp_a = std::get<13>(v.second);
    _TriggerSampleValue_a = std::get<14>(v.second);
    _Threshold_a = std::get<15>(v.second);
    _Baseline_a = std::get<16>(v.second);
    _TriggerTimeStamp_a = std::get<17>(v.second);
    fWaveformTree.Fill();

    _Run_a = -1;
    _Event_a = -1;
    _TriggerNumber_a = -1;
    _TimeStamp_a = -1;
    _Window_end_a = -1;
    _Window_begin_a = -1;
    _NFrames_a = -1;
    _Slot_a = -1;
    _Link_a = -1;
    _Crate_a = -1;
    _DaphneChannel_a = -1;
    _OfflineChannel_a = -1;
    _Deltatmst_a = -1;
    _FrameTimestamp_a = -1;
    adcvec.clear();
    _TriggerSampleValue_a = -1;
    _Threshold_a = -1;
    _Baseline_a = -1;
    _TriggerTimeStamp_a = -1;
  }

  std::cout << "\nWritting ROOT file... ";
  fWaveformTree.Write("", TObject::kWriteDelete);
  hf.Close();
  std::cout << "\nReading and writting complete!... \n";
  TLOG() << "\nClosing... " << std::endl;
  return 0;
}
