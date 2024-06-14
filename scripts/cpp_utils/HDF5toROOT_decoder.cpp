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
using std::cout;
using std::data;
using std::endl;
using std::map;
using std::vector;

template <typename S>
std::ostream &operator<<(std::ostream &os,
                         const std::vector<S> &vector)
{
  // Printing all the elements
  // using <<
  // for (int i = 0; i < vector.size(); i += 10000)
  for (int i = 0; i < 20; i++)
  {
    os << vector[i] << " ";
  }
  return os;
}

void print_usage()
{
  TLOG() << "Usage: HDF5toROOT_decoder <input_file_name>";
}

int main(int argc, char **argv)
{

  if (argc != 2)
  {
    print_usage();
    return 1;
  }

  std::map<std::vector<short>, std::tuple<unsigned int, unsigned int, unsigned int, long int, long int, long int, short, short, short, short, short, long int, bool, int, int, int>> allval;

  const std::string ifile_name = std::string(argv[1]);
  HDF5RawDataFile h5_raw_data_file(ifile_name);

  size_t b_slot, b_crate, b_link;
  bool b_is_stream;
  size_t b_channel_0;

  int _Run;
  int _Event;
  int _TriggerNumber;
  long int _TimeStamp;
  long int _Window_end;
  long int _Window_begin;
  int _Record;
  // int _NFrames;
  short _Slot;
  short _Link;
  short _Crate;
  short _DaphneChannel;
  short _OfflineChannel;
  // int _Deltatmst;
  long int _FrameTimestamp;
  // short _adc_value[1024] = {-1};
  short _TriggerSampleValue;
  // int _Threshold;
  int _Baseline;
  // long int _TriggerTimeStamp;

  TLOG() << "\nReading... " << h5_raw_data_file.get_file_name() << "\n"
         << std::endl;

  auto run_number = h5_raw_data_file.get_attribute<unsigned int>("run_number");
  auto app_name = h5_raw_data_file.get_attribute<std::string>("application_name");
  auto file_index = h5_raw_data_file.get_attribute<unsigned int>("file_index");
  std::string creation_timestamp = h5_raw_data_file.get_attribute<std::string>("creation_timestamp");

  TString appn = app_name;
  int rn = run_number;
  int idxrn = file_index;

  TFile hf(Form("run_%i_%i_%s_decode.root", rn, idxrn, appn.Data()), "recreate");
  // hf.mkdir("pdhddaphne");
  // hf.cd("pdhddaphne");

  auto records = h5_raw_data_file.get_all_record_ids();
  unsigned int nrec = records.size();
  size_t frag_header_size = sizeof(FragmentHeader);

  std::cout << "\nReading fragments and filling ROOT file... \n";
  size_t rep = 0;
  int counter = 0;
  int recordnum = 0;
  for (auto const &record_id : records)
  {
    recordnum = record_id.first;
    auto trh_ptr = h5_raw_data_file.get_trh_ptr(record_id);

    counter++;
    rep = trh_ptr->get_header().trigger_timestamp;

    size_t tmstp = trh_ptr->get_header().trigger_timestamp;

    std::set<uint64_t> frag_sid_list = h5_raw_data_file.get_geo_ids_for_subdetector(record_id, "HD_PDS");
    int countergeoid = 0;
    for (auto const &geo_id : frag_sid_list)
    {

      auto frag_ptr = h5_raw_data_file.get_frag_ptr(record_id, geo_id);

      if (frag_ptr->get_data_size() == 0)
      {
        continue;
      }

      if (DetID::subdetector_to_string(static_cast<DetID::Subdetector>(frag_ptr->get_detector_id())) != "HD_PDS")
        continue;

      auto datafrag = frag_ptr->get_data();

      int nframes = (fragment_type_to_string(frag_ptr->get_fragment_type()) == "DAPHNE") ? (frag_ptr->get_size() - sizeof(dunedaq::daqdataformats::FragmentHeader)) / sizeof(dunedaq::fddetdataformats::DAPHNEFrame) : (frag_ptr->get_size() - sizeof(dunedaq::daqdataformats::FragmentHeader)) / sizeof(dunedaq::fddetdataformats::DAPHNEStreamFrame);

      ComponentRequest cr = trh_ptr->get_component_for_source_id(frag_ptr->get_element_id());

      if (fragment_type_to_string(frag_ptr->get_fragment_type()) == "DAPHNE")
      {
        const auto adcs_per_channel = dunedaq::fddetdataformats::DAPHNEFrame::s_num_adcs;
        for (size_t i = 0; i < (size_t)nframes; ++i)
        {
          auto fr = reinterpret_cast<dunedaq::fddetdataformats::DAPHNEFrame *>(static_cast<char *>(datafrag) + i * sizeof(dunedaq::fddetdataformats::DAPHNEFrame));

          b_channel_0 = fr->get_channel();
          b_slot = (fr->daq_header.slot_id);
          b_crate = (fr->daq_header.crate_id);
          b_link = (fr->daq_header.link_id);
          std::tuple<size_t, size_t, size_t> slc = {b_slot, b_link, b_channel_0};

          short ofch = -1;

          _Run = -1;
          _Event = -1;
          _TriggerNumber = -1;
          _TimeStamp = -1;
          _Window_end = -1;
          _Window_begin = -1;
          _Record = -1;
          // _NFrames = -1;
          _Slot = -1;
          _Link = -1;
          _Crate = -1;
          _DaphneChannel = -1;
          _OfflineChannel = -1;
          // _Deltatmst = -1;
          // int deltatmstp = -1;
          _FrameTimestamp = -1;
          _TriggerSampleValue = -1;
          // _Threshold = -1;
          _Baseline = -1;
          // _TriggerTimeStamp = -1;

          // if (ofch == -1)
          //   continue;

          _Run = run_number;
          _TriggerNumber = trh_ptr->get_header().trigger_number;
          _Event = _TriggerNumber;
          _TimeStamp = tmstp;
          _Window_end = cr.window_end;
          _Window_begin = cr.window_begin;
          _Record = recordnum;
          // _NFrames = nframes;
          _Slot = b_slot;
          _Link = b_link;
          _Crate = b_crate;
          _DaphneChannel = b_channel_0;
          _OfflineChannel = ofch;
          // deltatmstp = fr->get_timestamp() - tmstp;
          // _Deltatmst = deltatmstp;
          _FrameTimestamp = fr->get_timestamp();
          _TriggerSampleValue = fr->header.trigger_sample_value;
          // _Threshold = fr->header.threshold;
          _Baseline = fr->header.baseline;
          // _TriggerTimeStamp = trh_ptr->get_trigger_timestamp();

          std::vector<short> adctemp;
          for (size_t j = 0; j < adcs_per_channel; ++j)
          {
            // _adc_value[j] = fr->get_adc(j);
            adctemp.push_back(fr->get_adc(j));
            // std::cout << fr->get_adc(j) << ", ";
          }
          // std::cout << "}" << std::endl;
          bool is_stream = false;

          // allval[adctemp] = std::make_tuple(_Run, _Event, _TriggerNumber, _TimeStamp, _Window_end, _Window_begin, _Slot, _Link, _Crate, _DaphneChannel, _OfflineChannel, _FrameTimestamp, is_stream);

          allval[adctemp] = std::make_tuple(_Run, _Event, _TriggerNumber, _TimeStamp, _Window_end, _Window_begin, _Slot, _Link, _Crate, _DaphneChannel, _OfflineChannel, _FrameTimestamp, is_stream, _Baseline, _TriggerSampleValue, _Record);
        }
      }
      if (fragment_type_to_string(frag_ptr->get_fragment_type()) == "DAPHNEStream")
      {
        bool print = true;
        short dpch[4] = {-1};
        short slotstream[4] = {-1};
        short linkstream[4] = {-1};
        short cratestream[4] = {-1};
        short tmstmpstream[4] = {-1};
        int samplevaluestream[4] = {-1};
        int baselinestream[4] = {-1};

        map<short, vector<short>> adcstream;
        const auto channels_per_daphne = dunedaq::fddetdataformats::DAPHNEStreamFrame::s_channels_per_frame;
        const auto adcs_per_channel_stream = dunedaq::fddetdataformats::DAPHNEStreamFrame::s_adcs_per_channel;
        for (size_t i = 0; i < (size_t)nframes; ++i)
        {
          auto frq = reinterpret_cast<dunedaq::fddetdataformats::DAPHNEStreamFrame *>(static_cast<char *>(datafrag) + i * sizeof(dunedaq::fddetdataformats::DAPHNEStreamFrame));

          if (print == true)
          {
            // cout << "\t\tChannels 0: " << (short)frq->header.channel_0 << "\t1: " << (short)frq->header.channel_1 << "\t2: " << (short)frq->header.channel_2 << "\t3: " << (short)frq->header.channel_3 << endl;
            dpch[0] = (short)frq->get_channel0();
            dpch[1] = (short)frq->get_channel1();
            dpch[2] = (short)frq->get_channel2();
            dpch[3] = (short)frq->get_channel3();

            slotstream[0] = (short)(frq->daq_header.slot_id);
            slotstream[1] = (short)(frq->daq_header.slot_id);
            slotstream[2] = (short)(frq->daq_header.slot_id);
            slotstream[3] = (short)(frq->daq_header.slot_id);

            linkstream[0] = (short)(frq->daq_header.link_id);
            linkstream[1] = (short)(frq->daq_header.link_id);
            linkstream[2] = (short)(frq->daq_header.link_id);
            linkstream[3] = (short)(frq->daq_header.link_id);

            cratestream[0] = (short)(frq->daq_header.crate_id);
            cratestream[1] = (short)(frq->daq_header.crate_id);
            cratestream[2] = (short)(frq->daq_header.crate_id);
            cratestream[3] = (short)(frq->daq_header.crate_id);

            tmstmpstream[0] = (short)(frq->get_timestamp());
            tmstmpstream[1] = (short)(frq->get_timestamp());
            tmstmpstream[2] = (short)(frq->get_timestamp());
            tmstmpstream[3] = (short)(frq->get_timestamp());

            samplevaluestream[0] = -1;
            samplevaluestream[1] = -1;
            samplevaluestream[2] = -1;
            samplevaluestream[3] = -1;

            baselinestream[0] = -1;
            baselinestream[1] = -1;
            baselinestream[2] = -1;
            baselinestream[3] = -1;

            print = false;
          }

          for (size_t k = 0; k < channels_per_daphne; ++k)
          {
            for (size_t j = 0; j < adcs_per_channel_stream; ++j)
            {
              // adcvect.push_back(frq->get_adc(j, k));
              // adcstream[adcs_per_channel_stream * i + j] = frq->get_adc(j, k);
              adcstream[k].push_back(frq->get_adc(j, k));
            }
          }
        }
        for (size_t k = 0; k < channels_per_daphne; ++k)
        {

          b_channel_0 = dpch[k];
          b_slot = slotstream[k];
          b_crate = cratestream[k];
          b_link = linkstream[k];

          std::tuple<size_t, size_t, size_t> slc = {b_slot, b_link, b_channel_0};

          short ofch = -1;

          // if (detmap.find(slc) != detmap.end())
          // {
          //   ofch = detmap[slc];
          // }

          _Run = -1;
          _Event = -1;
          _TriggerNumber = -1;
          _TimeStamp = -1;
          _Window_end = -1;
          _Window_begin = -1;
          _Record = -1;
          // _NFrames = -1;
          _Slot = -1;
          _Link = -1;
          _Crate = -1;
          _DaphneChannel = -1;
          _OfflineChannel = -1;
          // _Deltatmst = -1;
          _FrameTimestamp = -1;
          // _adc_value[1024] = {-1};
          _TriggerSampleValue = -1;
          // _Threshold = -1;
          _Baseline = -1;
          // _TriggerTimeStamp = -1;
          // int deltatmstp = -1;

          // cout << "\tslot: " << b_slot << " link: " << b_link << " dpCH: " << b_channel_0 << " offlinechannel: " << ofch << " vector: " << adcstream[k] << " size: " << adcstream[k].size() << endl;

          // if (ofch == -1)
          //   continue;

          _Run = run_number;
          _TriggerNumber = trh_ptr->get_header().trigger_number;
          _Event = _TriggerNumber;
          _TimeStamp = tmstp;
          _Window_end = cr.window_end;
          _Window_begin = cr.window_begin;
          _Record = recordnum;
          // _NFrames = nframes;
          _Slot = b_slot;
          _Link = b_link;
          _Crate = b_crate;
          _DaphneChannel = b_channel_0;
          _OfflineChannel = ofch;
          // deltatmstp = fr->get_timestamp() - tmstp;
          // _Deltatmst = deltatmstp;
          _FrameTimestamp = tmstmpstream[k];
          _TriggerSampleValue = samplevaluestream[k];
          // _Threshold = fr->header.threshold;
          _Baseline = baselinestream[k];
          // _TriggerTimeStamp = trh_ptr->get_trigger_timestamp();
          bool is_stream = true;

          // allval[adcstream[k]] = std::make_tuple(_Run, _Event, _TriggerNumber, _TimeStamp, _Window_end, _Window_begin, _Slot, _Link, _Crate, _DaphneChannel, _OfflineChannel, _FrameTimestamp, is_stream);

          allval[adcstream[k]] = std::make_tuple(_Run, _Event, _TriggerNumber, _TimeStamp, _Window_end, _Window_begin, _Slot, _Link, _Crate, _DaphneChannel, _OfflineChannel, _FrameTimestamp, is_stream, _Baseline, _TriggerSampleValue, _Record);
        }
      }
    }
  }

  unsigned int _Run_a;
  unsigned int _Event_a;
  unsigned int _TriggerNumber_a;
  long int _TimeStamp_a;
  long int _DAQ_TimeStamp_a;
  long int _Window_end_a;
  long int _Window_begin_a;
  int _Record_a;
  // int _NFrames_a;
  short _Slot_a;
  short _Link_a;
  short _Crate_a;
  short _DaphneChannel_a;
  short _OfflineChannel_a;
  short _Channel_a;
  // int _Deltatmst_a;
  long int _FrameTimestamp_a;
  short _TriggerSampleValue_a;
  // int _Threshold_a;
  short _Baseline_a;
  long int _TriggerTimeStamp_a;
  std::vector<short> adcvec;
  bool isstream;

  TTree fWaveformTree("raw_waveforms", "raw_waveforms");

  fWaveformTree.Branch("record", &_Record_a, "record/i");
  fWaveformTree.Branch("daq_timestamp", &_DAQ_TimeStamp_a, "daq_timestamp/l");
  fWaveformTree.Branch("adcs", &adcvec);
  fWaveformTree.Branch("timestamp", &_TimeStamp_a, "timestamp/l");
  fWaveformTree.Branch("channel", &_Channel_a, "channel/S");
  fWaveformTree.Branch("baseline", &_Baseline_a, "baseline/S");
  fWaveformTree.Branch("trigger_sample_value", &_TriggerSampleValue_a, "trigger_sample_value/S"); // only for self-trigger
  fWaveformTree.Branch("is_fullstream", &isstream, "is_fullstream/O");

  vector<short> ep;
  unsigned int th;

  for (auto &v : allval)
  {
    // std::cout << adcvec << "\n " << std::endl;

    _Record_a = std::get<15>(v.second);
    _DAQ_TimeStamp_a = std::get<3>(v.second);
    adcvec = v.first;
    _TimeStamp_a = std::get<11>(v.second);
    _Channel_a = 100 * (100 + std::get<6>(v.second)) + std::get<9>(v.second);
    _Baseline_a = std::get<13>(v.second);
    _TriggerSampleValue_a = std::get<14>(v.second);
    isstream = std::get<12>(v.second);

    fWaveformTree.Fill();

    _Run_a = -1;
    _Event_a = -1;
    _TriggerNumber_a = -1;
    _TimeStamp_a = -1;
    _Window_end_a = -1;
    _Window_begin_a = -1;
    // _NFrames_a = -1;
    _Slot_a = -1;
    _Link_a = -1;
    _Crate_a = -1;
    _DaphneChannel_a = -1;
    _OfflineChannel_a = -1;
    // _Deltatmst_a = -1;
    _FrameTimestamp_a = -1;
    _Record_a = -1;
    _DAQ_TimeStamp_a = -1;
    _Channel_a = -1;
    _Baseline_a = -1;
    _TriggerSampleValue_a = -1;
    adcvec.clear();
    // isstream = std::get<12>(v.second);

    // _Threshold_a = -1;
    // _Baseline_a = -1;
    // _TriggerTimeStamp_a = -1;
  }
  sort(ep.begin(), ep.end());
  auto itep = unique(ep.begin(), ep.end());
  ep.erase(itep, ep.end());

  unsigned int _ticks_to_nsec = 16;
  unsigned int _adcs_to_nvolts = 292986; //(1.5*3.2)/(2^(14)-1);
  ULong64_t date;
  std::stringstream ssdate;
  ssdate << creation_timestamp;
  ssdate >> date;
  // vector<short> _endpoint;
  TTree metadata("metadata", "metadata");
  // metadata.Branch("record", &_Record_a, "record/i");
  metadata.Branch("endpoint", &ep);
  metadata.Branch("threshold", &th, "threshold/i");
  metadata.Branch("run", &run_number, "run/i");
  metadata.Branch("nrecords", &nrec, "nrecords/i");
  metadata.Branch("date", &date, "date/l");
  metadata.Branch("ticks_to_nsec", &_ticks_to_nsec, "ticks_to_nsec/i");
  metadata.Branch("adcs_to_nvolts", &_adcs_to_nvolts, "adcs_to_nvolts/i");

  metadata.Fill();

  std::cout << "\nWritting ROOT file... ";
  fWaveformTree.Write("", TObject::kWriteDelete);
  // metadata.Write("", TObject::kWriteDelete);

  hf.Close();
  std::cout << "\nReading and writting complete!... \n";
  TLOG() << "\nClosing... " << std::endl;
  return 0;
}
