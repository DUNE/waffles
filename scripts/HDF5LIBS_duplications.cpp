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
  for (auto element : vector)
  {
    os << element << " ";
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

  std::map<std::vector<short>, long int> nodupadc;
  std::vector<std::vector<short>> alladc;

  size_t b_slot, b_crate, b_link;
  bool b_is_stream;
  size_t b_channel_0, b_channel_1, b_channel_2, b_channel_3;

  std::stringstream ssmap;
  // vector<size_t> vlink;
  std::vector<uint16_t> vslot;

  // std::map<size_t, std::map<size_t, std::map<size_t, size_t>>> detmap;
  std::map<std::tuple<size_t, size_t, size_t>, size_t> detmap;

  // String to store each line of the file.
  std::string line;

  if (file_map.is_open())
  {

    while (getline(file_map, line))
    {
      ssmap.clear();
      ssmap.str(line);

      while (ssmap >> sl >> lk >> dpch >> ch)
      {
        detmap[std::make_tuple(sl, lk, dpch)] = ch;
        vslot.push_back(sl);
      }
    }

    file_map.close();
  }
  else
  {

    std::cerr << "Unable to open file!" << std::endl;
    file_map.close();
  }

  std::sort(vslot.begin(), vslot.end());
  auto it = std::unique(vslot.begin(), vslot.end());
  vslot.erase(it, vslot.end());

  const std::string ifile_name = std::string(argv[1]);
  HDF5RawDataFile h5_raw_data_file(ifile_name);

  TLOG() << "\nReading... " << h5_raw_data_file.get_file_name() << "\n"
         << std::endl;

  auto records = h5_raw_data_file.get_all_record_ids();

  std::cout << "\nReading fragments and filling ROOT file... \n";

  for (auto const &record_id : records)
  {
    auto trh_ptr = h5_raw_data_file.get_trh_ptr(record_id);

    std::set<uint64_t> frag_sid_list = h5_raw_data_file.get_geo_ids_for_subdetector(record_id, "HD_PDS");
    int countergeoid = 0;
    for (auto const &geo_id : frag_sid_list)
    {
      uint16_t slot_id = (geo_id >> 32) & 0xffff;
      uint16_t link_id = (geo_id >> 48) & 0xffff;
      std::vector<uint16_t>::iterator it2;
      it2 = std::find(vslot.begin(), vslot.end(), slot_id);

      auto frag_ptr = h5_raw_data_file.get_frag_ptr(record_id, geo_id);

      if (frag_ptr->get_data_size() == 0)
      {
        continue;
      }

      if (DetID::subdetector_to_string(static_cast<DetID::Subdetector>(frag_ptr->get_detector_id())) != "HD_PDS")
        continue;

      int nframes = (frag_ptr->get_size() - sizeof(dunedaq::daqdataformats::FragmentHeader)) / sizeof(dunedaq::fddetdataformats::DAPHNEFrame);
      auto data = frag_ptr->get_data();

      for (size_t i = 0; i < (size_t)nframes; ++i)
      {
        auto fr = reinterpret_cast<dunedaq::fddetdataformats::DAPHNEFrame *>(static_cast<char *>(data) + i * sizeof(dunedaq::fddetdataformats::DAPHNEFrame));
        const auto adcs_per_channel = dunedaq::fddetdataformats::DAPHNEFrame::s_num_adcs;

        b_channel_0 = fr->get_channel();
        b_slot = (fr->daq_header.slot_id);
        b_link = (fr->daq_header.link_id);
        std::tuple<size_t, size_t, size_t> slc = {b_slot, b_link, b_channel_0};
        short ofch = -1;
        if (detmap.find(slc) != detmap.end())
        {
          ofch = detmap[slc];
        }
        if (ofch == -1)
          continue;

        vector<short> tempadc;
        for (size_t j = 0; j < adcs_per_channel; ++j)
        {
          tempadc.push_back(fr->get_adc(j));
        }

        alladc.push_back(tempadc);
        nodupadc[tempadc] = fr->get_timestamp();
        // std::cout << "}" << std::endl;
      }
    }
  }

  auto run_number = h5_raw_data_file.get_attribute<unsigned int>("run_number");
  auto app_name = h5_raw_data_file.get_attribute<std::string>("application_name");
  auto file_index = h5_raw_data_file.get_attribute<unsigned int>("file_index");
  auto creation_timestamp = h5_raw_data_file.get_attribute<std::string>("creation_timestamp");

  cout << "Processed file: " << run_number << "_" << file_index << "_" << app_name << endl;
  cout << "All waveforms (duplications): " << alladc.size() << endl;
  cout << "All waveforms (no duplications): " << nodupadc.size() << endl;
  cout << "Duplications: " << alladc.size() - nodupadc.size() << " (" << 100 * (alladc.size() - nodupadc.size()) / (double)alladc.size() << " %)" << endl;

  TLOG() << "\n\nClosing... " << std::endl;
  return 0;
}
