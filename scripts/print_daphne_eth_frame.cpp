#include "hdf5libs/HDF5RawDataFile.hpp"

#include "daqdataformats/Fragment.hpp"
#include "daqdataformats/FragmentHeader.hpp"
#include "daqdataformats/FragmentType.hpp"
#include "fddetdataformats/DAPHNEEthStreamFrame.hpp"

#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace dd = dunedaq::daqdataformats;
namespace fd = dunedaq::fddetdataformats;
namespace h5 = dunedaq::hdf5libs;

void print_daq_header(const dunedaq::detdataformats::DAQEthHeader& header)
{
  std::cout << "DAQEthHeader\n"
            << "  version     : " << static_cast<unsigned>(header.version) << '\n'
            << "  det_id      : " << static_cast<unsigned>(header.det_id) << '\n'
            << "  crate_id    : " << static_cast<unsigned>(header.crate_id) << '\n'
            << "  slot_id     : " << static_cast<unsigned>(header.slot_id) << '\n'
            << "  stream_id   : " << static_cast<unsigned>(header.stream_id) << '\n'
            << "  reserved    : " << static_cast<unsigned>(header.reserved) << '\n'
            << "  sequence_id : " << static_cast<unsigned>(header.seq_id) << '\n'
            << "  block_length: " << static_cast<unsigned>(header.block_length) << '\n'
            << "  timestamp   : " << header.get_timestamp() << '\n';
}

void print_channel_words(const fd::DAPHNEEthStreamFrame::Header& header)
{
  for (int ch = 0; ch < fd::DAPHNEEthStreamFrame::s_num_channels; ++ch) {
    const auto& cw = header.channel_words[ch];
    std::cout << "ChannelWord[" << ch << "]\n"
              << "  tbd     : 0x" << std::hex << std::setw(13) << std::setfill('0') << cw.tbd << std::dec
              << std::setfill(' ') << '\n'
              << "  version : " << static_cast<unsigned>(cw.version) << '\n'
              << "  channel : " << static_cast<unsigned>(cw.channel) << '\n';
  }
}

void print_adc_words(const fd::DAPHNEEthStreamFrame& frame)
{
  using word_t = fd::DAPHNEEthStreamFrame::word_t;
  constexpr int words = fd::DAPHNEEthStreamFrame::s_num_adc_words;
  std::cout << "ADC payload (" << words << " 64-bit words)\n";
  for (int i = 0; i < words; ++i) {
    word_t value = frame.adc_words[i];
    std::cout << "  adc_words[" << std::setw(3) << std::setfill(' ') << i << "] = 0x" << std::hex << std::setw(16)
              << std::setfill('0') << value << std::dec << std::setfill(' ') << '\n';
  }
}

int main(int argc, char* argv[])
{
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path-to-hdf5-file>\n";
    return 1;
  }

  try {
    const std::string input_path = argv[1];
    h5::HDF5RawDataFile h5file(input_path);

    h5::HDF5RawDataFile::record_id_set record_ids =
      h5file.is_trigger_record_type() ? h5file.get_all_trigger_record_ids() : h5file.get_all_timeslice_ids();
    if (record_ids.empty()) {
      throw std::runtime_error("No records found in file " + input_path);
    }
    const auto first_record = *record_ids.begin();

    auto fragment_paths = h5file.get_fragment_dataset_paths(first_record);
    if (fragment_paths.empty()) {
      throw std::runtime_error("No fragments found for first record");
    }

    std::unique_ptr<dd::Fragment> fragment;
    std::string chosen_dataset;
    for (const auto& path : fragment_paths) {
      auto candidate = h5file.get_frag_ptr(path);
      if (candidate->get_fragment_type() == dd::FragmentType::kDAPHNEEthStream) {
        fragment = std::move(candidate);
        chosen_dataset = path;
        break;
      }
      if (!fragment) {
        fragment = std::move(candidate);
        chosen_dataset = path;
      }
    }

    if (!fragment) {
      throw std::runtime_error("Unable to load fragment data");
    }

    if (fragment->get_fragment_type() != dd::FragmentType::kDAPHNEEthStream) {
      std::cerr << "Warning: first fragment in file is "
                << dd::fragment_type_to_string(fragment->get_fragment_type())
                << ", not DAPHNEEthStream. Dumping its payload anyway.\n";
    }

    if (fragment->get_data_size() < sizeof(fd::DAPHNEEthStreamFrame)) {
      throw std::runtime_error("Fragment payload smaller than a single DAPHNEEthStreamFrame");
    }

    const auto* frame = reinterpret_cast<const fd::DAPHNEEthStreamFrame*>(fragment->get_data());

    std::cout << "File              : " << input_path << '\n'
              << "Record            : (" << first_record.first << ", " << first_record.second << ")\n"
              << "Fragment dataset  : " << chosen_dataset << '\n'
              << "Fragment type     : " << dd::fragment_type_to_string(fragment->get_fragment_type()) << '\n'
              << "Fragment size (B) : " << fragment->get_size() << '\n'
              << "Payload bytes     : " << fragment->get_data_size() << '\n'
              << "Frame bytes       : " << sizeof(fd::DAPHNEEthStreamFrame) << '\n';

    print_daq_header(frame->daq_header);
    print_channel_words(frame->header);
    print_adc_words(*frame);
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << '\n';
    return 2;
  }

  return 0;
}
