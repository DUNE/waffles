//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Mon Jun 10 12:08:35 2024 by ROOT version 6.30/06
// from TTree raw_waveforms/raw_waveforms
// found on file: ../../test/run_26683_0_dataflow0_datawriter_0_decode.root
//////////////////////////////////////////////////////////

#ifndef hdf5torootclass_h
#define hdf5torootclass_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.
#include "vector"

class hdf5torootclass {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   UInt_t          record;
   ULong64_t       daq_timestamp;
   vector<short>   *adcs;
   ULong64_t       timestamp;
   Short_t         channel;
   Short_t         baseline;
   Short_t         trigger_sample_value;
   Bool_t          is_fullstream;

   // List of branches
   TBranch        *b_record;   //!
   TBranch        *b_daq_timestamp;   //!
   TBranch        *b_adcs;   //!
   TBranch        *b_timestamp;   //!
   TBranch        *b_channel;   //!
   TBranch        *b_baseline;   //!
   TBranch        *b_trigger_sample_value;   //!
   TBranch        *b_is_fullstream;   //!

   hdf5torootclass(TTree *tree=0);
   virtual ~hdf5torootclass();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef hdf5torootclass_cxx
hdf5torootclass::hdf5torootclass(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("../../test/run_26683_0_dataflow0_datawriter_0_decode.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("../../test/run_26683_0_dataflow0_datawriter_0_decode.root");
      }
      f->GetObject("raw_waveforms",tree);

   }
   Init(tree);
}

hdf5torootclass::~hdf5torootclass()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t hdf5torootclass::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t hdf5torootclass::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void hdf5torootclass::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set object pointer
   adcs = 0;
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("record", &record, &b_record);
   fChain->SetBranchAddress("daq_timestamp", &daq_timestamp, &b_daq_timestamp);
   fChain->SetBranchAddress("adcs", &adcs, &b_adcs);
   fChain->SetBranchAddress("timestamp", &timestamp, &b_timestamp);
   fChain->SetBranchAddress("channel", &channel, &b_channel);
   fChain->SetBranchAddress("baseline", &baseline, &b_baseline);
   fChain->SetBranchAddress("trigger_sample_value", &trigger_sample_value, &b_trigger_sample_value);
   fChain->SetBranchAddress("is_fullstream", &is_fullstream, &b_is_fullstream);
   Notify();
}

Bool_t hdf5torootclass::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void hdf5torootclass::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t hdf5torootclass::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef hdf5torootclass_cxx
