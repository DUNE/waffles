#define hdf5torootclass_cxx
#include "functions/hdf5torootclass.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include "functions/wffunctions2.h"

template <typename S>
ostream &operator<<(ostream &os,
                    const vector<S> &vector)
{
    for (auto element : vector)
    {
        os << element << " ";
    }
    return os;
}

void hdf5torootclass::Loop() {};

// Usage: root 'waveforms.C("file.root",run_number)' -b -q

void plotsAPA(TString inputfile, int run)
{

    // ROOT::EnableThreadSafety();
    // ROOT::EnableImplicitMT(4);

    std::ifstream file_map("functions/channelmap.txt"); // reading channel map
    Short_t sl, lk, dpch, ch;
    std::stringstream ssmap;

    std::map<Short_t, Short_t> detmap;
    std::map<Short_t, Short_t> invdetmap;

    std::string line;

    if (file_map.is_open())
    {
        while (getline(file_map, line))
        {
            ssmap.clear();
            ssmap.str(line);

            while (ssmap >> dpch >> ch)
            {
                detmap[dpch] = ch;
                invdetmap[ch] = dpch;
            }
        }

        file_map.close();
    }
    else
    {
        std::cerr << "Unable to open file!" << std::endl;
        file_map.close();
    }

    wffunctions bs;

    map<TString, TString> filename = {
        {"file1", Form("%s", inputfile.Data())}};

    TFile hf(Form("run_%i.root", run), "recreate"); // create root file with charge and persistence histograms per channel
    // hf.Close();
    hf.mkdir("chargehistos");
    hf.mkdir("persistancehistos");
    hf.mkdir("plots");

    for (auto f : filename)
    {

        TChain *t[] = {NULL};

        t[0] = new TChain();
        t[0]->Add(Form("%s?#raw_waveforms", f.second.Data()));
        // t[0]->SetImplicitMT(true);
        Long64_t nentries = t[0]->GetEntries();
        hdf5torootclass event(t[0]);

        cout << "\nFile open -> " << f.second << "\tentries: " << nentries << endl;

        int hl = -50; // Low Y-axis for persistance plot
        int hh = 300; // High Y-axis for persistance plot

        int binlimlow = 100;  // Low X-axis for persistance plot
        int binlimhigh = 300; // High X-axis for persistance plot

        TH2F *wfpersistenceall[160];
        TH2F *peakchg[160];
        TH1F *chg[160];
        TH1F *maxhisto[160];
        TH1F *maxpeak[160];

        for (int i = 0; i < 160; i++) // create histograms
        {
            wfpersistenceall[i] = new TH2F(Form("persistence_channel_%i", i), Form("persistence_channel_%i", i), (binlimhigh - binlimlow), binlimlow, binlimhigh, (hh - hl), hl, hh);
            chg[i] = new TH1F(Form("charge_channel_%i", i), Form("charge_channel_%i_run", i), 200, -150, 2500);
            maxhisto[i] = new TH1F(Form("max_%i", i), Form("max_%i", i), 200, 100, 300);
            maxpeak[i] = new TH1F(Form("peak_%i", i), Form("peak_%i", i), 200, 0, 200);
            peakchg[i] = new TH2F(Form("peak_vs_chg_%i", i), Form("peak_vs_chg_%i", i), 200, -150, 3000, 200, -15, 300);
        }

        for (auto ievt : ROOT::TSeqUL(nentries)) // loop over entries in root file
        {
            event.GetEntry(ievt);

            if (event.is_fullstream)
                continue;

            bs.setADCvector(event.adcs); // setting the adc vector to use function
            bs.setWindowBaseline(100);   // set window to calculate baseline from 0 to number, this case o to 100 ticks

            int basel = 0;
            if (event.is_fullstream == true)
                basel = std::reduce((event.adcs)->cbegin(), (event.adcs)->cend(), (double)0) / event.adcs->size();
            if (event.is_fullstream == false)
                basel = bs.getLimitBaseline();

            // std::vector<int> adcv; // vector of adc values
            std::vector<std::pair<int, int>> pairs;
            for (int i = 0; i < 1024; i++)
            {
                // adcv.push_back(adc); // filling adc vector
                // int adc = event.adcs[i];
                pairs.push_back(make_pair(i, -event.adcs->at(i)));
            }

            const auto p = max_element(pairs.begin(), pairs.end(), [](const auto &lhs, const auto &rhs)
                                       { return lhs.second < rhs.second; });

            auto label = p->first;
            auto prob = p->second;

            int labelmap = detmap[event.channel];
            // cout << "Channel: " << event.channel << " label map: " << labelmap << endl;
            maxhisto[labelmap]->Fill(label);
            maxpeak[labelmap]->Fill(prob + basel);

            bs.setWindowCharge(132, 155);                             // set window to calculate the charge
            int integ = bs.fillChargeHistogram(chg[labelmap], basel); // filling charge histo

            bs.fillWaveform2D(wfpersistenceall[labelmap], basel); // filling persistence histo

            peakchg[labelmap]->Fill(integ, prob + basel, 1);

            // chg = wfpersistenceall->ProjectionY();
        }

        // new TCanvas("", "", 5000, 5000);
        // maxhisto->Draw("histo");
        // gPad->SaveAs(Form("max_%i.png", run));

        // wfpersistenceall->ProfileX();

        map<int, map<int, TH1 *>> histomap;
        for (int i = 0; i < 160; i++)
        {

            histomap[0][i] = wfpersistenceall[i];
            histomap[1][i] = peakchg[i];
            histomap[2][i] = chg[i];
            histomap[3][i] = maxhisto[i];
            histomap[4][i] = maxpeak[i];
        }

        int histosize = histomap.size();
        const int ncanvas = histosize * 2;
        TCanvas *c[ncanvas];
        TVirtualPad *c1[ncanvas];
        TVirtualPad *c2[ncanvas];

        for (int i = 0; i < ncanvas; i++)
        {
            c[i] = new TCanvas("", "", 8000, 8000);
            c[i]->Divide(2, 1);
            c1[i] = c[i]->cd(1);
            c2[i] = c[i]->cd(2);
            c1[i]->Divide(4, 10, 0.0, 0.0);
            c2[i]->Divide(4, 10, 0.0, 0.0);
        }

        gStyle->SetOptStat(0);
        gStyle->SetTitleSize(0.2, "t");
        // gStyle->SetTitleFontSize(0.6);
        gStyle->SetTitleX(0.5);
        gStyle->SetTitleY(1);

        // TProfile *hprof[160];

        for (int h = 0; h < histomap.size(); h++)
        {
            for (int j = 1; j <= 4; j++) // plotting according map
            {
                for (int i = 1; i <= 10; i++) // plotting according map
                {

                    // histomap[2][i + 39 + 10 * (4 - j)]

                    c1[h]->cd(4 * (i - 1) + j);
                    // chg[i + 39 + 10 * (4 - j)]->SetTitle(Form("Channel: %i", invdetmap[i + 39 + 10 * (4 - j)]));
                    // chg[i + 39 + 10 * (4 - j)]->Draw("histo");
                    histomap[h][i + 39 + 10 * (4 - j)]->SetTitle(Form("Channel: %i", invdetmap[i + 39 + 10 * (4 - j)]));
                    histomap[h][i + 39 + 10 * (4 - j)]->Draw("histo");
                    gPad->SetTopMargin(0.2);
                    c1[h]->Modified();
                    c1[h]->Update();

                    c2[h]->cd(4 * (i - 1) + j);
                    // chg[i - 1 + 10 * (4 - j)]->SetTitle(Form("Channel: %i", invdetmap[i - 1 + 10 * (4 - j)]));
                    // chg[i - 1 + 10 * (4 - j)]->Draw("histo");
                    histomap[h][i - 1 + 10 * (4 - j)]->SetTitle(Form("Channel: %i", invdetmap[i - 1 + 10 * (4 - j)]));
                    histomap[h][i - 1 + 10 * (4 - j)]->Draw("histo");
                    gPad->SetTopMargin(0.2);
                    c2[h]->Modified();
                    c2[h]->Update();

                    c1[h + histosize]->cd(4 * (i - 1) + j);
                    // chg[i + 119 + 10 * (4 - j)]->SetTitle(Form("Channel: %i", invdetmap[i + 119 + 10 * (4 - j)]));
                    // chg[i + 119 + 10 * (4 - j)]->Draw("histo");
                    histomap[2][i + 119 + 10 * (4 - j)]->SetTitle(Form("Channel: %i", invdetmap[i + 119 + 10 * (4 - j)]));
                    histomap[2][i + 119 + 10 * (4 - j)]->Draw("histo");
                    gPad->SetTopMargin(0.2);
                    c1[h + histosize]->Modified();
                    c1[h + histosize]->Update();

                    c2[h + histosize]->cd(4 * (i - 1) + j);
                    // chg[i + 79 + 10 * (4 - j)]->SetTitle(Form("Channel: %i", invdetmap[i + 79 + 10 * (4 - j)]));
                    // chg[i + 79 + 10 * (4 - j)]->Draw("histo");
                    histomap[2][i + 79 + 10 * (4 - j)]->SetTitle(Form("Channel: %i", invdetmap[i + 79 + 10 * (4 - j)]));
                    histomap[2][i + 79 + 10 * (4 - j)]->Draw("histo");
                    gPad->SetTopMargin(0.2);
                    c2[h + histosize]->Modified();
                    c2[h + histosize]->Update();
                }
            }

            hf.cd("plots");
            c[h]->Modified();
            c[h]->Update();
            // c[h]->SaveAs(Form("run_%i_%i_side_1.png", run, h));
            c[h]->Write();

            c[h + histosize]->Modified();
            c[h + histosize]->Update();
            // c[h + histosize]->SaveAs(Form("run_%i_%i_side_2.png", run, h + histosize));
            c[h + histosize]->Write();
        }

        // // new TCanvas();
        // // TProfile *hprof3 = wfpersistenceall[0]->ProfileX();
        // // hprof3->Draw("histo");

        for (int i = 0; i < 160; i++)
        {
            int entriesh = wfpersistenceall[i]->GetEntries();
            if (entriesh != 0)
            {
                hf.cd("persistancehistos");
                wfpersistenceall[i]->Write();
            }
            int entriesc = chg[i]->GetEntries();
            if (entriesc != 0)
            {
                hf.cd("chargehistos");
                chg[i]->Write();
            }
        }
    }
}