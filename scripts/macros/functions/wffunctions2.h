#include <iostream>
#include <TH1.h>

class wffunctions
{
public:
    void setADCvector(std::vector<short> *adcv);
    // void setADChisto(TH1 *ha);
    // void setADChisto2D(TH2 *ha);
    void setWindowBaseline(int ll, int hh);
    void setWindowFilter(int ll, int hh);
    void setWindowBaseline(int ll);
    void setWindowCharge(int ll, int hh);
    void setBaseline(int bb);

    int getAverageBaseline(void);
    int getWindowBaseline(void);
    int getLimitBaseline(void);
    TH1F *getWaveform(void);
    void fillWaveform2D(TH2 *ha, int bs);
    void fillWaveform(TH1 *ha, int bs);
    bool filterleft(int adcutup, int adcutdw, int bsl);
    bool filterright(int adcutup, int adcutdw, int bsl);
    int fillChargeHistogram(TH1 *histo, int bs);
    void fillBaselineHisto(TH1 *histo, int bs);

private:
    std::vector<short> adc;
    // TH1F *histo;
    int low = -1;
    int high = -1;
    int lowfilter = -1;
    int highfilter = -1;
    int lowcharge = -1;
    int highcharge = -1;
    int baseline = -1;
    int averagebaseline = -1;
    int windowbaseline = -1;
    int limitbaseline = -1;
};

void wffunctions::setADCvector(std::vector<short> *adcv)
{
    adc = *adcv;
}

void wffunctions::setWindowBaseline(int ll, int hh)
{
    low = ll;
    high = hh;
    // return 0;
}

void wffunctions::setWindowFilter(int ll, int hh)
{
    lowfilter = ll;
    highfilter = hh;
    // return 0;
}

void wffunctions::setWindowCharge(int ll, int hh)
{
    lowcharge = ll;
    highcharge = hh;
    // return 0;
}
void wffunctions::setWindowBaseline(int ll)
{
    low = ll;
    // high = hh;
    // return 0;
}
void wffunctions::setBaseline(int bb)
{
    baseline = bb;
}

int wffunctions::getAverageBaseline(void)
{
    int adcsum = 0;
    for (auto &bs : adc)
    {
        adcsum += bs;
    }
    averagebaseline = adcsum / adc.size();
    return averagebaseline;
}

int wffunctions::getWindowBaseline(void)
{
    if (low == -1 || high == -1)
    {
        cerr << "\n\nWindow not define! First define setWindowBaseline(int first, int last)\n"
             << endl;
        exit(0);
    }

    int adcsum = 0;
    int window = 0;
    for (int i = 0; i < adc.size(); i++)
    {
        if (i <= low || i >= high)
        {
            adcsum += adc[i];
            ++window;
        }
    }
    windowbaseline = adcsum / window;
    return windowbaseline;
}

int wffunctions::getLimitBaseline(void)
{
    if (low == -1 || high > -1)
    {
        cerr << "\n\nWindow not define! First define setWindowBaseline(int first)\n"
             << endl;
        exit(0);
    }

    int adcsum = 0;
    int window = 0;
    for (int i = 0; i < adc.size(); i++)
    {
        if (i <= low)
        {
            adcsum += adc[i];
            ++window;
        }
    }
    limitbaseline = adcsum / window;
    return limitbaseline;
}

void wffunctions::fillWaveform(TH1 *histo, int bs)
{
    for (int i = 0; i < adc.size(); i++)
    {
        // if()
        histo->Fill(i, -(adc[i] - bs));
    }
}

void wffunctions::fillBaselineHisto(TH1 *histo, int bs)
{
    for (int i = 0; i < adc.size(); i++)
    {
        if (i <= 100)
        {
            histo->Fill(-adc[i] + bs);
        }
    }
}

int wffunctions::fillChargeHistogram(TH1 *histo, int bs)
{
    int charge = 0;
    for (int i = 0; i < adc.size(); i++)
    {
        if (i <= highcharge && i >= lowcharge)
        {
            charge = charge - (adc[i] - bs);
            // cout << "ADC " << i << ": " << -(adc[i] - bs) << " charge: " << charge << endl;
        }
    }
    histo->Fill(charge);

    return charge;
}

void wffunctions::fillWaveform2D(TH2 *histo2D, int bs)
{
    for (int i = 0; i < adc.size(); i++)
    {

        histo2D->Fill(i, -(adc[i] - bs), 1);
    }
}

bool wffunctions::filterleft(int adcutlw, int adcuthg, int bsl)
{

    if (lowfilter == -1 || highfilter == -1)
    {
        cerr << "\n\nWindow not define! First define setWindowFilter(int first, int last)\n"
             << endl;
        exit(0);
    }

    bool cutlowup = false;
    bool cutlowdown = false;

    bool endcutleft = false;

    for (int i = 0; i < adc.size(); i++)
    {
        int adcv = -(adc[i] - bsl);

        if ((i <= lowfilter && adcv > adcuthg))
        {
            cutlowup = true;
        }

        if ((i <= lowfilter && adcv < adcutlw))
        {
            cutlowdown = true;
        }

        endcutleft = cutlowup || cutlowdown;
        if (endcutleft)
            break;
    }

    return endcutleft;
}

bool wffunctions::filterright(int adcutlw, int adcuthg, int bsl)
{

    if (lowfilter == -1 || highfilter == -1)
    {
        cerr << "\n\nWindow not define! First define setWindowFilter(int first, int last)\n"
             << endl;
        exit(0);
    }

    bool cuthighup = false;
    bool cuthighdown = false;

    bool endcutright = false;

    for (int i = 0; i < adc.size(); i++)
    {
        int adcv = -(adc[i] - bsl);

        if ((i >= highfilter && adcv > adcuthg))
        {
            cuthighup = true;
        }

        if ((i >= highfilter && adcv < adcutlw))
        {
            cuthighdown = true;
        }
        endcutright = cuthighup || cuthighdown;
        if (endcutright)
            break;
    }

    return endcutright;
}