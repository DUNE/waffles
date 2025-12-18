WF_FILE = "Run_32148_ChSiPM_11121_ChST_11120_structured.hdf5";
TEMPL_F = "Template_files/mynewtemplate20240409.dat";
NOISE_F = "";
MUON_F  = "GMuon_files/mu20240409_oldel.dat";
TRG_F   = "noise2023018";
// daphne caen csv 
DATA_FORMAT    = "hdf5";
INVERT         = 1;
MEMORYDEPTH    = 1024;
N_WF           = 51926;
RES            = 14;
PREPULSE_TICKS = 127;
PRETRG         = 230;
AFTTRG         = 249;
TICK_LEN       = 0.016;
 
SAT_UP      = 70;
SAT_LOW     = -90;
BSL         = 40;
RMS         = 5;
 
 
//calibration
INT_LOW        = 128;
INT_UP         = 147;
NBINS          = 300;
NMAXPEAKS      = 6;
MU0_LOW        = -61;
MU0_UP         = 28;
SPE_LOW        = 65;
SPE_UP         = 125;
S0_LOW         = 20;
S0_UP          = 180;
SC_LOW         = 5;
SC_UP          = 80;
FIT_LOW        = -200;
FIT_UP         = 18;
HMIN           = -200;
HMAX           = 800;
SPE_AMPL       = 6.22911;
SPE_CHARGE     = 97.0853;
PEDESTAL       = -1.73926;
//deconvolution
INT_PROMPT     = 1520;
ROLL           = 80;
AMP            = 7.47582;
F_PROMPT       = 0.48;
AMP_LOW        = 5000;
AMP_UP         = 6800;
DECO_SM        = 0.141051;
N2_            = 5e-06;
FIT_L          = 6.15;
FIT_U          = 18;
A_FAST         = 89.5189;
TAU_FAST       = 0.0573684;
A_SLOW         = 0.904042;
TAU_SLOW       = 1.14084;
SIGMA          = 0.00019298;
T_0            = 0;
YERR           = 5;
//ProtoDUNE
CHANNEL        = 11121;
