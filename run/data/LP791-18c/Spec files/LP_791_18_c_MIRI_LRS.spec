# %ECSV 1.0
# ---
# datatype:
# - {name: instrname, datatype: string}
# - {name: reference, datatype: string}
# - {name: bandpass, datatype: string}
# - {name: iwave, datatype: int64}
# - {name: wave, unit: um, datatype: float64}
# - {name: waveMin, unit: um, datatype: float64}
# - {name: waveMax, unit: um, datatype: float64}
# - {name: xMin, datatype: int64}
# - {name: xMax, datatype: int64}
# - {name: yval, unit: ppm, datatype: float64}
# - {name: yerrLow, unit: ppm, datatype: float64}
# - {name: yerrUpp, unit: ppm, datatype: float64}
# - {name: wlcLow, datatype: float64}
# - {name: wlcUpp, datatype: float64}
# - {name: ignore, datatype: int64}
# - {name: referenceLink, datatype: string}
# delimiter: ','
# meta: !!omap
# - {spectype: dppm}
# - {color: purple}
# - {nicelabel: MIRI LRS}
# - {name: MIRI_LRS (simulated)}
# - {label: MIRI_LRS}
# - {instrname: MIRI_LRS}
# - {obstime: 42852.24}
# - {ngroup: 166}
# - pandexo_configuration:
#     detector: {nexp: 1, ngroup: 166, nint: 652, readmode: fast, readout_pattern: fast, subarray: slitlessprism}
#     instrument: {aperture: imager, disperser: p750l, filter: null, instrument: miri, mode: lrsslitless}
# - pandexo_exposure: {duty_cycle: 0.9939759036144578, exposure_time: 17213.217279999997, exposures: 1, frame0: false, measurement_time: 17109.523199999996,
#     ndrop1: 0, ndrop2: 0, ndrop3: 0, nexp: 1, nframe: 1, ngroup: 166, nint: 652, npostrej: 0, nprerej: 0, nramps: 652, nreset1: 0, nreset2: 0,
#     nsample: 1, nsample_skip: 0, nsample_total: 1, readout_pattern: fast, saturation_time: 26.40064, subarray: slitlessprism, tfffr: 0.0,
#     tframe: 0.15904, tgroup: 0.15904, total_exposure_time: 17213.217279999997, total_integrations: 652, tsample: 0.15904, webapp: true}
# - {duty_cycle: 0.9939759036144578}
# - {saturatedPixels: 0.0}
# - APT: {Groups/In: 166, Integrations/Exp: 652, Time/Integration incl reset (sec): 26.559679999999997, Time/Integration without reset (s): 26.40064,
#     Total Exposure Time: 17213.217279999997, 'Total Exposure Time [hours]': 4.781449244444444, 'Transit+Baseline, no overhead (hrs)': 4.810253155555555,
#     baseline: 4.1, saturation_time (sec): 26.40064}
# - sysModel: {addParas: null, bestfit: null, guess: null, sysModel: null}
# - {simulation: true}
# - {random: 0.75}
# - {model_file: /Users/bbenneke/Research/Proposals_for_Observations/53_JWST_Cycle4_Proposals/Cycle_2_Results/LP_791_18_c_WellMixed_H2_0.61_CH4_0.251189_He_0.1_TpTeqTint20f0.25A0.0_cHaze218776.16239495517_mieRpart0.954992586021436.atm}
# - {model_legendlabel: '''He'': 0.1, ''H2'': 0.61, ''CH4'': 0.251188643150958, ''H2O'': 2.754228703338169e-05, ''H2S'': 0.00039810717055349735,
#     ''SO2'': 0.0446683592150963, ''NH3'': 2.0417379446695274e-05, ''CO2'': 5.4954087385762485e-06, ''CO'': 2.4547089156850284e-06_Mie
#     Clouds at 1047.1 mbar'}
# - {model_shortlabel: He_0.1_H2_0.61_CH4_0.251188643150958_H2O_2.754228703338169e-05_H2S_0.00039810717055349735_SO2_0.0446683592150963_NH3_2.0417379446695274e-05_CO2_5.4954087385762485e-06_CO_2.4547089156850284e-06_Mie
#     Clouds at 1047.1 mbar}
# - {nobs: 2}
# - {specfilelabel: LP_791_18_c_He_0.1_H2_0.61_CH4_0.251188643150958_H2O_2.754228703338169e-05_H2S_0.00039810717055349735_SO2_0.0446683592150963_NH3_2.0417379446695274e-05_CO2_5.4954087385762485e-06_CO_2.4547089156850284e-06_Mie
#     Clouds at 1047.1 mbar_MIRI_LRS_nTra2r0}
# schema: astropy-2.0
instrname,reference,bandpass,iwave,wave,waveMin,waveMax,xMin,xMax,yval,yerrLow,yerrUpp,wlcLow,wlcUpp,ignore,referenceLink
MIRI_LRS,Benneke,uniform,0,5.0625390443466145,4.98141821998563,5.143659868707599,0,0,15133.0,109.2644,109.2644,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,2,5.215496188010285,5.143659868707599,5.287332507312971,0,0,14954.2111,100.3374,100.3374,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,4,5.388882073383067,5.287332507312971,5.490431639453163,0,0,15167.4829,86.1383,86.1383,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,7,5.581857333307044,5.490431639453163,5.673283027160924,0,0,15009.4436,93.0546,93.0546,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,10,5.759508648278272,5.673283027160924,5.845734269395622,0,0,15093.3952,98.9314,98.9314,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,13,5.949887155158647,5.845734269395622,6.054040040921672,0,0,15062.2924,88.2056,88.2056,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,17,6.152994372950494,6.054040040921672,6.251948704979316,0,0,15139.552,94.2312,94.2312,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,21,6.343555358949811,6.251948704979316,6.435162012920305,0,0,14960.5079,95.3654,95.3654,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,25,6.542433704437843,6.435162012920305,6.649705395955381,0,0,14980.5897,97.3332,97.3332,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,30,6.753016702920737,6.649705395955381,6.856328009886093,0,0,14965.916,107.79,107.79,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,35,6.973088861039027,6.856328009886093,7.0898497121919615,0,0,15113.3571,104.0728,104.0728,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,41,7.201750440345706,7.0898497121919615,7.313651168499451,0,0,15304.0238,109.102,109.102,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,47,7.437014359918621,7.313651168499451,7.560377551337791,0,0,15458.1616,109.5143,109.5143,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,54,7.678446729348117,7.560377551337791,7.796515907358444,0,0,15452.7339,118.1625,118.1625,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,61,7.92600202650673,7.796515907358444,8.055488145655016,0,0,15075.02,120.6369,120.6369,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,69,8.194151306302267,8.055488145655016,8.332814466949518,0,0,15192.9283,126.8208,126.8208,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,78,8.465107617550128,8.332814466949518,8.597400768150738,0,0,15336.4743,133.4266,133.4266,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,87,8.737450173715585,8.597400768150738,8.877499579280432,0,0,15348.7972,137.028,137.028,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,97,9.023922780204769,8.877499579280432,9.170345981129106,0,0,15253.6377,145.2278,145.2278,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,108,9.320824220090476,9.170345981129106,9.471302459051845,0,0,15122.6834,158.7646,158.7646,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,120,9.62585155536421,9.471302459051845,9.780400651676576,0,0,15346.0498,169.4488,169.4488,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,133,9.93882614235617,9.780400651676576,10.097251633035762,0,0,15003.2101,189.9598,189.9598,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,147,10.268150296460231,10.097251633035762,10.439048959884698,0,0,15074.6064,226.5239,226.5239,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,163,10.611611934936215,10.439048959884698,10.784174909987733,0,0,14743.3695,305.4276,305.4276,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,180,10.957597826532359,10.784174909987733,11.131020743076984,0,0,14937.2554,413.3154,413.3154,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,198,11.315625523338602,11.131020743076984,11.500230303600219,0,0,15042.3679,551.8124,551.8124,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,218,11.685661281647048,11.500230303600219,11.871092259693878,0,0,15863.3309,748.9778,748.9778,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,239,12.064883773802716,11.871092259693878,12.258675287911554,0,0,14123.537,1146.8024,1146.8024,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,262,12.45857700429442,12.258675287911554,12.658478720677287,0,0,14793.1564,1769.0194,1769.0194,0.0,0.0,0,none
MIRI_LRS,Benneke,uniform,287,12.861815279469704,12.658478720677287,13.065151838262121,0,0,12397.9758,2451.5281,2451.5281,0.0,0.0,0,none
