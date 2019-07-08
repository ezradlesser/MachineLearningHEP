#############################################################################
##  © Copyright CERN 2018. All rights not expressly granted are reserved.  ##
##                 Author: Gian.Michele.Innocenti@cern.ch                  ##
## This program is free software: you can redistribute it and/or modify it ##
##  under the terms of the GNU General Public License as published by the  ##
## Free Software Foundation, either version 3 of the License, or (at your  ##
## option) any later version. This program is distributed in the hope that ##
##  it will be useful, but WITHOUT ANY WARRANTY; without even the implied  ##
##     warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    ##
##           See the GNU General Public License for more details.          ##
##    You should have received a copy of the GNU General Public License    ##
##   along with this program. if not, see <https://www.gnu.org/licenses/>. ##
#############################################################################

"""
main script for doing data processing, machine learning and analysis
"""

import os
import yaml
import argparse
from multiprocesser import MultiProcesser  # pylint: disable=import-error
#from machine_learning_hep.doskimming import conversion, merging, merging_period, skim
#from machine_learning_hep.doclassification_regression import doclassification_regression
#from machine_learning_hep.doanalysis import doanalysis
#from machine_learning_hep.extractmasshisto import extractmasshisto
#from machine_learning_hep.efficiencyan import analysis_eff
from  machine_learning_hep.utilities import checkdirlist, checkdir
from optimiser import Optimiser

def do_entire_analysis(analysis_config_file): # pylint: disable=too-many-locals, too-many-statements, too-many-branches

    # Load configuration file specifying main options to execute: conversion, skimming, analysis, etc.
    with open("default_complete.yaml", 'r') as run_config:
        data_config = yaml.load(run_config)

    # Load configuration file containing all relevant analysis parameters
    with open(analysis_config_file, 'r', encoding='utf-8') as param_config:
        data_param = yaml.load(param_config)

    with open("data/config_model_parameters.yml", 'r') as mod_config:
        data_model = yaml.load(mod_config)

    with open("data/database_run_list.yml", 'r') as runlist_config:
        run_param = yaml.load(runlist_config)

    with open("data/database_ml_gridsearch.yml", 'r') as grid_config:
        grid_param = yaml.load(grid_config)

    # Load parameters from data_config -- Required parameters
    usemc = data_config["use_mc"]
    usedata = data_config["use_data"]

    case = data_config["case"]
    doconversionmc = data_config["conversion"]["mc"]["activate"]
    doconversiondata = data_config["conversion"]["data"]["activate"]
    domergingmc = data_config["merging"]["mc"]["activate"]
    domergingdata = data_config["merging"]["data"]["activate"]
    doskimmingmc = data_config["skimming"]["mc"]["activate"]
    doskimmingdata = data_config["skimming"]["data"]["activate"]
    domergingperiodsmc = data_config["mergingperiods"]["mc"]["activate"]
    domergingperiodsdata = data_config["mergingperiods"]["data"]["activate"]
    doapplydata = data_config["analysis"]["data"]["doapply"]
    doapplymc = data_config["analysis"]["mc"]["doapply"]
    domergeapplydata = data_config["analysis"]["data"]["domergeapply"]
    domergeapplymc = data_config["analysis"]["mc"]["domergeapply"]
    dojetdata = data_config["jetanalysis"]["data"]["activate"]
    dojetmc = data_config["jetanalysis"]["mc"]["activate"]

    # Load parameters from data_config -- Optional parameters
    doml = docorrelation = dotraining = dotesting = doapplytodatamc = docrossvalidation = \
           dolearningcurve = doroc = doboundary = doimportance = dogridsearch = dosignifopt = \
           dohistomassmc = dohistomassdata = doefficiency = None
    if 'Jet' not in case:
      doml = data_config["ml_study"]["activate"]
      docorrelation = data_config["ml_study"]['docorrelation']
      dotraining = data_config["ml_study"]['dotraining']
      dotesting = data_config["ml_study"]['dotesting']
      doapplytodatamc = data_config["ml_study"]['applytodatamc']
      docrossvalidation = data_config["ml_study"]['docrossvalidation']
      dolearningcurve = data_config["ml_study"]['dolearningcurve']
      doroc = data_config["ml_study"]['doroc']
      doboundary = data_config["ml_study"]['doboundary']
      doimportance = data_config["ml_study"]['doimportance']
      dogridsearch = data_config["ml_study"]['dogridsearch']
      dosignifopt = data_config["ml_study"]['dosignifopt']
      #doefficiency = run_config['doefficiency']
      dohistomassmc = data_config["analysis"]["mc"]["histomass"]
      dohistomassdata = data_config["analysis"]["data"]["histomass"]
      doefficiency = data_config["analysis"]["mc"]["efficiency"]

    # Load parameters from data_param -- Required parameters
    dirpklmc = data_param[case]["multi"]["mc"]["pkl"]
    dirpklevtcounter_allmc = data_param[case]["multi"]["mc"]["pkl_evtcounter_all"]
    dirpklskmc = data_param[case]["multi"]["mc"]["pkl_skimmed"]
    dirpkldata = data_param[case]["multi"]["data"]["pkl"]
    dirpklevtcounter_alldata = data_param[case]["multi"]["data"]["pkl_evtcounter_all"]
    dirpklskdata = data_param[case]["multi"]["data"]["pkl_skimmed"]
    dirresultsdata = data_param[case]["analysis"]["data"]["results"]
    dirresultsmc = data_param[case]["analysis"]["mc"]["results"]

    # Load parameters from data_param -- Optional parameters
    dirpklmlmc = dirpklmltotmc = dirpklmldata = dirpklmltotdata = dirpklskdecmc = \
                 dirpklskdec_mergedmc = dirpklskdecdata = dirpklskdec_mergeddata = \
                 binminarray = binmaxarray = raahp = mltype = mlout = mlplot = None
    if 'Jet' not in case:
      dirpklmlmc = data_param[case]["multi"]["mc"]["pkl_skimmed_merge_for_ml"]
      dirpklmltotmc = data_param[case]["multi"]["mc"]["pkl_skimmed_merge_for_ml_all"]
      dirpklmldata = data_param[case]["multi"]["data"]["pkl_skimmed_merge_for_ml"]
      dirpklmltotdata = data_param[case]["multi"]["data"]["pkl_skimmed_merge_for_ml_all"]
      dirpklskdecmc = data_param[case]["analysis"]["mc"]["pkl_skimmed_dec"]
      dirpklskdec_mergedmc = data_param[case]["analysis"]["mc"]["pkl_skimmed_decmerged"]
      dirpklskdecdata = data_param[case]["analysis"]["data"]["pkl_skimmed_dec"]
      dirpklskdec_mergeddata = data_param[case]["analysis"]["data"]["pkl_skimmed_decmerged"]
      binminarray = data_param[case]["ml"]["binmin"]
      binmaxarray = data_param[case]["ml"]["binmax"]
      raahp = data_param[case]["ml"]["opt"]["raahp"]
      mltype = data_param[case]["ml"]["mltype"]
      mlout = data_param[case]["ml"]["mlout"]
      mlplot = data_param[case]["ml"]["mlplot"]

    # Create instance of multiprocessor class
    mymultiprocessmc = None
    if usemc:
        mymultiprocessmc = MultiProcesser(case, data_param[case], run_param, "mc")
    if usedata:
        mymultiprocessdata = MultiProcesser(case, data_param[case], run_param, "data")

    #creating folder if not present
    if doconversionmc is True:
        if checkdirlist(dirpklmc) is True:
            exit()

    if doconversiondata is True:
        if checkdirlist(dirpkldata) is True:
            exit()

    if doskimmingmc is True:
        if checkdirlist(dirpklskmc) or checkdir(dirpklevtcounter_allmc) is True:
            exit()

    if doskimmingdata is True:
        if checkdirlist(dirpklskdata) or checkdir(dirpklevtcounter_alldata) is True:
            exit()

    if domergingmc is True:
        if checkdirlist(dirpklmlmc) is True:
            exit()

    if domergingdata is True:
        if checkdirlist(dirpklmldata) is True:
            exit()

    if domergingperiodsmc is True:
        if checkdir(dirpklmltotmc) is True:
            exit()

    if domergingperiodsdata is True:
        if checkdir(dirpklmltotdata) is True:
            exit()

    if doml is True:
        if checkdir(mlout) or checkdir(mlplot) is True:
            print("check mlout and mlplot")

    if doapplymc is True:
        if checkdirlist(dirpklskdecmc) is True:
            exit()

    if doapplydata is True:
        if checkdirlist(dirpklskdecdata) is True:
            exit()

    if domergeapplymc is True:
        if checkdirlist(dirpklskdec_mergedmc) is True:
            exit()

    if domergeapplydata is True:
        if checkdirlist(dirpklskdec_mergeddata) is True:
            exit()

    if dohistomassmc is True:
        if checkdirlist(dirresultsmc) is True:
            exit()

    if dohistomassdata is True:
        if checkdirlist(dirresultsdata) is True:
            print("folder exists")

    # Perform the analysis flow

    # Convert ROOT to pickle files if required
    if doconversionmc:
        mymultiprocessmc.multi_unpack_allperiods()

    if doconversiondata:
        mymultiprocessdata.multi_unpack_allperiods()

    # Skim the data if required
    if doskimmingmc:
        mymultiprocessmc.multi_skim_allperiods()

    if doskimmingdata:
        mymultiprocessdata.multi_skim_allperiods()

    # Do jet finding & analysis if desired
    if dojetdata:
        mymultiprocessdata.multi_jet()

    if dojetmc:
        mymultiprocessmc.multi_jet()

    # Merge data files for ML if required
    if domergingmc:
        mymultiprocessmc.multi_mergeml_allperiods()

    if domergingdata:
        mymultiprocessdata.multi_mergeml_allperiods()

    if domergingperiodsmc:
        mymultiprocessmc.multi_mergeml_allinone()

    if domergingperiodsdata:
        mymultiprocessdata.multi_mergeml_allinone()

    # Do machine learning if required
    if doml:
        index = 0
        for binmin, binmax in zip(binminarray, binmaxarray):
            myopt = Optimiser(data_param[case], case,
                              data_model[mltype], grid_param, binmin, binmax,
                              raahp[index])
            if docorrelation:
                myopt.do_corr()
            if dotraining:
                myopt.do_train()
            if dotesting:
                myopt.do_test()
            if doapplytodatamc:
                myopt.do_apply()
            if docrossvalidation:
                myopt.do_crossval()
            if dolearningcurve:
                myopt.do_learningcurve()
            if doroc:
                myopt.do_roc()
            if doimportance:
                myopt.do_importance()
            if dogridsearch:
                myopt.do_grid()
            if doboundary:
                myopt.do_boundary()
            if dosignifopt:
                myopt.do_significance()
            index = index + 1

    if doapplydata:
        mymultiprocessapplydata = MultiProcesser(data_param[case], run_param, "data")
        mymultiprocessapplydata.multi_apply_allperiods()
    if doapplymc:
        mymultiprocessapplymc = MultiProcesser(data_param[case], run_param, "mc")
        mymultiprocessapplymc.multi_apply_allperiods()
    if domergeapplydata:
        mymultiprocessmergeapplydata = MultiProcesser(data_param[case], run_param, "data")
        mymultiprocessmergeapplydata.multi_mergeapply_allperiods()
    if domergeapplymc:
        mymultiprocessmergeapplymc = MultiProcesser(data_param[case], run_param, "mc")
        mymultiprocessmergeapplymc.multi_mergeapply_allperiods()
    if dohistomassmc:
        mymultiprocessapplymc = MultiProcesser(data_param[case], run_param, "mc")
        mymultiprocessapplymc.multi_histomass()
    if dohistomassdata:
        mymultiprocessapplydata = MultiProcesser(data_param[case], run_param, "data")
        mymultiprocessapplydata.multi_histomass()
    if doefficiency:
        mymultiprocesseffmc = MultiProcesser(data_param[case], run_param, "mc")
        mymultiprocesseffmc.multi_efficiency()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
  # Define arguments
  parser = argparse.ArgumentParser(description="Do entire analysis")
  parser.add_argument("-c", "--analysis_config_file", action="store",
                      type=str, metavar="analysis_config_file",
                      default="data/database_ml_parameters.yml",
                      help="Path of config file for analysis parameters")

  # Parse the arguments
  args = parser.parse_args()
  
  arg = args.analysis_config_file
  print( "analysis_config_file: \"{0}\"".format(arg) )
  
  # If invalid configFile is given, exit
  if not os.path.exists(args.analysis_config_file):
    print("File \"{0}\" does not exist! Exiting!".format(args.analysis_config_file))
    sys.exit(0)

  do_entire_analysis(analysis_config_file = args.analysis_config_file)
