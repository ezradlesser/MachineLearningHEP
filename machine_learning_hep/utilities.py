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
Helper functions for processing & file operations
"""

from __future__ import print_function
import pickle
import os
import numpy as np
import pandas as pd
from machine_learning_hep.selectionutils import select_runs

def selectdfquery(dfr, selection):
    if selection is not None:
        dfr = dfr.query(selection)
    return dfr

def selectdfrunlist(dfr, runlist, runvar):
    if runlist is not None:
        isgoodrun = select_runs(runlist, dfr[runvar].values)
        dfr = dfr[np.array(isgoodrun, dtype=bool)]
    return dfr

def merge_method(listfiles, namemerged):
    dflist = []
    for myfilename in listfiles:
        myfile = open(myfilename, "rb")
        df = pickle.load(myfile)
        dflist.append(df)
    dftot = pd.concat(dflist)
    dftot.to_pickle(namemerged)

# pylint: disable=too-many-nested-blocks
def list_folders(main_dir, filenameinput, maxfiles):
    if not os.path.isdir(main_dir):
        print("the input directory =", main_dir, "doesnt exist")
    list_subdir0 = os.listdir(main_dir)
    listfolders = list()
    for subdir0 in list_subdir0:
        subdir0full = os.path.join(main_dir, subdir0)
        if os.path.isdir(subdir0full):
            list_subdir1 = os.listdir(subdir0full)
            for subdir1 in list_subdir1:
                subdir1full = os.path.join(subdir0full, subdir1)
                if os.path.isdir(subdir1full):
                    list_files_ = os.listdir(subdir1full)
                    for myfile in list_files_:
                        filefull = os.path.join(subdir1full, myfile)
                        if os.path.isfile(filefull) and \
                        myfile == filenameinput:
                            listfolders.append(os.path.join(subdir0, subdir1))
    if maxfiles is not -1:
        listfolders = listfolders[:maxfiles]
    return  listfolders

def create_folder_struc(maindir, listpath):
    for path in listpath:
        path = path.split("/")
        for subpath in path:
            folder = os.path.join(maindir, subpath)
            if not os.path.exists(folder):
                os.makedirs(folder)

def checkdirlist(dirlist):
    exfolders = 0
    for _, mydir in enumerate(dirlist):
        if os.path.exists(mydir):
            print("rm -rf ", mydir)
            exfolders = exfolders + 1
        else:
            print("creating folder ", mydir)
            os.makedirs(mydir)
    return exfolders > 0

def checkdir(mydir):
    exfolder = 0
    if os.path.exists(mydir):
        print("rm -rf ", mydir)
        exfolder = exfolder + 1
    else:
        print("creating folder ", mydir)
        os.makedirs(mydir)

    return exfolder > 0

def appendfiletolist(mylist, namefile):
    return [os.path.join(path, namefile) for path in mylist]

def appendmainfoldertolist(prefolder, mylist):
    return [os.path.join(prefolder, path) for path in mylist]

def createlist(prefolder, mylistfolder, namefile):
    listfiles = appendfiletolist(mylistfolder, namefile)
    listfiles = appendmainfoldertolist(prefolder, listfiles)
    return listfiles

def seldf_singlevar(dataframe, var, minval, maxval):
    dataframe = dataframe.loc[(dataframe[var] > minval) & (dataframe[var] < maxval)]
    return dataframe

def split_df_sigbkg(dataframe_, var_signal_):
    dataframe_sig_ = dataframe_.loc[dataframe_[var_signal_] == 1]
    dataframe_bkg_ = dataframe_.loc[dataframe_[var_signal_] == 0]
    return dataframe_sig_, dataframe_bkg_

def createstringselection(var, low, high):
    string_selection = "dfselection_"+(("%s_%.1f_%.1f") % (var, low, high))
    return string_selection

def concat_dir(main_dir, sub_dir):
    out_dir = ""
    if main_dir[-1] == '/':
        out_dir = main_dir + sub_dir
    else:
        out_dir = main_dir + '/' + sub_dir
    return out_dir

def get_pT_bin_list(pTbins):
    pT_bin_list = []
    for i, pTmin in list(enumerate(pTbins))[0:-1]:
        pTmax = pTbins[i+1]
        bin_name = "%i-%i" % (pTmin, pTmax)
        pT_bin_list.append(bin_name)
    return pT_bin_list
