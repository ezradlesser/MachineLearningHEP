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
Individual processing class used to process individual tasks called by multi-processing
"""

from __future__ import division, print_function
import math
import array
import multiprocessing as mp
import pickle
import os
import random as rd
import uproot
import pandas as pd
import numpy as np
import pyjet
from root_numpy import fill_hist # pylint: disable=import-error, no-name-in-module
from ROOT import TFile, TH1F # pylint: disable=import-error, no-name-in-module

from machine_learning_hep.selectionutils import selectfidacc
from machine_learning_hep.bitwise import filter_bit_df, tag_bit_df
from machine_learning_hep.utilities import selectdfquery, selectdfrunlist, merge_method
from machine_learning_hep.utilities import list_folders, createlist, appendmainfoldertolist
from machine_learning_hep.utilities import create_folder_struc, seldf_singlevar
from machine_learning_hep.models import apply # pylint: disable=import-error

class Processer: # pylint: disable=too-many-instance-attributes
    # Class Attribute
    species = 'processer'

    # Initializer / Instance Attributes
    # pylint: disable=too-many-statements, too-many-arguments
    def __init__(self, case, datap, run_param, mcordata, p_maxfiles,
                 d_root, d_pkl, d_pklsk, d_pkl_ml, p_period,
                 p_chunksizeunp, p_chunksizeskim, p_maxprocess,
                 p_frac_merge, p_rd_merge, d_pkl_dec, d_pkl_decmerged, d_results):

        #directories
        self.case = case
        self.d_root = d_root
        self.d_pkl = d_pkl
        self.d_pklsk = d_pklsk
        self.d_pkl_ml = d_pkl_ml
        self.datap = datap
        self.mcordata = mcordata
        self.p_frac_merge = p_frac_merge
        self.p_rd_merge = p_rd_merge
        self.period = p_period
        self.runlist = run_param[self.period]

        self.p_maxfiles = p_maxfiles
        self.p_chunksizeunp = p_chunksizeunp
        self.p_chunksizeskim = p_chunksizeskim

        #parameter names
        self.p_maxprocess = p_maxprocess
        self.indexsample = None

        #namefile root
        self.n_root = datap["files_names"]["namefile_unmerged_tree"]
        #troot trees names
        self.n_treereco = datap["files_names"]["treeoriginreco"]
        self.n_treegen = datap["files_names"]["treeorigingen"]
        self.n_treeevt = datap["files_names"]["treeoriginevt"]

        #namefiles pkl
        self.n_reco = datap["files_names"]["namefile_reco"]
        self.n_evt = datap["files_names"]["namefile_evt"]
        self.n_evtorig = datap["files_names"]["namefile_evtorig"]
        self.n_gen = datap["files_names"]["namefile_gen"]
        self.n_filemass = self.n_fileeff = None
        if 'Jet' not in case:
            self.n_filemass = datap["files_names"]["histofilename"]
            self.n_fileeff = datap["files_names"]["efffilename"]

        #selections
        self.s_reco_unp = datap["sel_reco_unp"]
        self.s_good_evt_unp = datap["sel_good_evt_unp"]
        self.s_cen_unp = datap["sel_cen_unp"]
        self.s_gen_unp = datap["sel_gen_unp"]
        self.s_reco_skim = datap["sel_reco_skim"]
        self.s_gen_skim = datap["sel_gen_skim"]

        #bitmap
        self.b_trackcuts = self.b_std = self.b_mcsig = self.b_mcsigprompt = \
            self.b_mcsigfd = self.b_mcbkg = None
        if 'Jet' not in case:
            self.b_trackcuts = datap["sel_reco_singletrac_unp"]
            self.b_std = datap["bitmap_sel"]["isstd"]
            self.b_mcsig = datap["bitmap_sel"]["ismcsignal"]
            self.b_mcsigprompt = datap["bitmap_sel"]["ismcprompt"]
            self.b_mcsigfd = datap["bitmap_sel"]["ismcfd"]
            self.b_mcbkg = datap["bitmap_sel"]["ismcbkg"]

        #variables name
        self.v_all = datap["variables"]["var_all"]
        self.v_evt = datap["variables"]["var_evt"][self.mcordata]
        self.v_gen = datap["variables"]["var_gen"]
        self.v_evtmatch = datap["variables"]["var_evt_match"]
        self.v_train = self.v_bitvar = self.v_isstd = self.v_ismcsignal = \
            self.v_ismcprompt = self.v_ismcfd = self.v_ismcbkg = self.v_var_binning = None
        if 'Jet' not in case:
            self.v_train = datap["variables"]["var_training"]
            self.v_bitvar = datap["bitmap_sel"]["var_name"]
            self.v_isstd = datap["bitmap_sel"]["var_isstd"]
            self.v_ismcsignal = datap["bitmap_sel"]["var_ismcsignal"]
            self.v_ismcprompt = datap["bitmap_sel"]["var_ismcprompt"]
            self.v_ismcfd = datap["bitmap_sel"]["var_ismcfd"]
            self.v_ismcbkg = datap["bitmap_sel"]["var_ismcbkg"]
            self.v_var_binning = datap["variables"]["var_binning"]


        #list of files names

        self.l_path = None
        if os.path.isdir(self.d_root):
            self.l_path = list_folders(self.d_root, self.n_root, self.p_maxfiles)
        else:
            self.l_path = list_folders(self.d_pkl, self.n_reco, self.p_maxfiles)

        self.l_root = createlist(self.d_root, self.l_path, self.n_root)
        self.l_reco = createlist(self.d_pkl, self.l_path, self.n_reco)
        self.l_evt = createlist(self.d_pkl, self.l_path, self.n_evt)
        self.l_evtorig = createlist(self.d_pkl, self.l_path, self.n_evtorig)

        if self.mcordata == "mc":
            self.l_gen = createlist(self.d_pkl, self.l_path, self.n_gen)

        self.f_totevt = os.path.join(self.d_pkl, self.n_evt)
        self.f_totevtorig = os.path.join(self.d_pkl, self.n_evtorig)

        self.lpt_anbinmin = datap["sel_skim_binmin"]
        self.lpt_anbinmax = datap["sel_skim_binmax"]
        self.p_nptbins = len(datap["sel_skim_binmax"])

        self.p_modelname = self.lpt_model = self.dirmodel = self.lpt_model = \
            self.lpt_probcutpre = self.lpt_probcutfin = self.d_pkl_decmerged = \
            self.n_filemass = self.n_fileeff = None

        if 'Jet' in case:
            self.jetRadii = datap['variables']['jetRadii']
            self.pTbins = datap['variables']['pTbins']
            self.betas = datap['variables']['betas']
            self.jets = None   # Will fill this in if needed using findJets()

        if 'Jet' not in case:
            self.p_modelname = datap["analysis"]["modelname"]
            self.lpt_model = datap["analysis"]["modelsperptbin"]
            self.dirmodel = datap["ml"]["mlout"]
            self.lpt_model = appendmainfoldertolist(self.dirmodel, self.lpt_model)
            self.lpt_probcutpre = datap["analysis"]["probcutpresel"]
            self.lpt_probcutfin = datap["analysis"]["probcutoptimal"]

            self.d_pkl_decmerged = d_pkl_decmerged
            self.n_filemass = os.path.join(d_results, self.n_filemass)
            self.n_fileeff = os.path.join(d_results, self.n_fileeff)

        self.d_pkl_dec = d_pkl_dec
        self.mptfiles_recosk = []
        self.mptfiles_gensk = []

        self.lpt_recosk = [self.n_reco.replace(".pkl", "%d_%d.pkl" % \
                          (self.lpt_anbinmin[i], self.lpt_anbinmax[i])) \
                          for i in range(self.p_nptbins)]
        self.lpt_gensk = [self.n_gen.replace(".pkl", "%d_%d.pkl" % \
                          (self.lpt_anbinmin[i], self.lpt_anbinmax[i])) \
                          for i in range(self.p_nptbins)]

        self.lpt_reco_ml = self.lpt_gen_ml = self.f_evt_ml = self.f_evtorig_ml = \
            self.lpt_recodec = self.mptfiles_recoskmldec = self.lpt_recodecmerged = None

        if 'Jet' not in case:
            self.lpt_reco_ml = [os.path.join(self.d_pkl_ml, self.lpt_recosk[ipt]) \
                               for ipt in range(self.p_nptbins)]
            self.lpt_gen_ml = [os.path.join(self.d_pkl_ml, self.lpt_gensk[ipt]) \
                              for ipt in range(self.p_nptbins)]
            self.f_evt_ml = os.path.join(self.d_pkl_ml, self.n_evt)
            self.f_evtorig_ml = os.path.join(self.d_pkl_ml, self.n_evtorig)

            self.lpt_recodec = [self.n_reco.replace(".pkl", "%d_%d_%.2f.pkl" % \
                             (self.lpt_anbinmin[i], self.lpt_anbinmax[i], \
                              self.lpt_probcutpre[i])) for i in range(self.p_nptbins)]
            self.mptfiles_recoskmldec = [createlist(self.d_pkl_dec, self.l_path, \
                                        self.lpt_recodec[ipt]) for ipt in range(self.p_nptbins)]
            self.lpt_recodecmerged = [os.path.join(self.d_pkl_decmerged, self.lpt_recodec[ipt])
                          for ipt in range(self.p_nptbins)]
  
        self.mptfiles_recosk = [createlist(self.d_pklsk, self.l_path, \
                                self.lpt_recosk[ipt]) for ipt in range(self.p_nptbins)]

        if self.mcordata == "mc":
            self.mptfiles_gensk = [createlist(self.d_pklsk, self.l_path, \
                                    self.lpt_gensk[ipt]) for ipt in range(self.p_nptbins)]
            self.lpt_gendecmerged = None
            if 'Jet' not in case:
                self.lpt_gendecmerged = [os.path.join(self.d_pkl_decmerged, self.lpt_gensk[ipt])
                                     for ipt in range(self.p_nptbins)]

        self.lpt_filemass = self.p_mass_fit_lim = self.p_bin_width = self.p_num_bins = \
            self.l_selml = self.s_presel_gen_eff = None
        if 'Jet' not in case:
            self.lpt_filemass = [self.n_filemass.replace(".root", "%d_%d_%.2f.root" % \
                  (self.lpt_anbinmin[ipt], self.lpt_anbinmax[ipt], \
                   self.lpt_probcutfin[ipt])) for ipt in range(self.p_nptbins)]

            self.p_mass_fit_lim = datap["analysis"]['mass_fit_lim']
            self.p_bin_width = datap["analysis"]['bin_width']
            self.p_num_bins = int(round((self.p_mass_fit_lim[1] - self.p_mass_fit_lim[0]) / \
                                        self.p_bin_width))
            self.l_selml = ["y_test_prob%s>%s" % (self.p_modelname, self.lpt_probcutfin[ipt]) \
                            for ipt in range(self.p_nptbins)]
            self.s_presel_gen_eff = datap["analysis"]['presel_gen_eff']

    def unpack(self, file_index):
        # Open root file and save event tree to dataframe
        treeevtorig = uproot.open(self.l_root[file_index])[self.n_treeevt]
        dfevtorig = treeevtorig.pandas.df(branches=self.v_evt)

        # Only save events within the given run period & required centrality
        dfevtorig = selectdfrunlist(dfevtorig, self.runlist, "run_number")
        dfevtorig = selectdfquery(dfevtorig, self.s_cen_unp)

        # Reset dataframe index and save to "original" pickle file
        dfevtorig = dfevtorig.reset_index(drop=True)
        dfevtorig.to_pickle(self.l_evtorig[file_index])

        # Select "good" events and save to a second pickle file
        dfevt = selectdfquery(dfevtorig, self.s_good_evt_unp)
        dfevt = dfevt.reset_index(drop=True)
        dfevt.to_pickle(self.l_evt[file_index])

        # Open root file again, get the reconstructed tree into a dataframe
        treereco = uproot.open(self.l_root[file_index])[self.n_treereco]
        if not treereco:
            print('Couldn\'t find tree %s in file %s' % \
                  (self.n_treereco, self.l_root[file_index]))
        dfreco = treereco.pandas.df(branches=self.v_all)

        # Only save events within the given run period & required cuts
        dfreco = selectdfrunlist(dfreco, self.runlist, "run_number")
        dfreco = selectdfquery(dfreco, self.s_reco_unp)
        dfreco = pd.merge(dfreco, dfevt, on=self.v_evtmatch)
        
        if 'Jet' not in self.case:
            isselacc = selectfidacc(dfreco.pt_cand.values, dfreco.y_cand.values)
            dfreco = dfreco[np.array(isselacc, dtype=bool)]
            if self.b_trackcuts is not None:
                dfreco = filter_bit_df(dfreco, self.v_bitvar, self.b_trackcuts)
            dfreco[self.v_isstd] = np.array(tag_bit_df(dfreco, self.v_bitvar,
                                                     self.b_std), dtype=int)
            dfreco = dfreco.reset_index(drop=True)
            if self.mcordata == "mc":
                dfreco[self.v_ismcsignal] = np.array(tag_bit_df(dfreco, self.v_bitvar,
                                                                self.b_mcsig), dtype=int)
                dfreco[self.v_ismcprompt] = np.array(tag_bit_df(dfreco, self.v_bitvar,
                                                                self.b_mcsigprompt), dtype=int)
                dfreco[self.v_ismcfd] = np.array(tag_bit_df(dfreco, self.v_bitvar,
                                                            self.b_mcsigfd), dtype=int)
                dfreco[self.v_ismcbkg] = np.array(tag_bit_df(dfreco, self.v_bitvar,
                                                             self.b_mcbkg), dtype=int)

        # Save reconstructed data to another pickle file
        dfreco.to_pickle(self.l_reco[file_index])

        if self.mcordata == "mc":
            treegen = uproot.open(self.l_root[file_index])[self.n_treegen]
            dfgen = treegen.pandas.df(branches=self.v_gen)
            dfgen = selectdfrunlist(dfgen, self.runlist, "run_number")
            dfgen = pd.merge(dfgen, dfevtorig, on=self.v_evtmatch)
            dfgen = selectdfquery(dfgen, self.s_gen_unp)
            dfgen[self.v_isstd] = np.array(tag_bit_df(dfgen, self.v_bitvar,
                                                      self.b_std), dtype=int)
            dfgen[self.v_ismcsignal] = np.array(tag_bit_df(dfgen, self.v_bitvar,
                                                           self.b_mcsig), dtype=int)
            dfgen[self.v_ismcprompt] = np.array(tag_bit_df(dfgen, self.v_bitvar,
                                                           self.b_mcsigprompt), dtype=int)
            dfgen[self.v_ismcfd] = np.array(tag_bit_df(dfgen, self.v_bitvar,
                                                       self.b_mcsigfd), dtype=int)
            dfgen[self.v_ismcbkg] = np.array(tag_bit_df(dfgen, self.v_bitvar,
                                                        self.b_mcbkg), dtype=int)
            dfgen = dfgen.reset_index(drop=True)
            dfgen.to_pickle(self.l_gen[file_index])

    def skim(self, file_index):
        dfreco = pickle.load(open(self.l_reco[file_index], "rb"))
        for ipt in range(self.p_nptbins):
            dfrecosk = seldf_singlevar(dfreco, self.v_var_binning,
                                       self.lpt_anbinmin[ipt], self.lpt_anbinmax[ipt])
            dfrecosk = selectdfquery(dfrecosk, self.s_reco_skim[ipt])
            dfrecosk = dfrecosk.reset_index(drop=True)
            dfrecosk.to_pickle(self.mptfiles_recosk[ipt][file_index])
            if self.mcordata == "mc":
                dfgen = pickle.load(open(self.l_gen[file_index], "rb"))
                dfgensk = seldf_singlevar(dfgen, self.v_var_binning,
                                          self.lpt_anbinmin[ipt], self.lpt_anbinmax[ipt])
                dfgensk = selectdfquery(dfgensk, self.s_gen_skim[ipt])
                dfgensk = dfgensk.reset_index(drop=True)
                dfgensk.to_pickle(self.mptfiles_gensk[ipt][file_index])

    def applymodel(self, file_index):
        for ipt in range(self.p_nptbins):
            dfrecosk = pickle.load(open(self.mptfiles_recosk[ipt][file_index], "rb"))
            mod = pickle.load(open(self.lpt_model[ipt], 'rb'))
            dfrecoskml = apply("BinaryClassification", [self.p_modelname], [mod],
                               dfrecosk, self.v_train)
            probvar = "y_test_prob" + self.p_modelname
            dfrecoskml = dfrecoskml.loc[dfrecoskml[probvar] > self.lpt_probcutpre[ipt]]
            dfrecoskml.to_pickle(self.mptfiles_recoskmldec[ipt][file_index])


    def parallelizer(self, function, argument_list, maxperchunk):
        chunks = [argument_list[x:x+maxperchunk] \
                  for x in range(0, len(argument_list), maxperchunk)]

        # If finding jets, we want to save a dataframe of jet info
        jet_df = pd.DataFrame(columns=self.jetRadii)

        for chunk in chunks:
            print("Processing new chunk size=", maxperchunk)
            pool = mp.Pool(self.p_maxprocess)
            return_vals = [pool.apply_async(function, args=chunk[i]).get() for i in range(len(chunk))]
            print("Return vals: ", return_vals)

            if function == self.findJets:
                for i in return_vals:
                    jet_df.concat(i)

            pool.close()
            pool.join()

        if function == self.findJets:
            return jet_df
        return 0
            

    def process_unpack_par(self):
        print("doing unpacking", self.mcordata, self.period)
        create_folder_struc(self.d_pkl, self.l_path)
        arguments = [(i,) for i in range(len(self.l_root))]
        self.parallelizer(self.unpack, arguments, self.p_chunksizeunp)

    def process_skim_par(self):
        print("doing skimming", self.mcordata, self.period)
        create_folder_struc(self.d_pklsk, self.l_path)
        arguments = [(i,) for i in range(len(self.l_reco))]
        self.parallelizer(self.skim, arguments, self.p_chunksizeskim)
        merge_method(self.l_evt, self.f_totevt)
        merge_method(self.l_evtorig, self.f_totevtorig)

    def process_applymodel_par(self):
        print("doing apply model", self.mcordata, self.period)
        create_folder_struc(self.d_pkl_dec, self.l_path)
        for ipt in range(self.p_nptbins):
            arguments = [(i,) for i in range(len(self.mptfiles_recosk[ipt]))]
            self.parallelizer(self.applymodel, arguments, self.p_chunksizeskim)

    def process_mergeforml(self):
        nfiles = len(self.mptfiles_recosk[0])
        if nfiles == 0:
            print("increase the fraction of merged files or the total number")
            print(" of files you process")
        ntomerge = (int)(nfiles * self.p_frac_merge)
        rd.seed(self.p_rd_merge)
        filesel = rd.sample(range(0, nfiles), ntomerge)
        for ipt in range(self.p_nptbins):
            list_sel_recosk = [self.mptfiles_recosk[ipt][j] for j in filesel]
            merge_method(list_sel_recosk, self.lpt_reco_ml[ipt])
            if self.mcordata == "mc":
                list_sel_gensk = [self.mptfiles_gensk[ipt][j] for j in filesel]
                merge_method(list_sel_gensk, self.lpt_gen_ml[ipt])

        list_sel_evt = [self.l_evt[j] for j in filesel]
        list_sel_evtorig = [self.l_evtorig[j] for j in filesel]
        merge_method(list_sel_evt, self.f_evt_ml)
        merge_method(list_sel_evtorig, self.f_evtorig_ml)

    def process_mergedec(self):
        for ipt in range(self.p_nptbins):
            merge_method(self.mptfiles_recoskmldec[ipt], self.lpt_recodecmerged[ipt])
            if self.mcordata == "mc":
                merge_method(self.mptfiles_gensk[ipt], self.lpt_gendecmerged[ipt])

    def process_histomass(self):
        for ipt in range(self.p_nptbins):
            myfile = TFile.Open(self.lpt_filemass[ipt], "recreate")
            df = pickle.load(open(self.lpt_recodecmerged[ipt], "rb"))
            df = df.query(self.l_selml[ipt])
            h_invmass = TH1F("hmass", "", self.p_num_bins,
                             self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
            fill_hist(h_invmass, df.inv_mass)
            myfile.cd()
            h_invmass.Write()

    def process_efficiency(self):
        n_bins = len(self.lpt_anbinmin)
        analysis_bin_lims_temp = list(self.lpt_anbinmin)
        analysis_bin_lims_temp.append(self.lpt_anbinmax[n_bins-1])
        analysis_bin_lims = array.array('f', analysis_bin_lims_temp)

        h_gen_pr = TH1F("h_gen_pr", "Prompt Generated in acceptance |y|<0.5", \
                        n_bins, analysis_bin_lims)
        h_presel_pr = TH1F("h_presel_pr", "Prompt Reco in acc |#eta|<0.8 and sel", \
                           n_bins, analysis_bin_lims)
        h_sel_pr = TH1F("h_sel_pr", "Prompt Reco and sel in acc |#eta|<0.8 and sel", \
                        n_bins, analysis_bin_lims)
        h_gen_fd = TH1F("h_gen_fd", "FD Generated in acceptance |y|<0.5", \
                        n_bins, analysis_bin_lims)
        h_presel_fd = TH1F("h_presel_fd", "FD Reco in acc |#eta|<0.8 and sel", \
                           n_bins, analysis_bin_lims)
        h_sel_fd = TH1F("h_sel_fd", "FD Reco and sel in acc |#eta|<0.8 and sel", \
                        n_bins, analysis_bin_lims)
        h_gen_pr = TH1F("h_gen_pr", "Prompt Generated in acceptance |y|<0.5", \
                        n_bins, analysis_bin_lims)
        h_presel_pr = TH1F("h_presel_pr", "Prompt Reco in acc |#eta|<0.8 and sel", \
                           n_bins, analysis_bin_lims)
        h_sel_pr = TH1F("h_sel_pr", "Prompt Reco and sel in acc |#eta|<0.8 and sel", \
                        n_bins, analysis_bin_lims)
        h_gen_fd = TH1F("h_gen_fd", "FD Generated in acceptance |y|<0.5", \
                        n_bins, analysis_bin_lims)
        h_presel_fd = TH1F("h_presel_fd", "FD Reco in acc |#eta|<0.8 and sel", \
                           n_bins, analysis_bin_lims)
        h_sel_fd = TH1F("h_sel_fd", "FD Reco and sel in acc |#eta|<0.8 and sel", \
                        n_bins, analysis_bin_lims)

        bincounter = 0
        for ipt in range(self.p_nptbins):
            df_mc_reco = pd.read_pickle(self.lpt_recodecmerged[ipt])
            df_mc_gen = pd.read_pickle(self.lpt_gendecmerged[ipt])
            df_mc_gen = df_mc_gen.query(self.s_presel_gen_eff)
            df_gen_sel_pr = df_mc_gen[df_mc_gen.ismcprompt == 1]
            df_reco_presel_pr = df_mc_reco[df_mc_reco.ismcprompt == 1]
            df_reco_sel_pr = df_reco_presel_pr.query(self.l_selml[ipt])
            df_gen_sel_fd = df_mc_gen[df_mc_gen.ismcfd == 1]
            df_reco_presel_fd = df_mc_reco[df_mc_reco.ismcfd == 1]
            df_reco_sel_fd = df_reco_presel_fd.query(self.l_selml[ipt])

            h_gen_pr.SetBinContent(bincounter + 1, len(df_gen_sel_pr))
            h_gen_pr.SetBinError(bincounter + 1, math.sqrt(len(df_gen_sel_pr)))
            h_presel_pr.SetBinContent(bincounter + 1, len(df_reco_presel_pr))
            h_presel_pr.SetBinError(bincounter + 1, math.sqrt(len(df_reco_presel_pr)))
            h_sel_pr.SetBinContent(bincounter + 1, len(df_reco_sel_pr))
            h_sel_pr.SetBinError(bincounter + 1, math.sqrt(len(df_reco_sel_pr)))
            print("prompt efficiency tot ptbin=", bincounter, ", value = ",
                  len(df_reco_sel_pr)/len(df_gen_sel_pr))

            h_gen_fd.SetBinContent(bincounter + 1, len(df_gen_sel_fd))
            h_gen_fd.SetBinError(bincounter + 1, math.sqrt(len(df_gen_sel_fd)))
            h_presel_fd.SetBinContent(bincounter + 1, len(df_reco_presel_fd))
            h_presel_fd.SetBinError(bincounter + 1, math.sqrt(len(df_reco_presel_fd)))
            h_sel_fd.SetBinContent(bincounter + 1, len(df_reco_sel_fd))
            h_sel_fd.SetBinError(bincounter + 1, math.sqrt(len(df_reco_sel_fd)))
            print("fd efficiency tot ptbin=", bincounter, ", value = ",
                  len(df_reco_sel_fd)/len(df_gen_sel_fd))
            bincounter = bincounter + 1
        out_file = TFile.Open(self.n_fileeff, "recreate")
        out_file.cd()
        h_gen_pr.Write()
        h_presel_pr.Write()
        h_sel_pr.Write()
        h_gen_fd.Write()
        h_presel_fd.Write()
        h_sel_fd.Write()


    # REQUIRES list of particles [(pT, eta, phi, m), (pT, eta, phi, m), ...]
    #          and the jet radius jetR desired for reconstruction
    # RETURNS list of clustered jets
    #         [ [jet_pT, jet_eta, jet_phi, jet_m, (pT, eta, phi, m), ...],
    #           [jet_pT, jet_eta, jet_phi, jet_m, (pT, eta, phi, m), ...], ... ]
    def findJetsSingleEvent(self, particles, jetR):
        jets = pyjet.cluster(particles, R=jetR, p=-1).inclusive_jets()
        jetList = []
        for jet in jets:
            etaMax = self.datap['variables']['etaMax'] - jetR
            if abs(jet.eta) < etaMax:
                l = [jet.pt, jet.eta, jet.phi, jet.mass] + list(jet.constituents_array())
                jetList.append(l)
        return jetList


    # Does anti-kT jet-finding on given reconstructed particle tracks in an event
    # and returns a dataframe of the jets for given jet radius:
    # ________|____________JetRadii______________|
    # ev_id   |  JetR  |  JetR  |  JetR  |  ...  |
    def findJets(self, file_index):
        # Open root file and save particle tree to dataframe
        treereco = uproot.open(self.l_root[file_index])[self.n_treereco]
        if not treereco:
            print('Couldn\'t find tree %s in file %s' % \
                  (self.n_treereco, self.l_root[file_index]))
            return pd.DataFrame(columns=self.jetRadii)
        dfreco = treereco.pandas.df(branches=self.v_all)

        # Create list of particles per event for jet finding
        df_iter = dfreco.iterrows()
        dt = np.dtype([('pT', 'f8'), ('eta', 'f8'), ('phi', 'f8'), ('mass', 'f8')])
        current_ev = np.array([], dtype=dt)
        jet_df = pd.DataFrame(columns=self.jetRadii)
        prev_ev_id = None
        try:
            while True:
                # Iterate through each column in the dataframe
                row = next(df_iter)[1]

                # Check to see if this is the same event or a new one
                if row['ev_id'] != prev_ev_id:
                    if prev_ev_id != None:
                        for jetR in self.jetRadii:
                            jet_df.at[prev_ev_id, jetR] = self.findJetsSingleEvent(current_ev, jetR)
                        current_ev = np.array([], dtype=dt)
                    prev_ev_id = int(row['ev_id'])

                # For now just estimate everything as having pion mass
                mass = 0.1396   # GeV/c^2
                particle = np.array([(row['ParticlePt'], row['ParticleEta'], 
                                      row['ParticlePhi'], mass)], dtype=dt)
                current_ev = np.append(current_ev, particle)

        except StopIteration:
            # Save final event from dataframe to list
            for jetR in self.jetRadii:
                jet_df.at[prev_ev_id, jetR] = self.findJetsSingleEvent(current_ev, jetR)

        return jet_df


    def find_jets_all(self):
        print("doing jet finding", self.mcordata, self.period)
        arguments = [(i,) for i in range(len(self.l_root))]
        return self.parallelizer(self.findJets, arguments, self.p_chunksizeunp)


    # Calculate the jet substructure variable lambda for given jet, beta, kappa, and jet R)
    def calcLambda_single_jet(jet, b, k, jet_R):
        
        # Sanity check
        if len(jet) < 5:
            print("ERROR! Jet is nonsensical. Check jet-finding algorithm in processer.findJets")
            exit(1)

        # Read jet info for calculations
        jet_pT = jet[0]
        jet_eta = jet[1]
        jet_phi = jet[2]

        # Calculate & sum lambda for all jet constituents
        lambda_bk = 0
        for constituent in jet:
            eta = constituent[1]
            phi = constituent[2]
            deltaR = np.sqrt( (jet_eta - eta)**2 + (jet_phi - phi)**2 )
            lambda_bk += (constituent[0] / jet_pT)**k * (deltaR / jet_R)**b

        return lambda_bk


    # Returns the expected pT bin index if given pT within bins, else returns -1
    def get_pT_bin_num(self, pT):

        for bin, pTmin in list(enumerate(self.pTbins))[0:-1]:
            pTmax = self.pTbins[bin+1]
            if pTmin <= pT <= pTmax:
                return bin

        return -1


    # Generates list of dataframes with expected structure for each pT bin, e.g.:
    #      ____________ Jet R _____________     
    #      |     | 0.1  0.2  0.3  0.4  ...
    #      | 1.0 | []   []   []   []
    # beta | 1.5 | []   []   []   []
    #      | 2.0 | []   []   []   []
    def gen_lambdas_per_pT_bin(self):
        lambdas_per_bin = []
        for i, pTmin in list(enumerate(self.pTbins))[0:-1]:
            lambdas_per_bin.append(pd.DataFrame(columns=self.jetRadii, index=self.betas))
            # Initialize each value of the dataframe with an empty list
            for jetR in self.jetRadii:
                lambdas_per_bin[i][jetR] = np.empty((len(lambdas_per_bin[i]), 0)).tolist()
        return lambdas_per_bin


    # Calculate the jet substructure variable lambda_k for all values
    # of k indicated in the configuration file
    def calc_lambda(self):

        # Get the events & their corresponding jets (if not already done)
        if self.jets == None:
            self.jets = self.find_jets_all()
        print(self.jets)

        # We want to create a different analysis for each pT bin
        lambdas_per_bin = self.gen_lambdas_per_pT_bin()

        # Calculate histogram entries for each element in dataframe
        for ev_id, row in self.jets.iterrows():
            for jetR in self.jetRadii:
                for beta in self.betas:
                    for jet in row[jetR]:
                        bin = get_pT_bin_num(jet[0])
                        if bin != -1:
                            kappa = 1   # just use this for now
                            l = self.calcLambda_single_jet(jet, beta, kappa, jetR)
                            lambdas_per_bin[bin].at[beta, jetR].append(l)

        # Return list of dataframes per pT bin
        return lambdas_per_bin
