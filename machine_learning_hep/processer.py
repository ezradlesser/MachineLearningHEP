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
from itertools import chain
import random as rd
import uproot
import pandas as pd
import numpy as np
from pyjet import cluster, testdata
from root_numpy import fill_hist # pylint: disable=import-error, no-name-in-module
from ROOT import TFile, TH1F # pylint: disable=import-error, no-name-in-module

from machine_learning_hep.selectionutils import selectfidacc
from machine_learning_hep.bitwise import filter_bit_df, tag_bit_df
from machine_learning_hep.utilities import selectdfquery, selectdfrunlist, merge_method, \
    list_folders, createlist, appendmainfoldertolist, create_folder_struc, seldf_singlevar, \
    get_pT_bin_list
from machine_learning_hep.models import apply # pylint: disable=import-error

class Processer: # pylint: disable=too-many-instance-attributes
    # Class Attribute
    species = 'processer'

    # Initializer / Instance Attributes
    # pylint: disable=too-many-statements, too-many-arguments
    def __init__(self, case, datap, run_param, mcordata, p_maxfiles,
                 d_root, d_pkl, d_pklsk, d_pkl_ml, p_period,
                 p_chunksizeunp, p_chunksizeskim, p_chunksizejet, p_maxprocess,
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
        self.p_chunksizejet = p_chunksizejet

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
        self.n_jet = datap["files_names"]["namefile_jets"]
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

        self.ignore_prev_jet_calc = datap["ignore_prev_jet_calc"] if ('Jet' in case) else None
        self.l_root = createlist(self.d_root, self.l_path, self.n_root)
        self.l_reco = createlist(self.d_pkl, self.l_path, self.n_reco)
        self.l_evt = createlist(self.d_pkl, self.l_path, self.n_evt)
        self.l_jet = createlist(self.d_pkl, self.l_path, self.n_jet) if ('Jet' in case) else None
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
            #self.jets = None   # Will fill this in if needed using findJets()

            # 4D list of binned lambda values. Filled during jet-finding.
            # [ (pTbin1:) [ (jetR1:) [ (beta1:) [lambdabin1, lambdabin2, ... ],
            #                          (beta2:) [ ... ], 
            #                          ... ],
            #               (jetR2:) [ ... ],
            #               ... ],
            #   (pTbin2:) [ ... ],
            #   ... ]
            self.jet_lambda = None
            self.n_lambda_bins = datap['variables']['N_lambda_bins']
            self.lambda_max = datap['variables']['lambda_max']

            # 2D list of dictionaries of N_jet_constits. Filled during jet-finding.
            # [ (pTbin1:) [ (jetR1:) { "2": N_2, "3": N_3, ... },
            #               (jetR2:) [ ... ],
            #               ... ],
            #   (pTbin2:) [ ... ],
            #   ... ]
            self.N_constits = None

        else: #if 'Jet' not in case:
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


    def init_jet_lambda(self):
        return np.array([ [ [ [0] * self.n_lambda_bins for k in self.betas] 
                            for j in range(self.n_lambda_bins) ] 
                          for i in list(self.pTbins)[0:-1] ])

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
        '''
        # If finding jets, we want to save a dataframe of jet info
        jet_df = pd.DataFrame(columns=self.jetRadii)
        '''

        for chunk in chunks:
            print("Processing new chunk size =", maxperchunk)
            pool = mp.Pool(self.p_maxprocess)
            return_vals = [pool.apply_async(function, args=chunk[i]) for i in range(len(chunk))]

            pool.close()
            pool.join()

            '''
            if function == self.findJets:
                for i in np.arange(len(return_vals)):
                    return_vals[i] = return_vals[i].get()
                l = [jet_df] + return_vals
                jet_df = pd.concat(l)
                # For testing, just get some events quickly
                if len(jet_df) > 1e6:
                    break
            '''

            if function == self.calc_lambda:
                for i in np.arange(len(return_vals)):
                    return_vals[i] = return_vals[i].get()
                for li in return_vals:
                    self.jet_lambda = np.add(self.jet_lambda, li)
                # For now don't have to run over the whole data set
                #if sum(self.jet_lambda[0][0][0]) > 1:
                #    return 0

        '''
        if function == self.findJets:
            print("Jet finding complete. Total number of events: %i" % len(jet_df))
            return jet_df
        '''
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


    # Return True if jet to be cut; False otherwise
    def cut_jet(self, jet, jetR):
        etaMax = self.datap['variables']['etaMax'] - jetR
        if abs(jet.eta) > etaMax:   # Entire jet must fit within desired region
            return True
        elif len(jet.constituents_array()) < 2:  # Jet must have more than one constituent
            return True
        return False


    # REQUIRES list of particles [(pT, eta, phi, m), (pT, eta, phi, m), ...]
    #          and the jet radius jetR desired for reconstruction
    # RETURNS list of clustered jets
    #         [ [jet_pT, jet_eta, jet_phi, jet_m, (pT, eta, phi, m), ...],
    #           [jet_pT, jet_eta, jet_phi, jet_m, (pT, eta, phi, m), ...], ... ]
    def findJetsSingleEvent(self, particles, jetR):
        jets = cluster(particles, R=float(jetR), p=-1).inclusive_jets()  # p=-1  -->  anti-kT
        jetList = []
        for jet in jets:
            if not self.cut_jet(jet, jetR):
                l = [jet.pt, jet.eta, jet.phi, jet.mass] + list(jet.constituents_array())
                jetList.append(l)
        return jetList

    # Does jet-finding on given reconstructed particle tracks in an event
    # and returns a dataframe of the jets for given jet radius:
    # ________|____________JetRadii______________|
    # ev_id   |  JetR  |  JetR  |  JetR  |  ...  |
    def findJets(self, file_index):

        # Check if jet calculation already exists
        if not self.ignore_prev_jet_calc and os.path.exists(self.l_jet[file_index]):
            print("Loading previously calculated jets at %s" % self.l_jet[file_index])
            try:
                return pd.read_pickle(self.l_jet[file_index])
            except:
                print("Pickle file failed to load. Recalculating jets for this file.")

        # Open root file and save particle tree to dataframe
        print(self.l_root[file_index], self.n_treereco)
        treereco = None
        try:
            treereco = uproot.open(self.l_root[file_index])[self.n_treereco]
        except:
            print("Unable to open ROOT TTree for particle data. Please check directory and filepath.")
            return pd.DataFrame(columns=self.jetRadii)
        if not treereco:
            print('Couldn\'t find tree %s in file %s' % \
                  (self.n_treereco, self.l_root[file_index]))
            return pd.DataFrame(columns=self.jetRadii)

        dfreco = treereco.pandas.df(branches=self.v_all)
        dt = np.dtype([('pT', 'f8'), ('eta', 'f8'), ('phi', 'f8'), ('mass', 'f8')])
        # For now just estimate everything as having pi meson mass
        mass = 0.1396   # GeV/c^2
        particles = [np.array([(row[0], row[1], row[2], mass)], dtype=dt)[0] 
                     for row in dfreco[['ParticlePt', 'ParticleEta', 'ParticlePhi']].values]
        dfreco = pd.DataFrame({'ev_id': dfreco['ev_id'], 'particles': particles}).groupby(
            'ev_id', sort=False)['particles'].apply(lambda x: np.array(x, dtype=dt))

        # Find jets for each event
        jet_df = pd.DataFrame(columns=self.jetRadii, index=dfreco.index.tolist())
        for jetR in self.jetRadii:
            jet_df[jetR] = [self.findJetsSingleEvent(event, jetR) for event in dfreco]

        print("Successfully found jets from %i events in %s" % 
              (len(jet_df), self.l_root[file_index]))

        # Save jets to a pickle file so that they don't have to be calculated again
        os.makedirs(os.path.dirname(self.l_jet[file_index]), exist_ok=True)
        jet_df.to_pickle(self.l_jet[file_index])

        return jet_df


    def find_jets_all(self):
        print("doing jet finding", self.mcordata, self.period)
        arguments = [(i,) for i in range(len(self.l_root))]
        return self.parallelizer(self.findJets, arguments, self.p_chunksizejet)


    # Calculate the jet substructure variable lambda for given jet, beta, kappa, and jet R)
    def calc_lambda_single_jet(self, jet, b, k, jet_R):

        # Calculate & sum lambda for all jet constituents
        lambda_bk = 0
        for constituent in jet[4:]:
            deltaR = np.sqrt( (jet[1] - constituent[1])**2 + (jet[2] - constituent[2])**2 )
            lambda_bk += (constituent[0] / jet[0])**k * (deltaR / jet_R)**b

        return lambda_bk


    # Returns the expected pT bin index if given pT within bins, else returns -1
    def get_pTbin_num(self, pT):

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

    '''
    def get_pT_jet_list(self, bin_num, beta, jetR, jet_list):
        li = []
        kappa = 1   # just use this for now
        for jet in jet_list:
            jet_bin_num = self.get_pTbin_num(jet[0])
            if bin_num != jet_bin_num:
                continue
            li.append(self.calc_lambda_single_jet(jet, beta, kappa, jetR))
        return li
    '''

    # Returns list of tuples [ (pTbin_num, lambda), ...] 
    def pTbin_lambdas_list(self, beta, jetR, jet_list):
        kappa = 1  # just use this for now
        li = [ ( self.get_pTbin_num(jet[0]), 
                 self.calc_lambda_single_jet(jet, beta, kappa, jetR) )
               for jet in jet_list ]
        return [ tupl for tupl in li if tupl[0] >= 0 ]


    def get_num_jet_constits(self, jet_list):
        num_constits = []
        for jet in jet_list:
            num_constits.append( len(jet) - 4 )
        return num_constits


    def get_jet_pT_dist(self, jet_list):
        jet_pT = []
        for jet in jet_list:
            jet_pT.append( jet[0] )
        return jet_pT


    def get_z_dist(self, jet_list):
        z = []
        for jet in jet_list:
            for constit in jet[4:]:
                z.append(constit[0] / jet[0])
        return z

    '''
    # Calculate the jet substructure variable lambda_k for all values
    # of k indicated in the configuration file
    def calc_lambda(self):

        # Get the events & their corresponding jets (if not already done)
        if self.jets == None:
            self.jets = self.find_jets_all()

        # We want to create a different analysis for each pT bin
        lambdas_per_bin = self.gen_lambdas_per_pT_bin()

        pT_bin_list = get_pT_bin_list(self.pTbins)
        for bin_num in np.arange(len(self.pTbins) - 1):
            print("Calculating lambda observable for jet pT = %s GeV/c" % pT_bin_list[bin_num])
            lambdas = lambdas_per_bin[bin_num]        
            for jetR in self.jetRadii:
                lambdas[jetR] = [list(chain(
                                 *[self.get_pT_jet_list(bin_num, beta, jetR, x) 
                                   for x in self.jets[jetR]])) for beta in self.betas]
        # Return list of dataframes per pT bin
        return lambdas_per_bin
    '''

    def calc_lambda(self, file_index):
        jet_df = self.findJets(file_index)

        jet_lambda = self.init_jet_lambda()
        norm = self.lambda_max / self.n_lambda_bins
        for k, beta in enumerate(self.betas):
            for j, jetR in enumerate(self.jetRadii):
                lambdas = list(chain(*[self.pTbin_lambdas_list(beta, jetR, jet_list)
                                       for jet_list in jet_df[jetR]]))
                for pTbin_num, l in lambdas:
                    lambda_bin = math.floor(l / norm)
                    if lambda_bin < self.n_lambda_bins:
                        jet_lambda[pTbin_num][j][k][lambda_bin] += 1
                        #print("[%i][%i][%i][%i]: %i" % (pTbin_num, j, k, lambda_bin, 
                        #                                jet_lambda[pTbin_num][j][k][lambda_bin]))
                
                # TODO: Implement saving this somehow
                '''
                N_constit = list(chain([self.get_num_jet_constits(jet_list) 
                                        for jet_list in jet_df[jetR]]))
                pT_dist = list(chain([self.get_jet_pT_dist(jet_list) 
                                      for jet_list in jet_df[jetR]]))
                z_dist = list(chain([self.get_z_dist(jet_list)
                                     for jet_list in jet_df[jetR]]))
                '''
        return jet_lambda

    def calc_lambda_dist(self):

        # Make sure jet lambda histograms have been initialized
        if self.jet_lambda == None:
            print("Initializing jet lambda histograms...")
            self.jet_lambda = self.init_jet_lambda()

        print("Starting calculation of lambda observable", self.mcordata, self.period)
        arguments = [(i,) for i in range(len(self.l_root))]
        self.parallelizer(self.calc_lambda, arguments, self.p_chunksizejet)
        return self.jet_lambda

    '''
    # Calculate general jet distributions: pT, z=pT,track/pT,jet, N constituents
    def calc_gen_jet_plots(self):

        # Get the events & their corresponding jets (if not already done)
        if self.jets == None:
            self.jets = self.find_jets_all()

        print("Calculating general jet distributions: pT, z=pT,track/pT,jet, N constituents...")
        pT_dist = pd.DataFrame(columns=self.jetRadii)
        N_constit = pd.DataFrame(columns=self.jetRadii)
        z_dist = pd.DataFrame(columns=self.jetRadii)
        for jetR in self.jetRadii:
            N_constit[jetR] = list(chain([self.get_num_jet_constits(jet_list) 
                                          for jet_list in self.jets[jetR]]))
            pT_dist[jetR] = list(chain([self.get_jet_pT_dist(jet_list) 
                                        for jet_list in self.jets[jetR]]))
            z_dist[jetR] = list(chain([self.get_z_dist(jet_list)
                                       for jet_list in self.jets[jetR]]))
            
        return N_constit, pT_dist, z_dist
    '''
