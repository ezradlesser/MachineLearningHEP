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

from __future__ import division, print_function
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from machine_learning_hep.processer import Processer
from machine_learning_hep.utilities import merge_method, concat_dir, \
    create_folder_struc, get_pT_bin_list


#---------------------------------------------------------------------------------------------------
class MultiProcesser: # pylint: disable=too-many-instance-attributes, too-many-statements
    species = "multiprocesser"
    def __init__(self, case, datap, run_param, mcordata):
      
        # Required parameters
        self.case = case
        self.datap = datap
        self.run_param = run_param
        self.mcordata = mcordata
        self.prodnumber = len(datap["multi"][mcordata]["unmerged_tree_dir"])
        self.p_period = datap["multi"][mcordata]["period"]
        self.p_seedmerge = datap["multi"][mcordata]["seedmerge"]
        self.p_fracmerge = datap["multi"][mcordata]["fracmerge"]
        self.p_maxfiles = datap["multi"][mcordata]["maxfiles"]
        self.p_chunksizeunp = datap["multi"][mcordata]["chunksizeunp"]
        self.p_chunksizeskim = datap["multi"][mcordata]["chunksizeskim"]
        self.p_chunksizejet = datap["multi"][mcordata]["chunksizejet"]
        self.p_nparall = datap["multi"][mcordata]["nprocessesparallel"]
        self.lpt_anbinmin = datap["sel_skim_binmin"]
        self.lpt_anbinmax = datap["sel_skim_binmax"]
        self.p_nptbins = len(datap["sel_skim_binmax"])
        self.dlper_root = datap["multi"][mcordata]["unmerged_tree_dir"]
        self.dlper_pkl = datap["multi"][mcordata]["pkl"]
        self.dlper_pklsk = datap["multi"][mcordata]["pkl_skimmed"]
        self.d_pklevt_mergedallp = datap["multi"][mcordata]["pkl_evtcounter_all"]
        self.n_reco = datap["files_names"]["namefile_reco"]
        self.n_evt = datap["files_names"]["namefile_evt"]
        self.n_evtorig = datap["files_names"]["namefile_evtorig"]
        self.n_gen = datap["files_names"]["namefile_gen"]
        self.lpt_recosk = [self.n_reco.replace(".pkl", "%d_%d.pkl" % \
                          (self.lpt_anbinmin[i], self.lpt_anbinmax[i])) \
                          for i in range(self.p_nptbins)]
        self.lpt_gensk = [self.n_gen.replace(".pkl", "%d_%d.pkl" % \
                          (self.lpt_anbinmin[i], self.lpt_anbinmax[i])) \
                          for i in range(self.p_nptbins)]
        self.lper_evt = [os.path.join(direc, self.n_evt) for direc in self.dlper_pkl]
        self.lper_evtorig = [os.path.join(direc, self.n_evtorig) for direc in self.dlper_pkl]
        self.d_results = datap["analysis"][mcordata]["results"]
        self.f_evt_mergedallp = os.path.join(self.d_pklevt_mergedallp, self.n_evt)
        self.f_evtorig_mergedallp = os.path.join(self.d_pklevt_mergedallp, self.n_evtorig)
        
        # Optional parameters
        self.dlper_pklml = self.d_pklml_mergedallp = self.lper_evtml = \
            self.lper_evtorigml = self.lptper_recoml = self.f_evtorigml_mergedallp = \
            self.lptper_genml = self.lpt_recoml_mergedallp = self.lpt_genml_mergedallp = \
            self.f_evtml_mergedallp = self.dlper_reco_modapp = \
            self.dlper_reco_modappmerged = self.lpt_probcutpre = self.lpt_probcut = None

        if 'Jet' not in case:
            self.dlper_pklml = datap["multi"][mcordata]["pkl_skimmed_merge_for_ml"]
            self.d_pklml_mergedallp = datap["multi"][mcordata]["pkl_skimmed_merge_for_ml_all"]
            self.lper_evtml = [os.path.join(direc, self.n_evt) for direc in self.dlper_pklml]
            self.lper_evtorigml = [os.path.join(direc, self.n_evtorig) for direc in self.dlper_pklml]
            self.lptper_recoml = [[os.path.join(direc, self.lpt_recosk[ipt]) \
                                   for direc in self.dlper_pklml] \
                                  for ipt in range(self.p_nptbins)]
            self.f_evtorigml_mergedallp = os.path.join(self.d_pklml_mergedallp, self.n_evtorig)
            self.lptper_genml = [[os.path.join(direc, self.lpt_gensk[ipt]) \
                                  for direc in self.dlper_pklml] \
                                 for ipt in range(self.p_nptbins)]
            self.lpt_recoml_mergedallp = [os.path.join(self.d_pklml_mergedallp, self.lpt_recosk[ipt]) \
                                          for ipt in range(self.p_nptbins)]
            self.lpt_genml_mergedallp = [os.path.join(self.d_pklml_mergedallp, self.lpt_gensk[ipt]) \
                                         for ipt in range(self.p_nptbins)]
            self.f_evtml_mergedallp = os.path.join(self.d_pklml_mergedallp, self.n_evt)
            self.dlper_reco_modapp = datap["analysis"][mcordata]["pkl_skimmed_dec"]
            self.dlper_reco_modappmerged = datap["analysis"][mcordata]["pkl_skimmed_decmerged"]
            self.lpt_probcutpre = datap["analysis"]["probcutpresel"]
            self.lpt_probcut = datap["analysis"]["probcutoptimal"]
        else:   # 'Jet' in case
            self.jetRadii = datap["variables"]["jetRadii"]
            self.pTbins = datap["variables"]["pTbins"]
            self.betas = datap["variables"]["betas"]
            self.j_dir = self.datap["multi"][self.mcordata]["jet_plot_dir"]
            self.pTbins = self.datap["variables"]["pTbins"]
            self.n_lambda_bins = datap['variables']['N_lambda_bins']
            self.lambda_max = datap['variables']['lambda_max']

        self.process_listsample = []
        for indexp in range(self.prodnumber):
            self_dlper_pklml = self_dlper_reco_modapp = self_dlper_reco_modappmerged = None
            if 'Jet' not in case:
              self_dlper_pklml = self.dlper_pklml[indexp]
              self_dlper_reco_modapp = self.dlper_reco_modapp[indexp]
              self_dlper_reco_modappmerged = self.dlper_reco_modappmerged[indexp]
            
            myprocess = Processer(self.case, self.datap, self.run_param, self.mcordata,
                                  self.p_maxfiles[indexp], self.dlper_root[indexp],
                                  self.dlper_pkl[indexp], self.dlper_pklsk[indexp],
                                  self_dlper_pklml, self.p_period[indexp], 
                                  self.p_chunksizeunp[indexp], self.p_chunksizeskim[indexp],
                                  self.p_chunksizejet[indexp], self.p_nparall,
                                  self.p_fracmerge[indexp], self.p_seedmerge[indexp],
                                  self_dlper_reco_modapp,
                                  self_dlper_reco_modappmerged,
                                  self.d_results[indexp])
            self.process_listsample.append(myprocess)


    def multi_unpack_allperiods(self):
        for indexp in range(self.prodnumber):
            self.process_listsample[indexp].process_unpack_par()

    def multi_skim_allperiods(self):
        for indexp in range(self.prodnumber):
            self.process_listsample[indexp].process_skim_par()
        merge_method(self.lper_evt, self.f_evt_mergedallp)
        merge_method(self.lper_evtorig, self.f_evtorig_mergedallp)

    def multi_mergeml_allperiods(self):
        for indexp in range(self.prodnumber):
            self.process_listsample[indexp].process_mergeforml()

    def multi_mergeml_allinone(self):
        for ipt in range(self.p_nptbins):
            merge_method(self.lptper_recoml[ipt], self.lpt_recoml_mergedallp[ipt])
            if self.mcordata == "mc":
                merge_method(self.lptper_genml[ipt], self.lpt_genml_mergedallp[ipt])
        merge_method(self.lper_evtml, self.f_evtml_mergedallp)
        merge_method(self.lper_evtorigml, self.f_evtorigml_mergedallp)

    def multi_apply_allperiods(self):
        for indexp in range(self.prodnumber):
            self.process_listsample[indexp].process_applymodel_par()

    def multi_mergeapply_allperiods(self):
        for indexp in range(self.prodnumber):
            self.process_listsample[indexp].process_mergedec()

    def multi_histomass(self):
        for indexp in range(self.prodnumber):
            self.process_listsample[indexp].process_histomass()

    def multi_efficiency(self):
        for indexp in range(self.prodnumber):
            self.process_listsample[indexp].process_efficiency()

    def merge_lambdas(self, lambdas1, lambdas2):
        if lambdas1 == None:
            return lambdas2
        return np.add(lambdas1, lambdas2)

    def calc_jet_lambda(self):
        lambdas_per_bin = None

        for indexp in range(self.prodnumber):
            print("Calculating jet lambda substructure observable for %s" % self.dlper_root[indexp])
            lambdas_list = self.process_listsample[indexp].calc_lambda_dist()
            lambdas_per_bin = self.merge_lambdas(lambdas_per_bin, lambdas_list)

        return lambdas_per_bin

    def calc_gen_jet(self):
        N_constit = pT_dist = z_dist = None
        for indexp in range(self.prodnumber):
            print("Calculating genereal jet distributions for %s" % self.dlper_root[indexp])
            N_constit_part, pT_dist_part, z_dist_part = \
                self.process_listsample[indexp].calc_gen_jet_plots()
            N_constit.merge(N_constit_part)
            pT_dist.merge(pT_dist_part)
            z_dist.merge(z_dist_part)
        return N_constit, pT_dist, z_dist

    def initialize_lambda_plots(self, pTmin, pTmax):

        plt.subplots_adjust(wspace=3, hspace=3)
        plt.rc('font', size=20)          # controls main title size (defalut text size below)
        plt.rc('axes', labelsize=18)     # fontsize of the x and y labels
        plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=14)    # fontsize of the tick labels

        # Create figure for tau_alpha plot & arrange subplots
        fig, a = plt.subplots(len(self.betas), len(self.jetRadii))
        fig.suptitle(r'Calculated $\lambda_\beta$ values for $p_{T, jet}=%s-%s$' % 
                     (str(pTmin), str(pTmax)))
        plt.rc('font', size=14)          # controls default text sizes        
        fig.text(0.5, 0.02, 'Jet Radius', ha='center', va='center')   # Major x-axis label
        fig.text(0.02, 0.5, 'Beta', ha='center', va='center', 
                 rotation='vertical')                                 # Major y-axis label
        fig.set_size_inches(27, 15)

        return fig, a

    def plot_jet_lambda(self):
        # Create directroy to save files
        pTbins_ranges = get_pT_bin_list(self.pTbins)
        create_folder_struc(self.j_dir, pTbins_ranges)

        # Get the list of dataframes containing lambdas info per pT bin, beta, & jet R
        lambdas_per_bin = self.calc_jet_lambda()

        # Make plots per jet pT bin
        print("Plotting jet lambda observable per pT bin")
        for i, pTmin in list(enumerate(self.pTbins))[0:-1]:
            pTmax = self.pTbins[i + 1]
            fig, a = self.initialize_lambda_plots(pTmin, pTmax)

            # Create subplot per jet R and beta (for each individual pT bin)
            for j, jetR in enumerate(self.jetRadii):
                for k, beta in enumerate(self.betas):
                    #lambda_list = lambdas_per_bin[i].at[beta, jetR]
                    binned_lambdas = lambdas_per_bin[i][j][k]
                    ax = a[k, j]

                    ''' old df sln
                    # Normalize histogram by 1/N
                    wg = np.ones(len(lambda_list)) / len(lambda_list)
                    # Approx bin size for good statistics
                    n_bins = 100 #len(lambda_list) // (5 * 10**4) + 1
                    # Make the subplot a histogram
                    n, bins, patches = ax.hist(lambda_list, range=(0, 0.6), bins=n_bins, weights=wg)
                    '''
                    wg = sum(binned_lambdas)
                    x = [ (i + 0.5) * self.lambda_max / self.n_lambda_bins 
                          for i in range(self.n_lambda_bins) ]
                    ax.bar(x, binned_lambdas/wg, width=(self.lambda_max / self.n_lambda_bins), align='center')
                    #ax.set_ylim(top=max(binned_lambdas/wg))
                    # Scientific notation on y-axis
                    #ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                    # Add labels to sub axes
                    ax.set_title(r"$\beta = " + str(beta) + "$, $R = " + 
                                 str(jetR) + "$, $N = " + str(wg) + '$')
                    ax.set_ylabel(r"$\frac{1}{N}\frac{dN}{d\lambda_\beta}$")
                    ax.set_xlabel(r"$\lambda_\beta$")
                    #if j == 0:
                    #    ax.set_ylabel(r"$\beta = " + str(beta) + '$')
                    #if k == (len(self.betas) - 1):
                    #    ax.set_xlabel(r"$R = " + str(jetR) + '$')

            # Save plot
            plt.subplots_adjust(left = 0.08, right = .97, top = 0.92, bottom = 0.08, wspace = 0.3, hspace = 0.3)
            out_dir = concat_dir(self.j_dir, pTbins_ranges[i])
            fig.savefig('%s/lambda.png' % out_dir)

        print("Lambda plots successfully saved")
        return 0

    def initialize_gen_plot(self, title):

        plt.subplots_adjust(wspace=0.6)
        plt.rc('font', size=20)          # controls default text sizes
        plt.rc('axes', labelsize=18)     # fontsize of the x and y labels
        plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=14)    # fontsize of the tick labels

        # Create figure for general plot & arrange subplots
        fig, a = plt.subplots(1, len(self.jetRadii))
        fig.suptitle(title)
        fig.text(0.5, 0.04, 'Jet Radius', ha='center', va='center')   # Major x-axis label
        fig.set_size_inches(18.5, 5.5)

        return fig, a


    def plot_gen_jet(self):
        # Create directroy to save files
        pTbins_ranges = get_pT_bin_list(self.pTbins)
        create_folder_struc(self.j_dir, pTbins_ranges)

        # Get the dataframes containing relevant general info
        N_constit, pT_dist, z_dist = self.calc_gen_jet()

        # Make general plots
        print("Plotting general jet distributions")
        fig_pT, a_pT = self.initialize_gen_plot("Jet pT distribution")
        fig_N, a_N = self.initialize_gen_plot("Jet constituent multiplicity distribution")
        fig_z, a_z = self.initialize_gen_plot("Jet z=pT,track/pT,jet distribution")

        for i, jetR in enumerate(self.jetRadii):
            ax = a_pT[0, i]
            pT_list = pT_dist[jetR]
            n, bins, patches = ax.hist(pT_list)
            # Scientific notation on y-axis
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            # Add labels to sub axes
            ax.set_title(r"$R = " + str(jetR) + '$')
            ax.set_ylabel(r"$\frac{dN}{dp_T}$")
            ax.set_xlabel(r"$p_T$")

            ax = a_N[0, i]
            N_list = N_constit[jetR]
            n, bins, patches = ax.hist(N_list)
            # Scientific notation on y-axis
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            # Add labels to sub axes
            ax.set_title(r"$R = " + str(jetR) + '$')
            ax.set_ylabel(r"$\frac{dN}{dN_{constit}}$")
            ax.set_xlabel(r"$N_{constit}$")

            ax = a_z[0, i]
            z_list = z_dist[jetR]
            n, bins, patches = ax.hist(z_list)
            # Scientific notation on y-axis
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            # Add labels to sub axes
            ax.set_title(r"$R = " + str(jetR) + '$')
            ax.set_ylabel(r"$\frac{dN}{dz}$")
            ax.set_xlabel(r"$z$")
            
        # Save plots
        fig_pT.savefig('%s/pT_dist.png' % self.j_dir)
        fig_N.savefig('%s/N_dist.png' % self.j_dir)
        fig_z.savefig('%s/z_dist.png' % self.j_dir)
        print("General jet plots successfully saved.")
        return 0


    def multi_jet(self):
        self.plot_jet_lambda()
        #self.plot_gen_jet()
        return 0
