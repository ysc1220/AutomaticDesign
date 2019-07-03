import glob
import operator
import datetime
import math
import numpy
import matplotlib.pyplot as plt
import subprocess
import argparse
import os
import random
import shutil
# #####IMPORTS ADDED BY ADITYA
# from molSimplify.Classes.mol3D import *
# from molSimplify.Classes.atom3D import *
# from molSimplify.Informatics.RACassemble import *
# from molSimplify.Classes.globalvars import *
from molSimplify.python_nn.tf_ANN import *
# ############################
from molSimplifyAD.ga_tools import *
from molSimplifyAD.ga_complex import *
from molSimplifyAD.ga_check_jobs import *
from molSimplifyAD.utils.pymongo_tools import connect2db, query_one
from molSimplifyAD.dbclass_mongo import tmcMongo


########################

class GA_generation:
    def __init__(self, name):
        path_dictionary = setup_paths()
        ligands_list = get_ligands()
        self.base_path_dictionary = path_dictionary
        self.name = name
        self.genes = dict()
        self.gene_fitness_dictionary = dict()
        self.ligands_list = ligands_list
        self.status_dictionary = dict()
        self.gene_compound_dictionary = dict()
        self.total_counter = 0

    def configure_gen(self, gen_num, npool, ncross, pmut, maxgen, scoring_function="split", property_parameter=15.0,
                      distance_parameter=1, DFT=True,
                      RTA=False, mean_fitness=0, monitor_diversity=False, monitor_distance=False,
                      **kwargs):
        self.current_path_dictionary = advance_paths(self.base_path_dictionary, gen_num)
        self.status_dictionary.update({'gen': gen_num})
        self.status_dictionary.update({'scoring_function': scoring_function})
        self.status_dictionary.update({'property_parameter': property_parameter})
        self.status_dictionary.update({'distance_parameter': distance_parameter})
        self.status_dictionary.update({'npool': npool, 'maxgen': maxgen})
        self.status_dictionary.update({'ncross': ncross})
        self.status_dictionary.update({'pmut': pmut})
        self.status_dictionary.update({'ready_to_advance': RTA})
        self.status_dictionary.update({'mean_fitness': mean_fitness})
        self.status_dictionary.update({'DFT': DFT})
        self.status_dictionary.update({'monitor_diversity': monitor_diversity})
        self.status_dictionary.update({'monitor_distance': monitor_distance})

    def populate_random(self):
        ## clear the pool
        self.gene_compound_dictionary = dict()
        self.genes = dict()
        self.total_counter = 0
        ### fill the pool with random structures
        counter = 0
        while counter < self.status_dictionary['npool']:
            this_complex = octahedral_complex(self.ligands_list)
            this_complex.random_gen()
            this_gene = this_complex.name

            print('trying to add ' + str(this_gene))
            print('list of genes is : ' + str(self.genes.values()))
            ## check if unique
            if not this_gene in self.genes.values():
                print('added successfully')
                self.genes[counter] = this_gene
                self.gene_compound_dictionary[counter] = this_complex
                counter += 1
                print('total of ' + str(counter) + ' genes')
            print('\n')
        self.total_counter = counter

    def populate_metal_ox_lig_combo(self, metal, ox, ligs):
        ### function to add a given complex to the pool
        ### arguments are positions in ligand names (1st elemet)
        ### of ligand list (not smiles)
        if isinstance(ligs[0][0], basestring):
            ligands_list_inds = [i[0] for i in self.ligands_list]
        else:
            ligands_list_inds = [i[0][0] for i in self.ligands_list]
        metal_list_inds = get_metals()

        ## check if ligs are known
        print('ligands requested:', [ligs[0][0], ligs[1][0], ligs[1][1]])
        print('indicies:', ligands_list_inds[0:7])
        print('ligs:',ligs)

        ## now test if each lig is a SMILEs or
        ## dictionary ligand
        procd_ligs = []
        proc_inds = []
        found_smi = False

        for l in ligs:
            if isinstance(l, basestring):
                print('dictionary  lig: ' + str(l))
                procd_ligs.append(l)
                print(procd_ligs)
            else:
                print('smiles lig: ' + str(l[0]))
                procd_ligs.append(l[0][0])
                found_smi = True

        if found_smi:
            print('Warning, we cannot check SMILES for ligand uniqueness for SMILEs strings')
        else:
            print('checking for ligand availability')
            print(ligands_list_inds)
            print(procd_ligs)

            if not set(procd_ligs).issubset(set(ligands_list_inds)):
                print('Error: requested ligs not available in list, aborting')
                exit()

        if not metal in metal_list_inds:
            print(metal)
            print(metal_list_inds)
            print('Error: requested metal not available in list, aborting')
            exit()

        inds = [ligands_list_inds.index(i) for i in procd_ligs]
        print('final ligand inds are ' + str(inds))
        metal_ind = metal_list_inds.index(metal)
        this_complex = octahedral_complex(self.ligands_list)
        this_complex.random_gen()
        counter = self.total_counter
        if True:#try:
            this_complex.replace_metal(metal_ind)
            this_complex.replace_ox(ox)
            this_complex.replace_ligands(inds)
            this_gene = this_complex.name
            print('this this_unique_name ', this_gene)
            # this_gene = this_complex.name
            if not this_gene in self.genes.values():
                ## we can accept this complex
                self.genes[counter] = this_gene
                self.gene_compound_dictionary[counter] = this_complex
                counter += 1
                self.total_counter = self.total_counter + 1
                print('adding eq: ' + str(ligs[0][0]) + ' and ax ' + str(ligs[1][0]) + ' + ' + str(ligs[1][1]))
            else:
                print(' this gene is a duplicate and is not added')
        else:#except:
            print('cannot make eq: ' + str(ligs[0][0]) + ' and ax ' + str(ligs[1][0]) + ' + ' + str(ligs[1][1]))
            sardines

    def write_state(self):
        ## first write genes to path
        state_path = self.current_path_dictionary["state_path"] + "current_genes.csv"
        if not os.path.isfile(state_path):
            open(state_path, 'a').close()
        else:  ## backup state data
            shutil.copyfile(state_path, self.current_path_dictionary["state_path"] + "current_genes.csv.bcp")
            ### Temperary fix for current_gene.csv
            _state_path = self.current_path_dictionary["state_path"] + "_current_genes.csv"
            with open(state_path, 'r') as fin:
                with open(_state_path, 'w') as fo:
                    if not '/gen_0/' in state_path:
                        for idx, line in enumerate(fin):
                            if idx >= self.status_dictionary['npool']:
                                ll = str(idx - self.status_dictionary['npool'])+','+line.split(',')[-1].strip('\n')+','+str(self.status_dictionary['distance_parameter'])+','+str(self.status_dictionary['scoring_function'])+'\n'
                                # print('this is the ll', ll)
                                fo.write(ll)
                    else:
                        for line in fin:
                            line = line.strip('\n') + ','+str(self.status_dictionary['distance_parameter'])+','+str(self.status_dictionary['scoring_function'])+'\n'
                            fo.write(line)
        emsg = write_dictionary(self.genes, state_path)
        ## second write live info to base directory
        state_path = self.base_path_dictionary["state_path"] + "/current_status.csv"
        if not os.path.isfile(state_path):
            open(state_path, 'a').close()
        emsg = write_dictionary(self.status_dictionary, state_path)
        if emsg:
            print(emsg)
        print('!!!!!!!!--------------------!!!!!!!!!!!!--------------------!!!!!!!!!!!!!!')
        # print(self.genes)
        ## third,  write gene-fitness info to path
        state_path = self.current_path_dictionary["state_path"] + "/gene_fitness.csv"
        if not os.path.isfile(state_path):
            open(state_path, 'a').close()
        emsg = write_dictionary(self.gene_fitness_dictionary, state_path)
        if emsg:
            print(emsg)

    def read_state(self):
        ## first read live info from base directory
        state_path = self.base_path_dictionary["state_path"] + "/current_status.csv"
        emsg, read_dict = read_dictionary(state_path)
        property_parameter = read_dict['property_parameter']
        distance_parameter = read_dict['distance_parameter']
        if '[' in property_parameter:
            import ast
            property_parameter = ast.literal_eval(property_parameter)
        else:
            property_parameter = float(property_parameter)
        if '[' in distance_parameter:
            import ast
            distance_parameter = ast.literal_eval(distance_parameter)
        else:
            distance_parameter = float(distance_parameter)
        if emsg:
            print(emsg)
        self.configure_gen(gen_num=int(read_dict["gen"]),
                           npool=int(read_dict["npool"]),
                           ncross=int(read_dict["ncross"]),
                           pmut=float(read_dict["pmut"]),
                           maxgen=int(read_dict["maxgen"]),
                           scoring_function=read_dict["scoring_function"],
                           property_parameter=property_parameter,
                           distance_parameter=distance_parameter,
                           RTA=bool((read_dict["ready_to_advance"] == 'True')),
                           mean_fitness=float(read_dict["mean_fitness"]),
                           DFT=bool((read_dict["DFT"] == 'True')),
                           monitor_diversity=bool((read_dict["monitor_diversity"] == 'True')),
                           monitor_distance=bool((read_dict["monitor_distance"] == 'True')))

        ## next read  genes from path
        state_path = self.current_path_dictionary["state_path"] + "current_genes.csv"
        emsg, gene_dict = read_dictionary(state_path)
        if emsg:
            print(emsg)
        print(gene_dict)
        for keys in gene_dict.keys():
            self.genes[int(keys)] = gene_dict[keys]
        for keys in self.genes.keys():
            genes = self.genes[keys]
            this_complex = octahedral_complex(self.ligands_list)
            this_complex.encode(genes)
            self.gene_compound_dictionary[keys] = this_complex
        self.total_counter = len(self.gene_compound_dictionary.keys())
        ## third,  read gene-fitness info to path
        state_path = self.current_path_dictionary["state_path"] + "/gene_fitness.csv"
        emsg, fit_dict = read_dictionary(state_path)
        if emsg:
            print(emsg)
        self.gene_fitness_dictionary = fit_dict

    def check_results(self):
        ## load gene fitness dict
        fitkeys = self.gene_fitness_dictionary.keys()
        ## if doing a DFT run, we need to check the filestytem for updates
        if self.status_dictionary["DFT"]:
            final_results, all_runs = check_all_current_convergence()
            for genes in final_results.keys():
                if genes in fitkeys:
                    print('gene ' + str(genes) + ' already in dict, no action')
                else:
                    this_prop = float(final_results[genes].split)
                    if self.status_dictionary['scoring_function'] == "prop+dist":
                        print('error, cannot using prop+dist fitness with ANN only. Switching to prop only.')
                        logger(self.base_path_dictionary['state_path'], str(datetime.datetime.now()) + ":  Gen " +
                               str(self.status_dictionary['gen']) +
                               ' error, cannot using prop+dist fitness with ANN only. Switching to prop only')
                    fitness = find_prop_fitness(this_prop, self.status_dictionary['property_parameter'])
                    logger(self.base_path_dictionary['state_path'], str(datetime.datetime.now()) + ":  Gen " +
                           str(self.status_dictionary['gen']) +
                           ' setting fitness to ' + "{0:.2f}".format(fitness) + ' for new genes ' + str(genes))
                    self.gene_fitness_dictionary.update({genes: fitness})

    def model_information_loader(self):
        ##### Loads model and training data
        runtype = isKeyword("runtype")
        model_list = []
        train_data_matrices = []
        runlist = []
        if type(runtype) == list:
            for run in runtype:
                runlist.append(run)
                loaded_model = load_keras_ann(run)
                model_list.append(loaded_model)
                mat = load_training_data(run)
                train_data_matrices.append(mat)
        else:
            runlist.append(runtype)
            loaded_model = load_keras_ann(runtype)
            model_list.append(loaded_model)
            mat = load_training_data(runtype)
            train_data_matrices.append(mat)
        if runtype in ['homo','gap']:
            runlist.append('split')
            loaded_model = load_keras_ann('split')
            model_list.append(loaded_model)
            mat = load_training_data('split')
            train_data_matrices.append(mat)
        return model_list, train_data_matrices, runlist

    def normalization_info_getter(self):
        ##### 
        runtype = isKeyword("runtype")
        mean_x_list = []
        mean_y_list = []
        var_x_list = []
        var_y_list = []
        if type(runtype) == list:
            for run in runtype:
                train_mean_x, train_mean_y, train_var_x, train_var_y = load_normalization_data(run)
                mean_x_list.append(train_mean_x)
                mean_y_list.append(train_mean_y)
                var_x_list.append(train_var_x)
                var_y_list.append(train_var_y)
        else:
            train_mean_x, train_mean_y, train_var_x, train_var_y = load_normalization_data(runtype)
            mean_x_list.append(train_mean_x)
            mean_y_list.append(train_mean_y)
            var_x_list.append(train_var_x)
            var_y_list.append(train_var_y)
        if runtype in ['homo','gap']:
            train_mean_x, train_mean_y, train_var_x, train_var_y = load_normalization_data('split')
            mean_x_list.append(train_mean_x)
            mean_y_list.append(train_mean_y)
            var_x_list.append(train_var_x)
            var_y_list.append(train_var_y)
        return mean_x_list, mean_y_list, var_x_list, var_y_list

    def assess_fitness(self):
        print('***********')
        #print(self.genes)
        #print(self.gene_compound_dictionary.keys())
        # print('now printing what the gene-compound dictionary knows:')
        # for keys in self.gene_compound_dictionary.keys():
        #        print('key: ' + str(keys) + ' val is  ' +  str(self.gene_compound_dictionary[keys]))
        print('***********')
        ## loop all over genes in the pool and the selected set
        fitkeys = self.gene_fitness_dictionary.keys()
        print('now printing what the gene-fitness dictionary knows:')
        for keys in fitkeys:
            print('key: ' + str(keys) + ' val is  ' + str(self.gene_fitness_dictionary[keys])+'in assess_fitness')
        fitness_values = dict()
        print('is code ready to advance?: ' + str(self.status_dictionary["ready_to_advance"]))
        logger(self.base_path_dictionary['state_path'], str(datetime.datetime.now()) + ":  Gen "
               + str(self.status_dictionary['gen'])
               + " is code ready to advance? " + str(self.status_dictionary["ready_to_advance"]))
        self.ready_to_advance = False
        self.outstanding_genes = dict()
        for genekeys in self.genes.keys():
            print('gene is ' + self.genes[genekeys])
            genes = self.genes[genekeys]
            ## see if this gene is in the fitness dictionary
            if genes in fitkeys:
                fitness_values[genes] = self.gene_fitness_dictionary[genes]
                print('genekey is ' + str(genekeys) + ' gene ' + str(
                    genes) + ' present with fitness ' + "{0:.2f}".format(float(fitness_values[genes])))
            else:
                ## add to outstanding jobs
                self.outstanding_genes.update({genekeys: self.gene_compound_dictionary[genekeys]})
                print('genekey is ' + str(genekeys) + ' gene ' + str(genes) + ' fitness  not known')
        logger(self.base_path_dictionary['state_path'], str(datetime.datetime.now()) + ":  Gen "
               + str(self.status_dictionary['gen'])
               + " with " + str(len(self.outstanding_genes.keys())) + " calculations to be completed")
        print('length of outstanding jobskeys', len(self.outstanding_genes.keys()))
        if (len(self.outstanding_genes.keys()) == 0):
            logger(self.base_path_dictionary['state_path'], str(datetime.datetime.now())
                   + ": Gen " + str(self.status_dictionary['gen'])
                   + " all jobs completed, ranking ")
            self.status_dictionary["ready_to_advance"] = True
        else:
            if not self.status_dictionary['DFT'] and isKeyword('no_geo'):
                print('Loading models and latent train data now to hand to job_dispatcher...')
                model_list, matrix_list, runlist= self.model_information_loader()
                mean_x_list, mean_y_list, var_x_list, var_y_list = self.normalization_info_getter()
                self.job_dispatcher(loaded_model_list=model_list, train_matrices=matrix_list, 
                    mean_info= [mean_x_list,mean_y_list], var_info = [var_x_list,var_y_list], run_list=runlist)
            else:
                self.job_dispatcher()
            # pass
        ## if we are using the ANN only, populate the gene-fitnes dictionary
        if self.status_dictionary["DFT"] == False:
            self.ANN_fitness()
        # sardines
        if os.path.exists('bad_initgeo_log.txt'):
            with open('bad_initgeo_log.txt', 'r') as fin:
                print('These are jobs with bad initial geometry generated from molSimplify')
                print('Please check them carefully:')
                for line in fin:
                    print(line)
        else:
            print('All initial geometry is good. Okay to go!')

    def random_fitness(self):
        ## test function for validating GA = white noise fitness
        for keys in self.genes:
            gene = self.genes[keys]
            random_fitness = random.uniform(0, 1)
            logger(self.base_path_dictionary['state_path'], str(datetime.datetime.now())
                   + ":  Gen " + str(self.status_dictionary['gen'])
                   + " assign random fitness  " + "{0:.2f}".format(random_fitness) + ' to  gene ' + str(gene))
            self.gene_fitness_dictionary.update({gene: random_fitness})

    def ANN_fitness(self):
        msg, ANN_dict = read_ANN_results_dictionary(self.current_path_dictionary["ANN_output"] + 'ANN_results.csv')
        print(ANN_dict)
        gene_template = get_gene_template()
        print('---------------- THIS IS THE GENE TEMPLATE', gene_template)
        #GA_run = get_current_GA()
        runtype = isKeyword("runtype")
        for keys in ANN_dict.keys():
            #gene, gen, slot, metal, ox, eqlig, axlig1, axlig2, eqlig_ind, axlig1_ind, axlig2_ind, spin, spin_cat, ahf, basename, basegene = translate_job_name(keys)
            translate_dict = translate_job_name(keys)
            gene = translate_dict['gene']
            gen = translate_dict['gen']
            slot = translate_dict['slot']
            metal = translate_dict['metal']
            ox = translate_dict['ox']
            liglist = translate_dict['liglist']
            indlist = translate_dict['indlist']
            spin = translate_dict['spin']
            spin_cat = translate_dict['spin_cat']
            ahf = translate_dict['ahf']
            base_name = translate_dict['basename']
            base_gene = translate_dict['basegene']
            set_fitness = False
            if type(runtype) != list:
                this_prop = float(ANN_dict[keys][runtype])
                this_dist = float(ANN_dict[keys][runtype + '_dist'])
            if runtype == 'split':
                set_fitness = True
            elif runtype == 'homo':
                this_spin = float(ANN_dict[keys]['split'])
                if (this_spin > 0 and spin < 3) or (this_spin <= 0 and spin >= 3):
                    set_fitness = True
            elif runtype == 'gap':
                this_spin = float(ANN_dict[keys]['split'])
                if (this_spin > 0 and spin < 3) or (this_spin <= 0 and spin >= 3):
                    set_fitness = True
            elif runtype in ['oxo','hat']:
                # gene = gene + '_'+ str(spin)
                metals_list = get_metals()
                # if spin_cat == 'HS' or (metal == 'cr' and int(spin) == 2): #This is temporary, only on the high spin cases...
                if gene_template['legacy']:
                    if spin_cat == isKeyword('spin_constraint'):
                        print('FITNESS SET OXO GA!')
                        print('THIS PROP',this_prop,'THIS DIST',this_dist)
                        print('SPINCAT:',spin_cat, spin)
                        print('METAL',metals_list[metal],'OX:', ox)
                        set_fitness = True
                    elif isKeyword('spin_constraint') and get_metals()[metal].lower() == 'cr' and ox == 5:
                        print('Making exception for HS Cr(V) in ANN fitness')
                        print(this_prop)
                        this_prop = 10000
                        set_fitness = True
                else:
                    set_fitness = True
            elif type(runtype) == list:
                this_prop = []
                this_dist = []
                print(ANN_dict[keys])
                if len(ANN_dict[keys])==0:
                    print('ZERO', keys)
                for run in runtype:
                    this_prop.append(float(ANN_dict[keys][str(run)]))
                    this_dist.append(float(ANN_dict[keys][str(run) + '_dist']))
                if gene_template['legacy']:
                    if spin_cat == isKeyword('spin_constraint'): #Constraining this to a single spin state.
                        print('Multiple factors in fitness')
                        set_fitness = True
                    elif (isKeyword('spin_constraint') == 'HS') and get_metals()[metal].lower() == 'cr' and ox == 5:
                        print('Making exception for HS Cr(V) in ANN fitness')
                        this_prop = [10000, 10000]
                        set_fitness = True
                else:
                    print('Nonlegacy mode with list provided for fitness!')
                    set_fitness = True
            else:
                print('-------------------RUNTYPE is invalid!--------------------')

            if set_fitness:
                print('setting fitness~~~~~~~~~~~~~~~~~~~~~~~')
                if self.status_dictionary['scoring_function'] == "prop+dist":
                    fitness = find_prop_dist_fitness(this_prop, self.status_dictionary['property_parameter'],
                                                     this_dist, self.status_dictionary['distance_parameter'])
                elif self.status_dictionary['scoring_function'] == "prop_hinge+dist":
                    # print('ENTERED THE CORRECT SCORING FUNCTION')
                    fitness = find_prop_hinge_dist_fitness(this_prop, self.status_dictionary['property_parameter'],
                                                           this_dist, self.status_dictionary['distance_parameter'], range_value = 1.0)
                    # print('gene:', gene)
                    # print('prop:', this_prop)
                    # print('dist:', this_dist)
                    print('fitness:', fitness)
                    if runtype in ['gap', 'homo','oxo20','homo_empty']:
                        print('this_spin', this_spin)
                    print('-----------')

                elif self.status_dictionary['scoring_function'] == "prop_hinge":
                    fitness = find_prop_hinge_fitness(this_prop, self.status_dictionary['property_parameter'],range_value = 1.0)
                    # print('gene:', gene)
                    # print('prop:', this_prop)
                    # print('dist:', this_dist)
                    print('fitness:', fitness)
                elif self.status_dictionary['scoring_function'] == 'nsga':
                    if type(this_prop) != list:
                        print('This is not a multiple objective optimization')
                        sardines
                    else:
                        fitness = 1 #Temporary setting everything to be fit. NSGA will sort later.
                else:
                    fitness = find_prop_fitness(this_prop, self.status_dictionary['property_parameter'])

                logger(self.base_path_dictionary['state_path'], str(datetime.datetime.now())
                       + ":  Gen " + str(self.status_dictionary['gen'])
                       + " fitness from ANN  " + "{0:.2f}".format(fitness) + ' assigned to  gene ' + str(gene))
                self.gene_fitness_dictionary.update({gene: fitness})
                # print('GFD: ',self.gene_fitness_dictionary)

    def job_dispatcher(self,loaded_model_list=False, train_matrices=False, 
                    mean_info= False, var_info = False, run_list=False):
        jobpaths = list()
        emsg, ANN_results_dict = read_ANN_results_dictionary(self.current_path_dictionary["ANN_output"] + '/ANN_results.csv')
        current_outstanding = get_outstanding_jobs()
        converged_jobs = find_converged_job_dictionary()
        gene_template = get_gene_template()
        spins_dict = spin_dictionary()
        try:
            db = connect2db(user="readonly_user", pwd="readonly", host="localhost", port=27017, database="tmc", auth=True)
            print("# of complex in db: ", db.oct.count())
            connected = True
        except:
            connected = False
        properties = ['split','split_dist','homo','homo_dist','gap','gap_dist','oxo','oxo_dist',
                                'hat','hat_dist','oxo20','oxo20_dist','homo_empty','homo_empty_dist']
        for keys in self.outstanding_genes.keys():
            job_prefix = "gen_" + str(self.status_dictionary["gen"]) + "_slot_" + str(keys) + "_"
            genes = self.outstanding_genes[keys]
            metal = genes.metals_list[genes.core]
            job_dict = []
            ## If ox in gene_template, then use the ox provided, else loop over possible ox.
            if gene_template['ox']:
                ox_list = [genes.ox]
            else:
                ox_list = get_ox_states()
            for ox in ox_list:
                ## Same for spin
                if gene_template['spin']:
                    spin_list = [genes.spin]
                else:
                    spin_list = spins_dict[metal][ox]
                flag_oct_spin = True
                for spin in spin_list:
                    ## generate HS/LS
                    ## convert the gene into a job file and geometery
                    tmcdoc = False
                    if isKeyword('no_geo') and self.status_dictionary['DFT'] == False:
                        if self.status_dictionary["DFT"] == False:
                            descriptor_names, descriptors, flag_oct, mol_name,jobpath = genes.geo_free_RAC_generator(prefix=job_prefix,
                                                                                           ox = ox,
                                                                                           spin=spin,
                                                                                           path_dictionary=self.current_path_dictionary,
                                                                                           rundirpath=isKeyword('rundir'),
                                                                                           gen=self.status_dictionary['gen'])
                            print(len(descriptor_names),len(descriptors))
                            ANN_results = {}
                            for model_num, model in enumerate(loaded_model_list):
                                excitation = tf_ANN_excitation_prepare(predictor=str(run_list[model_num]),descriptors=descriptors, descriptor_names=descriptor_names)
                                norm_excitation = data_normalize(data=excitation, train_mean=mean_info[0][model_num], train_var=var_info[0][model_num])
                                result = data_rescale(model.predict(norm_excitation), mean_info[1][model_num], var_info[1][model_num])
                                ANN_results.update({run_list[model_num]:float(result)})
                                ###### NEED TO ADJUST LATER
                                min_dist = 100000000
                                min_ind = 0
                                best_scaled_row = 0
                                for i, rows in enumerate(np.array(train_matrices[model_num],dtype='float64')):
                                    scaled_row = np.squeeze(data_normalize(rows, mean_info[0][model_num].T, var_info[0][model_num].T))  # Normalizing the row before finding the distance
                                    # print(scaled_row)
                                    this_dist = np.linalg.norm(np.subtract(scaled_row, np.array(norm_excitation)))
                                    if this_dist < min_dist:
                                        min_dist = this_dist
                                        min_ind = i
                                        # best_row = rownames[i]
                                        best_scaled_row = scaled_row
                                        min_row = rows
                                ANN_results.update({run_list[model_num]+'_dist':float(min_dist)})
                            if len(list(set(properties).difference(ANN_results.keys())))>0:
                                for i in properties:
                                    if i not in ANN_results.keys():
                                        ANN_results.update({i:float(10000)}) #Chosen to be arbitrarily large to reduce the fitness value to 0.
                                        print(str(i)+ ' set to 10000 in ANN_results, chosen so that the fitness goes to 0. The key was not present.')
                                    else:
                                        print(str(i)+ ' set to '+str(ANN_results[i])+' since the key was present')
                    else:
                        if gene_template['legacy']:
                            jobpath, mol_name, ANN_results, flag_oct = genes.generate_geometry_legacy(prefix=job_prefix,
                                                                                               spin=spin,
                                                                                               path_dictionary=self.current_path_dictionary,
                                                                                               rundirpath=isKeyword('rundir'),
                                                                                               gen=self.status_dictionary['gen'])
                        else:
                            if connected:
                                constraints = genes.assemble_constraints(ox=ox, spin=spin)
                                print("query constraints: ", constraints)
                                tmcdoc = query_one(db, collection="oct", constraints=constraints)
                                if not tmcdoc ==  None:
                                    if not tmcdoc["converged"]:
                                        tmcdoc = False
                                        print("Found in the database but whose geometry optimization did not converge.", constraints)
                                    else:
                                        print("using results from database with constraints: ", constraints)
                                else:
                                    tmcdoc = False
                                    print("Not found in the database. ")
                            jobpath, mol_name, ANN_results, flag_oct = genes.generate_geometry(prefix=job_prefix,
                                                                                               ox = ox,
                                                                                               spin=spin,
                                                                                               path_dictionary=self.current_path_dictionary,
                                                                                               rundirpath=isKeyword('rundir'),
                                                                                               gen=self.status_dictionary['gen'],
                                                                                               tmcdoc=tmcdoc)
                        
                    if flag_oct:
                        if (jobpath not in current_outstanding) and (jobpath not in converged_jobs.keys()):
                            msg, ANN_dict = read_ANN_results_dictionary(self.current_path_dictionary["ANN_output"] + 'ANN_results.csv')
                            print('saving result in ANN dict: ' + mol_name)
                            ANN_results_dict.update({mol_name: ANN_results})
                            if not tmcdoc:
                                jobpaths.append(jobpath)
                                logger(self.base_path_dictionary['state_path'], str(datetime.datetime.now()) + ":  Gen "
                                       + str(self.status_dictionary['gen'])
                                       + " missing information for gene number  " + str(keys) + ' with  name ' + str(genes.name))
                            else:
                                print("in DB.")
                                logger(self.base_path_dictionary['state_path'], str(datetime.datetime.now()) + ":  Gen "
                                       + str(self.status_dictionary['gen'])
                                       + " completed geometry optimization for gene number " + str(keys) +
                                       ' with  name ' + str(genes.name) + ' Extracted from database with a complex named uniquely as ' +
                                       tmcdoc["unique_name"])
                                update_converged_job_dictionary(jobpath, 2)
                                log_indb_pairs(jobpath, tmcdoc)
                    else:
                        logger(self.base_path_dictionary['state_path'], str(datetime.datetime.now()) + ":  Gen "
                                   + str(self.status_dictionary['gen'])
                                   + " flag_oct false for  " + str(keys) + ' with  name ' + str(genes.name))
                        if (jobpath not in current_outstanding) and (jobpath not in converged_jobs.keys()):
                            msg, ANN_dict = read_ANN_results_dictionary(
                                self.current_path_dictionary["ANN_output"] + 'ANN_results.csv')
                            if not mol_name in ANN_dict.keys():
                                print('saving result in ANN dict: ' + mol_name)
                                ANN_results_dict.update({mol_name: ANN_results})
                        log_bad_initial(jobpath)
                        update_converged_job_dictionary(jobpath, 3)
            write_ANN_results_dictionary(self.current_path_dictionary["ANN_output"] + 'ANN_results.csv',ANN_results_dict)
        if isKeyword('no_geo'):
            from keras import backend as K
            K.clear_session()
        set_outstanding_jobs(current_outstanding + jobpaths)

    # Tree doctor will do checkup on tree's diversity and distance. Functionality can be switched on or off. Automatically off if DFT enabled.
    def get_full_values(self, curr_gen):
        full_gene_info = dict()
        #GA_run = get_current_GA()
        gene_template = get_gene_template()
        runtype = isKeyword("runtype")
        for gen in xrange(curr_gen + 1):
            ANN_dir = isKeyword('rundir') + "ANN_ouput/gen_" + str(gen) + "/ANN_results.csv"
            emsg, ANN_dict = read_ANN_results_dictionary(ANN_dir)
            for keys in ANN_dict.keys():
                #_, _, _, metal, ox, eqlig, axlig1, axlig2, _, _, _, spin, spin_cat, ahf, _, _ = translate_job_name(keys)
                translate_dict = translate_job_name(keys)
                metal = translate_dict['metal']
                ox = translate_dict['ox']
                spin = translate_dict['spin']
                spin_cat = translate_dict['spin_cat']
                # this_gene = "_".join(keys.split("_")[4:10]) #### now switch to gene from translate job?
                this_gene = translate_dict['gene']
                set_fitness = False
                # if runtype in ['oxo', 'hat']:
                #     this_gene = this_gene + '_'+str(spin)
                print('using ' + str(runtype) + ': ' + "_".join(keys.split("_")))
                if type(runtype) != list:
                    this_prop = float(ANN_dict[keys][runtype])
                    this_dist = float(ANN_dict[keys][runtype + '_dist'])
                if runtype == 'split':
                    set_fitness = True
                elif runtype == 'homo':
                    this_spin = float(ANN_dict[keys]['split'])
                    if (this_spin > 0 and spin < 3) or (this_spin <= 0 and spin >= 3):
                        set_fitness = True
                elif runtype == 'gap':
                    this_spin = float(ANN_dict[keys]['split'])
                    if (this_spin > 0 and spin < 3) or (this_spin <= 0 and spin >= 3):
                        set_fitness = True
                elif runtype in ['oxo','hat']:
                # gene = gene + '_'+ str(spin)
                    metals_list = get_metals()
                    # if spin_cat == 'HS' or (metal == 'cr' and int(spin) == 2): #This is temporary, only on the high spin cases...
                    if gene_template['legacy']:
                        if spin_cat == isKeyword('spin_constraint'):
                            print('FITNESS SET OXO GA!')
                            print('THIS PROP',this_prop,'THIS DIST',this_dist)
                            print('SPINCAT:',spin_cat, spin)
                            print('METAL',metals_list[metal],'OX:', ox)
                            set_fitness = True
                        elif isKeyword('spin_constraint') and get_metals()[metal].lower() == 'cr' and ox == 5:
                            print('Making exception for HS Cr(V) in get full values')
                            print(this_prop)
                            set_fitness = True
                    else:
                        set_fitness = True
                elif type(runtype) == list:
                    this_prop = []
                    this_dist = []
                    for run in runtype:
                        this_prop.append(float(ANN_dict[keys][run]))
                        this_dist.append(float(ANN_dict[keys][run + '_dist']))
                    if gene_template['legacy']:
                        if spin_cat == isKeyword('spin_constraint'): #Constraining this to a single spin state.
                            print('Multiple factors in fitness')
                            set_fitness = True
                        elif isKeyword('spin_constraint') and get_metals()[metal].lower() == 'cr' and ox == 5:
                            print('Making Cr(V) exception in full_gene_info')
                            set_fitness = True
                    else:
                        print('Multiple factors in fitness, nonlegacy mode')
                        set_fitness = True
                else:
                    print('-------------------RUNTYPE is invalid!--------------------')
                if set_fitness and not (this_gene in full_gene_info.keys()):
                    full_gene_info.update({this_gene: [this_prop, this_dist]})
        return full_gene_info

    def calc_mean_dist(self, genes_list, full_gene_info):
        mean_dist = 0
        npool = int(self.status_dictionary['npool'])
        for i in range(0, npool):
            print(full_gene_info[genes_list[i]][1])
            if type(full_gene_info[genes_list[i]][1]) == list:
                if int(full_gene_info[genes_list[i]][1][0])==10000 or int(full_gene_info[genes_list[i]][1][1])==10000:
                    npool -= 1
                    continue
                else:
                    mean_dist += float(np.mean([full_gene_info[genes_list[i]][1][0], full_gene_info[genes_list[i]][1][1]]))
            else:
                if int(full_gene_info[genes_list[i]][1])==10000: #Do not count these in mean distance because they arent assigned.
                    npool -= 1
                    continue
                else:
                    mean_dist += float(full_gene_info[genes_list[i]][1])
        if npool == 0:
            npool += 1
        mean_dist = mean_dist / npool  # average distance
        return mean_dist

    def update_gene_fitness(self, full_gene_info):
        # use self.gene_fitness_dictionary
        ## update gene-fitness
        for gene in self.gene_fitness_dictionary.keys():
            if type(full_gene_info[gene][0]) == list:
                this_prop = full_gene_info[gene][0]
                this_dist = full_gene_info[gene][1]
            else:
                this_prop = float(full_gene_info[gene][0])
                this_dist = float(full_gene_info[gene][1])
            if self.status_dictionary['scoring_function'] == "prop+dist":
                print('here')
                print(self.status_dictionary)
                fitness = find_prop_dist_fitness(this_prop, self.status_dictionary['property_parameter'],
                                                 this_dist, self.status_dictionary['distance_parameter'])
            elif self.status_dictionary['scoring_function'] == "prop_hinge+dist":
                fitness = find_prop_hinge_dist_fitness(this_prop, self.status_dictionary['property_parameter'],
                                                       this_dist, self.status_dictionary['distance_parameter'],range_value = 2.5)
            elif self.status_dictionary['scoring_function'] == "prop_hinge":
                fitness = find_prop_hinge_fitness(this_prop, self.status_dictionary['property_parameter'],range_value = 2.5)
            elif self.status_dictionary['scoring_function'].lower() == 'nsga':
                if type(this_prop) != list:
                    print('This is not a multiple objective optimization')
                    sardines
                else:
                    print('NSGA function!')
                    fitness = 1 #Temporary setting
            else:
                fitness = find_prop_fitness(this_prop, self.status_dictionary['property_parameter'])
            self.gene_fitness_dictionary.update({gene: fitness})

    def get_diversity(self):
        genes = list()
        gene_dict = self.genes

        for key in gene_dict.keys():
            this_gene = gene_dict[key]
            if not (this_gene in genes):
                genes.append(this_gene)

        diversity = len(genes)
        return diversity

    def decide(self):
        curr_gen = self.status_dictionary["gen"]
        diagnosis = ['***************************************************************']
        diagnosis.append("GA doctor checked on end gen " + str(curr_gen))

        healthy = True
        ## read in scoring function info
        dist_score = ("dist" in self.status_dictionary['scoring_function'])
        print(self.status_dictionary['distance_parameter'])
        if type(self.status_dictionary['distance_parameter']) == list:
            dist_param = self.status_dictionary['distance_parameter']
        else:
            dist_param = float(self.status_dictionary['distance_parameter'])

        pmut = float(self.status_dictionary['pmut'])

        ## print mean_distance, mean_fitness, and diversity
        full_gene_info = self.get_full_values(curr_gen)
        print('-----------FULL GENE INFO IN DECIDE',full_gene_info)

        mean_dist = float(self.calc_mean_dist(self.genes, full_gene_info))
        diversity = self.get_diversity()

        mean_dist_info = ": ".join(["Mean distance", "{0:.2f}".format((mean_dist))])
        mean_fit_info = ": ".join(["Mean fitness", "{0:.2f}".format(self.status_dictionary['mean_fitness'])])
        div_info = ": ".join(["Diversity", str(diversity)])
        diagnosis.extend([mean_dist_info, mean_fit_info, div_info])

        ## check and adjust pmut based on diversity
        ### if diversity drops below 25% of npool, pmut will be raised to 0.50
        ### if diveristy is at least 25% of npool, pmut will return to original
        if (self.status_dictionary['monitor_diversity']):
            print('ENTERED INTO DIVERSITY')
            current_pmut = self.status_dictionary['pmut']
            low_diversity = (diversity < 0.25 * (int(self.status_dictionary['npool'])))
            if (low_diversity) and (current_pmut < 0.50) and (current_pmut >= 0):
                healthy = False
                symptom0 = "WARNING: Diversity low (less than 25% of npool): inflating pmut (" + str(
                    current_pmut) + ") to 0.50."
                symptom01 = "WARNING Pmut, in current_status.csv, is (original_pmut - 0.50). It will be shown as negative."
                diagnosis.extend([symptom0, symptom01])
                new_pmut = float(current_pmut - 0.50)
                self.status_dictionary.update({'pmut': new_pmut})
            elif (low_diversity):  # high pmut
                healthy = False
                symptom00 = "Diversity low. Pmut already high. No medicine available. Check after a few more generations."
                diagnosis.append(symptom00)
            elif (current_pmut < 0):  # check if pmut has been inflated - if it has and diversity is ok, restore to original
                orig_pmut = float(current_pmut + 0.50)
                diagnosis.append("Low diversity has been cured. Pmut now restored to " + str(orig_pmut))
                self.status_dictionary.update({'pmut': orig_pmut})
            elif self.status_dictionary['mean_fitness'] == 0:
                healthy = False
                symptom02 = "Fitness bad. Pmut increased. Check after a few more generations."
                new_pmut = float(current_pmut - 0.50)
                diagnosis.append(symptom02)

        ## adjust scoring_function based on calculated mean_distance
        if (self.status_dictionary['monitor_distance']):
            print('ENTERED INTO DISTANCE')
            if (mean_dist > 0.75):
                healthy = False
                if type(dist_param) == list:
                    if dist_score and (max(dist_param) > 0.75):  # Decrease distance_parameter for tighter control
                        dist_arg = np.argmax(dist_param)
                        dist_param[dist_arg] = dist_param[dist_arg] - 0.05
                        symptom1 = ("Mean distance too high. Lowering distance_parameter to " + str(dist_param))
                        diagnosis.append(symptom1)
                    elif dist_score and (max(dist_param) <= 0.25):  # distance_parameter too low
                        symptom2 = ("Distance parameter low, but mean distance high. Try a few more generations.")
                        diagnosis.append(symptom2)
                    else:  # Turn on split+dist
                        dist_score = True
                        dist_arg = np.argmax(dist_param)
                        dist_param[dist_arg] = dist_param[dist_arg] - 0.05
                        symptom3 = ("Mean distance high. Using prop_hinge+dist with dist_param = " + str(dist_param))
                        diagnosis.append(symptom3)
                else:
                    if dist_score and (dist_param > 0.75):  # Decrease distance_parameter for tighter control
                        dist_param = dist_param - 0.05
                        symptom1 = ("Mean distance too high. Lowering distance_parameter to " + str(dist_param))
                        diagnosis.append(symptom1)
                    elif dist_score and dist_param <= 0.25:  # distance_parameter too low
                        symptom2 = ("Distance parameter low, but mean distance high. Try a few more generations.")
                        diagnosis.append(symptom2)
                    else:  # Turn on split+dist
                        dist_score = True
                        dist_param = dist_param - 0.05
                        symptom3 = ("Mean distance high. Using prop_hinge+dist with dist_param = " + str(dist_param))
                        diagnosis.append(symptom3)
            elif dist_score:  # loosen distance control
                dist_score = False
                treat1 = "Mean distance below 0.75: loosening distance control by using prop only."
                diagnosis.append(treat1)

            ## update scoring function in status_dictionary
            if 'nsga' in self.status_dictionary['scoring_function'].lower():
                self.status_dictionary.update({'scoring_function': "nsga"})
            elif dist_score and 'hinge' in self.status_dictionary['scoring_function']:
                self.status_dictionary.update({'scoring_function': "prop_hinge+dist"})
            elif dist_score and 'hinge' not in self.status_dictionary['scoring_function']:
                self.status_dictionary.update({'scoring_function': "prop+dist"})
            elif 'hinge' in self.status_dictionary['scoring_function']:
                self.status_dictionary.update({'scoring_function': "prop_hinge"})
            else:
                self.status_dictionary.update({'scoring_function': "prop"})
            self.status_dictionary.update({'distance_parameter': dist_param})
            diagnosis.append("Update~ Scoring_function: " + self.status_dictionary['scoring_function'] + ", Dist_param: " + str(dist_param))

            ## update gene_fitness
            diagnosis.append("Updating gene_fitness...")
            self.update_gene_fitness(full_gene_info)
            diagnosis.append("Checkup finished.")

        if healthy:
            diagnosis.append("Tree healthy. Carry on.")

        diagnosis.append('****************************************************************')
        for message in diagnosis:
            print(message)
            logger(self.base_path_dictionary['state_path'], str(datetime.datetime.now()) + '   ' + str(message))

    ################################################################

    def select_best_genes(self):
        ## first write genes to path
        summary_path = self.current_path_dictionary["state_path"] + "all_genes.csv"
        outcome_list = list()
        npool = self.status_dictionary["npool"]
        mean_fitness = 0
        
        for keys in self.genes.keys():
            outcome_list.append((keys, self.genes[keys], float(self.gene_fitness_dictionary[self.genes[keys]]), self.status_dictionary['scoring_function'], self.status_dictionary['distance_parameter']))
            logger(self.base_path_dictionary['state_path'], str(datetime.datetime.now()) +
                   ":  Gen " + str(self.status_dictionary['gen']) + '  gene is ' + str(keys)
                   + " fitness is = " + "{0:.2f}".format(float(self.gene_fitness_dictionary[self.genes[keys]]))
                   + " fitness function is = "+ self.status_dictionary['scoring_function'])

        outcome_list.sort(key=lambda tup: tup[2], reverse=True)
        full_size = len(outcome_list)

        if not os.path.isfile(summary_path):
            open(summary_path, 'a').close()
        emsg = write_summary_list(outcome_list, summary_path)
        self.genes = dict()
        self.gene_compound_dictionary = dict()
        for i in range(0, npool):
            self.genes[i] = outcome_list[i][1]
            this_complex = octahedral_complex(self.ligands_list)
            this_complex.encode(self.genes[i])
            self.gene_compound_dictionary[i] = this_complex
            mean_fitness += float(outcome_list[i][2])
        mean_fitness = mean_fitness / npool  # average fitness
        self.status_dictionary.update({'mean_fitness': mean_fitness})
        logger(self.base_path_dictionary['state_path'], str(datetime.datetime.now()) +
               ":  Gen " + str(self.status_dictionary['gen'])
               + " complete, mean_fitness = " + "{0:.2f}".format(mean_fitness))

        if not (self.status_dictionary['DFT']) and (
                self.status_dictionary['monitor_diversity'] or self.status_dictionary['monitor_distance']):
            self.decide()
        if self.status_dictionary['DFT'] and (self.status_dictionary['monitor_distance']):
            self.decide()

    def advance_generation(self):
        ## The old ANN dict from the previous generation must be loaded to gather the results
        ## advance counter
        self.status_dictionary['gen'] += 1
        msg, ANN_dict = read_ANN_results_dictionary(self.base_path_dictionary["ANN_output"] + 'full_ANN_results.csv')
        logger(self.base_path_dictionary['state_path'], str(datetime.datetime.now()) +
               ":  Gen " + str(self.status_dictionary['gen'] - 1)
               + " advancing to Gen " + str(self.status_dictionary['gen']))
        self.status_dictionary['ready_to_advance'] = False
        self.current_path_dictionary = advance_paths(self.base_path_dictionary, self.status_dictionary['gen'])
        print('selected_compound_dictionary is ' + str(self.gene_compound_dictionary))
        npool = self.status_dictionary["npool"]
        ncross = self.status_dictionary["ncross"]
        pmut = self.status_dictionary["pmut"]
        if (pmut < 0):
            original_pmut = float(pmut + 0.50)
            pmut = 0.50

        ## generation of selected set
        selected_genes = dict()
        selected_compound_dictionary = dict()
        number_selected = 0
        # counter = 0
        ## populate selected pool
        whilecounter = 0
        guarantee_mutation = []
        if self.status_dictionary['scoring_function'].lower() == 'nsga':
            print('NSGA scoring.')
            runtype = isKeyword("runtype")
            if type(runtype) != list:
                print('Not a multiobjective runtype.')
                sardines
            print(ANN_dict)
            objective1_values, objective2_values = [], []
            objective1_dists, objective2_dists = [], []
            keylist = []
            for keys in ANN_dict.keys():
                print('this is the key',keys)
                keylist.append(keys)
                objective1_values.append(-float(ANN_dict[keys][runtype[0]]))
                objective1_dists.append(float(ANN_dict[keys][runtype[0] + '_dist']))
                objective2_values.append(-float(ANN_dict[keys][runtype[1]]))
                objective2_dists.append(float(ANN_dict[keys][runtype[1] + '_dist']))
            genelist = [translate_job_name(key)['gene'] for key in keylist]
            non_dominated_sorted_solution = fast_non_dominated_sort(objective1_values,objective2_values)
            print('nds solution', non_dominated_sorted_solution)
            for i, val in enumerate(non_dominated_sorted_solution):
                print(i, 'frontier')
                for val_num in val:
                    print('values',objective1_values[val_num],objective2_values[val_num])
            # print(objective1_values, objective2_values)
            # plt.scatter(objective1_values,objective2_values)
            # plt.show()
            crowding_distance_values=[]
            for i in range(0,len(non_dominated_sorted_solution)):
                crowding_distance_values.append(crowding_distance(objective1_values,objective2_values,non_dominated_sorted_solution[i][:]))
            new_solution= []
            for i in range(0,len(non_dominated_sorted_solution)):
                full_nondominated_sort_solution = [non_dominated_sorted_solution[i].index(non_dominated_sorted_solution[i][j]) for j in range(0,len(non_dominated_sorted_solution[i]))]
                front22 = sort_by_values(full_nondominated_sort_solution[:], crowding_distance_values[i][:])
                front = [non_dominated_sorted_solution[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution[i]))]
                front.reverse()
                print('this is the front', front)
                for value in front:
                    new_solution.append(value)
                    if(len(new_solution)==npool):
                        break
                #### New_solution is a list of indices, reflecting the complexes that are selected.
                if (len(new_solution) == npool):
                    break
            print(len(keylist))
            print(self.genes)
            print('---')
            for pareto_solution in new_solution:
                this_gene = genelist[pareto_solution]
                print('this is the gene', this_gene)
                print('On pareto front. Selected.')
                selected_genes[number_selected + npool] = this_gene
                number_selected += 1
                guarantee_mutation.append(0)
            print('npool',npool)
            print('numselect',number_selected)
            while number_selected < npool:
                print('here adding some genes')
                this_int = random.randint(0, len(genelist) - 1)
                this_gene = genelist[this_int]
                selected_genes[number_selected + npool] = this_gene
                number_selected += 1
                guarantee_mutation.append(1)
                print('now numselect',number_selected)

            ## populate compound list
            for keys in selected_genes.keys():
                genes = selected_genes[keys]
                this_complex = octahedral_complex(self.ligands_list)
                this_complex.encode(genes)
                selected_compound_dictionary[keys] = this_complex
            print('selected compound dict' ,len(selected_compound_dictionary.keys()))
            ## now perfrom ncross exchanges
            number_of_crosses = 0
            while number_of_crosses < ncross:
                ## choose partners to exchange
                print('*************************')
                print('crossover ' + str(number_of_crosses + 1) + ' :')
                these_partners = random.sample(range(npool, (2 * npool - 1)), 2)
                keep_axial = selected_compound_dictionary[these_partners[0]]
                keep_equitorial = selected_compound_dictionary[these_partners[1]]
                old_genes = [selected_genes[key] for key in these_partners]
                new_complex_1 = keep_axial.exchange_ligands(keep_equitorial, True)
                # new_complex_1 = new_complex_1.exchange_metal(keep_equitorial)
                # new_complex_1 = new_complex_1.exchange_ox(keep_equitorial)
                print('FINAL : 1st new gene from this cross ' + str(new_complex_1.name) + '\n')

                new_complex_2 = keep_equitorial.exchange_ligands(keep_axial, True)
                # new_complex_2 = new_complex_2.exchange_metal(keep_axial)
                # new_complex_2 = new_complex_2.exchange_ox(keep_axial)
                new_gene_1 = new_complex_1.name
                new_gene_2 = new_complex_2.name
                print('FINAL : 2nd new gene from this cross ' + str(new_complex_2.name) + '\n')
                selected_genes[these_partners[0]] = new_gene_1
                selected_compound_dictionary[these_partners[0]] = new_complex_1
                selected_genes[these_partners[1]] = new_gene_2
                selected_compound_dictionary[these_partners[1]] = new_complex_2
                new_genes = [selected_genes[key] for key in these_partners]

                number_of_crosses += 1
                logger(self.base_path_dictionary['state_path'], str(datetime.datetime.now()) +
                       ":  Gen " + str(self.status_dictionary['gen'])
                       + " crossing " + str(these_partners) + " " +
                       str(old_genes) + " -> " + str(new_genes))
                print('\n')

            ## mutate
            for key_i, keys in enumerate(selected_genes.keys()):
                does_mutate = random.uniform(0, 1)
                if guarantee_mutation[key_i]:
                    does_mutate = 0
                if does_mutate < pmut:
                    old_gene = selected_genes[keys]
                    mutant = selected_compound_dictionary[keys].mutate()
                    selected_compound_dictionary[keys] = mutant
                    selected_genes[keys] = selected_compound_dictionary[keys].name
                    logger(self.base_path_dictionary['state_path'], str(datetime.datetime.now()) +
                           ":  Gen " + str(self.status_dictionary['gen'])
                           + " mutating " + str(keys) + ": " + old_gene + " -> " + mutant.name)
        else:
            while number_selected < npool:
                # counter += 1
                whilecounter += 1
                if whilecounter % 10000000 == 0:
                    print('Been in while loop for '+str(whilecounter))
                if whilecounter == 100000000:
                    print('Hit the 100,000,000 mark on the while loop for selection')
                    print('Guaranteeing mutation later.')
                    while number_selected < npool:
                        selected_genes[number_selected + npool] = this_gene
                        number_selected += 1
                        guarantee_mutation.append(1)
                    break
                this_int = random.randint(0, npool - 1)
                this_barrier = random.uniform(0, 1)
                # print('THIS INT: ',this_int, 'THIS BARRIER: ',this_barrier)
                this_gene = self.genes[this_int]
                # print('GFD value: ',self.gene_fitness_dictionary[this_gene])
                if self.gene_fitness_dictionary[this_gene] > this_barrier:
                    print('GFD>This barrier. Selected.')
                    selected_genes[number_selected + npool] = this_gene
                    number_selected += 1
                    guarantee_mutation.append(0)
            ## populate compound list
            for keys in selected_genes.keys():
                genes = selected_genes[keys]
                this_complex = octahedral_complex(self.ligands_list)
                this_complex.encode(genes)
                selected_compound_dictionary[keys] = this_complex
            ## now perfrom ncross exchanges
            number_of_crosses = 0
            while number_of_crosses < ncross:
                ## choose partners to exchange
                print('*************************')
                print('crossover ' + str(number_of_crosses + 1) + ' :')
                these_partners = random.sample(range(npool, (2 * npool - 1)), 2)
                keep_axial = selected_compound_dictionary[these_partners[0]]
                keep_equitorial = selected_compound_dictionary[these_partners[1]]
                old_genes = [selected_genes[key] for key in these_partners]
                new_complex_1 = keep_axial.exchange_ligands(keep_equitorial, True)
                # new_complex_1 = new_complex_1.exchange_metal(keep_equitorial)
                # new_complex_1 = new_complex_1.exchange_ox(keep_equitorial)
                print('FINAL : 1st new gene from this cross ' + str(new_complex_1.name) + '\n')

                new_complex_2 = keep_equitorial.exchange_ligands(keep_axial, True)
                # new_complex_2 = new_complex_2.exchange_metal(keep_axial)
                # new_complex_2 = new_complex_2.exchange_ox(keep_axial)
                new_gene_1 = new_complex_1.name
                new_gene_2 = new_complex_2.name
                print('FINAL : 2nd new gene from this cross ' + str(new_complex_2.name) + '\n')
                selected_genes[these_partners[0]] = new_gene_1
                selected_compound_dictionary[these_partners[0]] = new_complex_1
                selected_genes[these_partners[1]] = new_gene_2
                selected_compound_dictionary[these_partners[1]] = new_complex_2
                new_genes = [selected_genes[key] for key in these_partners]

                number_of_crosses += 1
                logger(self.base_path_dictionary['state_path'], str(datetime.datetime.now()) +
                       ":  Gen " + str(self.status_dictionary['gen'])
                       + " crossing " + str(these_partners) + " " +
                       str(old_genes) + " -> " + str(new_genes))
                print('\n')

            ## mutate
            for key_i, keys in enumerate(selected_genes.keys()):
                does_mutate = random.uniform(0, 1)
                if guarantee_mutation[key_i]:
                    does_mutate = 0
                if does_mutate < pmut:
                    old_gene = selected_genes[keys]
                    mutant = selected_compound_dictionary[keys].mutate()
                    selected_compound_dictionary[keys] = mutant
                    selected_genes[keys] = selected_compound_dictionary[keys].name
                    logger(self.base_path_dictionary['state_path'], str(datetime.datetime.now()) +
                           ":  Gen " + str(self.status_dictionary['gen'])
                           + " mutating " + str(keys) + ": " + old_gene + " -> " + mutant.name)

        ## merge the lists
        self.genes.update(selected_genes)
        self.gene_compound_dictionary.update(selected_compound_dictionary)

    def write_elec_prop_infile(self, filepath):
        with open(filepath, 'r') as f:
            ftxt = f.readlines()
        with open(filepath, 'w') as f:
            if not ftxt == None:
                f.writelines(ftxt[:-1])
            f.write('### props ####\n')
            f.write('ml_prop yes\n')
            f.write('poptype mulliken\n')
            f.write('bond_order_list yes\n')
            f.write('end\n')


########################
def update_current_gf_dictionary(gene, fitness):
    ## set up environment:
    path_dictionary = setup_paths()
    new_tree = GA_generation('temp tree')
    ## read in info
    new_tree.read_state()
    new_tree.gene_fitness_dictionary.update({gene: fitness})
    logger(path_dictionary['state_path'], str(datetime.datetime.now())
           + "  Gen " + str(new_tree.status_dictionary['gen']) + " :  updating gene-fitness dictionary")
    ## save
    new_tree.write_state()
