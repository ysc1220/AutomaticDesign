import os
import shutil
import numpy
import pickle
import pandas as pd
from molSimplifyAD.ga_io_control import *


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        print('creating' + dir_path)
        os.makedirs(dir_path)


########################
def get_run_dir():
    GA_run = get_current_GA()
    rdir = GA_run.config['rundir']
    return rdir


########################
def get_current_GA():
    GA_run = GA_run_defintion()
    GA_run.deserialize('.madconfig')
    return GA_run


########################
def get_infile_from_job(job):
    ## given a job (file under jobs/gen_x/
    ## this returns the appropraite infile/gen_x file
    ## if there is no infile, this will make one
    ## using progress geometry if available
    ## else intital only
    ## process job name
    _, gen, _, _, _, _, _, _, _, _, _, _, _, _, base_name, _ = translate_job_name(job)
    ## create paths
    path_dictionary = setup_paths()
    path_dictionary = advance_paths(path_dictionary, gen)
    target_inpath = path_dictionary["infiles"] + base_name + '.in'
    path_dictionary = setup_paths()
    ll = os.path.split(job)
    base_name = ll[1]
    ll = os.path.split(ll[0])
    generation_folder = ll[1]
    target_inpath = path_dictionary["infiles"] + generation_folder + '/' + base_name
    if os.path.isfile(target_inpath):
        infile = target_inpath
    else:
        print('no infile found for job ' + job + ' , creating a new one ')
        create_generic_infile(job, restart=True)
    if 'track_elec_prop' in get_current_GA().config.keys():
        track_elec_prop = get_current_GA().config['track_elec_prop']
    else:
        track_elec_prop = False
    if track_elec_prop:
        add_ml_prop_infiles(target_inpath)
    return target_inpath


#########################
def add_ml_prop_infiles(filepath):
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
def get_initial_geo_path_from_job(job):
    ## given a job (file under jobs/gen_x/
    ## this returns the path to the initial geo file
    _, gen, _, _, _, _, _, _, _, _, _, _, _, _, base_name, _ = translate_job_name(job)
    ## create paths
    path_dictionary = setup_paths()
    path_dictionary = advance_paths(path_dictionary, gen)
    target_initial_geo_path = path_dictionary["initial_geo_path"] + base_name + '.xyz'
    return target_initial_geo_path


########################
def create_generic_infile(job, restart=False, use_old_optimizer=False, custom_geo_guess=False):
    ## custom_geo_guess is ANOTHER JOB NAME, from which the geom and wavefunction guess
    ## will attempt to be extracted
    ## process job name
    _, gen, _, _, _, _, _, _, _, _, _, this_spin, _, _, base_name, _ = translate_job_name(job)
    ## create paths
    path_dictionary = setup_paths()
    path_dictionary = advance_paths(path_dictionary, gen)
    target_inpath = path_dictionary["infiles"] + base_name + '.in'
    initial_geo_path = path_dictionary["initial_geo_path"] + base_name + '.xyz'
    prog_geo_path = path_dictionary["prog_geo_path"] + base_name + '.xyz'
    guess_path = path_dictionary["scr_path"] + base_name + '/'

    ## set up guess:
    if restart:
        if os.path.isfile(prog_geo_path):
            geometry_path = prog_geo_path
            guess_string = "guess " + guess_path + 'ca0' + ' ' + guess_path + 'cb0\n'
        else:
            geometry_path = initial_geo_path
            guess_string = "guess generate \n"
    elif custom_geo_guess:
        _, guess_gen, _, _, _, _, _, _, _, _, _, _, _, _, guess_base_name, _ = translate_job_name(custom_geo_guess)
        guess_path_dictionary = setup_paths()
        guess_path_dictionary = advance_paths(guess_path_dictionary, guess_gen)
        guess_geo_path = path_dictionary["optimial_geo_path"] + guess_base_name + '.xyz'
        guess_path = path_dictionary["scr_path"] + guess_base_name + '/'
        if os.path.isfile(guess_geo_path):
            geometry_path = guess_geo_path
            guess_string = "guess generate \n"
        else:
            geometry_path = initial_geo_path
            guess_string = "guess generate \n"
    else:
        guess_string = "guess generate \n"
        geometry_path = initial_geo_path
        ## copy file to infiles
    shutil.copy(job, target_inpath)
    with open(job, 'r') as sourcef:
        source_lines = sourcef.readlines()
        with open(target_inpath, 'w') as newf:
            for line in source_lines:
                if not ("coordinates" in line) and (not "end" in line) and (not "new_minimizer" in line):
                    if ("method ub3lyp" in line) and this_spin == 1:
                        newf.write('method b3lyp\n')
                    else:
                        newf.write(line)

    ## append geo
    with open(target_inpath, 'a') as newf:
        newf.write('coordinates ' + geometry_path + '\n')
        if use_old_optimizer:
            newf.write('min_coordinates cartesian\n')
        else:
            newf.write("new_minimizer yes\n")
        newf.write(guess_string)
        newf.write('end\n')


########################
def output_properties(comp=False, oxocatalysis=False, SASA=False):
    list_of_props = list()
    list_of_props.append('name')
    list_of_props.append('gene')
    list_of_props.append('metal')
    list_of_props.append('alpha')
    list_of_props.append('axlig1')
    list_of_props.append('axlig2')
    list_of_props.append('eqlig')
    list_of_prop_names = ['chem_name', 'converged', 'status', 'time', 'charge', 'spin',
                          'energy', 'init_energy',
                          'ss_act', 'ss_target',
                          'ax1_MLB', 'ax2_MLB', 'eq_MLB',
                          "alphaHOMO", "alphaLUMO", "betaHOMO", "betaLUMO",
                          'geopath', 'attempted',
                          'flag_oct', 'flag_list', 'num_coord_metal',
                          'rmsd_max', 'atom_dist_max',
                          'oct_angle_devi_max', 'max_del_sig_angle', 'dist_del_eq', 'dist_del_all',
                          'devi_linear_avrg', 'devi_linear_max',
                          'flag_oct_loose', 'flag_list_loose',
                          'prog_num_coord_metal', 'prog_rmsd_max', 'prog_atom_dist_max',
                          'prog_oct_angle_devi_max', 'prog_max_del_sig_angle', 'prog_dist_del_eq',
                          'prog_dist_del_all',
                          'prog_devi_linear_avrg', 'prog_devi_linear_max',
                          'rmsd', 'maxd',
                          'init_ax1_MLB', 'init_ax2_MLB', 'init_eq_MLB', 'thermo_cont', 'imag', 'solvent_cont',
                          'terachem_version', 'terachem_detailed_version',
                          'basis', 'alpha_level_shift', 'beta_level_shift', 'functional', 'mop_energy',
                          'mop_coord']
    if SASA:
        list_of_prop_names.append("area")
    if oxocatalysis:
        if comp:
            list_of_props.append('convergence')
            for props in list_of_prop_names:
                for spin_cat in ['LS', 'IS', 'HS']:
                    for catax in ['x', 'oxo', 'hydroxyl']:
                        if catax == 'x':
                            for ox in ['2', '3']:
                                list_of_props.append("_".join(['ox', str(ox), spin_cat, str(catax), props]))
                        elif catax == 'oxo':
                            for ox in ['4', '5']:
                                list_of_props.append("_".join(['ox', str(ox), spin_cat, str(catax), props]))
                        else:
                            for ox in ['3', '4']:
                                list_of_props.append("_".join(['ox', str(ox), spin_cat, str(catax), props]))
            list_of_props.append('attempted')
        else:
            list_of_props += list_of_prop_names
    else:
        if comp:
            list_of_props.insert(1, 'ox2RN')
            list_of_props.insert(2, 'ox3RN')
            for props in list_of_prop_names:
                for spin_cat in ['LS', 'HS']:
                    for ox in ['2', '3']:
                        list_of_props.append("_".join(['ox', str(ox), spin_cat, props]))
            list_of_props.append('attempted')
        else:
            list_of_props.insert(1, 'number')
            list_of_props.insert(2, 'ox')
            list_of_props += list_of_prop_names
    return list_of_props


########################
def find_live_jobs():
    path_dictionary = setup_paths()
    live_job_dictionary = dict()
    if os.path.exists(path_dictionary["job_path"] + "/live_jobs.csv"):
        emsg, live_job_dictionary = read_dictionary(path_dictionary["job_path"] + "/live_jobs.csv")
    else:
        live_job_dictionary = dict()
    return live_job_dictionary


########################
def get_metals():
    metals_list = ['cr', 'mn', 'fe', 'co']
    return metals_list


########################
def get_ox_states():  # could be made metal dependent like spin
    GA_run = get_current_GA()
    if GA_run.config["oxocatalysis"]:
        ox_list = [2, 3, 4, 5]
    else:
        ox_list = [2, 3]
    return ox_list


########################
def spin_dictionary():
    GA_run = get_current_GA()
    if GA_run.config["use_singlets"]:
        if GA_run.config["all_spins"]:
            metal_spin_dictionary = {'co': {2: [2, 4], 3: [1, 3, 5], 4: [2, 4, 6], 5: [1, 3, 5]},
                                     'cr': {2: [1, 3, 5], 3: [2, 4], 4: [1, 3], 5: [2]},
                                     'fe': {2: [1, 3, 5], 3: [2, 4, 6], 4: [1, 3, 5], 5: [2, 4]},
                                     'mn': {2: [2, 4, 6], 3: [1, 3, 5], 4: [2, 4], 5: [1, 3]}}
        else:
            metal_spin_dictionary = {'co': {2: [2, 4], 3: [1, 5]},
                                     'cr': {2: [1, 5], 3: [2, 4]},
                                     'fe': {2: [1, 5], 3: [2, 6]},
                                     'mn': {2: [2, 6], 3: [1, 5]}}
    else:
        if GA_run.config["all_spins"]:
            metal_spin_dictionary = {'co': {2: [2, 4], 3: [1, 3, 5], 4: [2, 4], 5: [1, 3, 5]},
                                     'cr': {2: [1, 3, 5], 3: [2, 4], 4: [1, 3], 5: [2]},
                                     'fe': {2: [1, 3, 5], 3: [2, 4, 6], 4: [1, 3, 5], 5: [2, 4]},
                                     'mn': {2: [2, 4, 6], 3: [1, 3, 5], 4: [2, 4], 5: [1, 3]}}
        else:
            metal_spin_dictionary = {'co': {2: [2, 4], 3: [1, 5]},
                                     'cr': {2: [3, 5], 3: [2, 4]},
                                     'fe': {2: [1, 5], 3: [2, 6]},
                                     'mn': {2: [2, 6], 3: [3, 5]}}
    return metal_spin_dictionary


########################
def isDFT():
    GA_run = get_current_GA()
    if GA_run.config["DFT"]:
        return True
    else:
        return False
    return rdir


########################
def isSASA():
    GA_run = get_current_GA()
    try:
        if GA_run.config["SASA"]:
            return True
        else:
            return False
    except:
        return False


########################
def isOxocatalysis():
    GA_run = get_current_GA()
    try:
        if GA_run.config["oxocatalysis"]:
            return True
        else:
            return False
    except:
        return False


########################
def isall_post():
    GA_run = get_current_GA()
    if unicode('post_all', 'utf-8') in GA_run.config.keys():
        return GA_run.config["post_all"]
    else:
        return False


########################
def isOptimize():
    GA_run = get_current_GA()
    if GA_run.config["optimize"]:
        return True
    else:
        return False
    return rdir


########################
def translate_job_name(job):
    base = os.path.basename(job)
    base = base.strip("\n")
    basename = base.strip(".in")
    basename = basename.strip(".xyz")
    basename = basename.strip(".out")
    ll = (str(basename)).split("_")
    # print(ll)
    gen = ll[1]
    slot = ll[3]
    metal = int(ll[4])
    ox = int(ll[5])
    eqlig_ind = int(ll[6])
    axlig1_ind = int(ll[7])
    axlig2_ind = int(ll[8])
    ligands_dict = get_ligands()
    if hasattr(ligands_dict[int(eqlig_ind)][0], '__iter__'):  # SMILEs string
        #eqlig = 'smi' + str(eqlig_ind)
        eqlig = ligands_dict[int(eqlig_ind)][0][0]
    else:
        eqlig = ligands_dict[int(eqlig_ind)][0]
    if hasattr(ligands_dict[int(axlig1_ind)][0], '__iter__'):  # SMILEs string
        #axlig1 = 'smi' + str(axlig1_ind)
        axlig1 = ligands_dict[int(axlig1_ind)][0][0]
    else:
        axlig1 = ligands_dict[int(axlig1_ind)][0]

    if hasattr(ligands_dict[int(axlig2_ind)][0], '__iter__'):  # SMILEs string
        #axlig2 = 'smi' + str(axlig2_ind)
        axlig2 = ligands_dict[int(axlig2_ind)][0][0]
    else:
        axlig2 = ligands_dict[int(axlig2_ind)][0]
    ahf = int(ll[9])
    spin = int(ll[10])
    metal_list = get_metals()
    metal_key = metal_list[metal]
    metal_spin_dictionary = spin_dictionary()
    these_states = metal_spin_dictionary[metal_key][ox]
    if spin == these_states[0]:  # First element of list
        spin_cat = 'LS'
    elif spin == these_states[-1]:  # Last element of list
        spin_cat = 'HS'
    else:
        # print('spin assigned as ll[9]  = ' + str(spin) + ' on  ' +str(ll))
        # print('critical erorr, unknown spin: '+ str(spin))
        spin_cat = 'IS'  # Intermediate Spin
    gene = "_".join([str(metal), str(ox), str(eqlig_ind), str(axlig1_ind), str(axlig2_ind), str(ahf).zfill(2)])
    basegene = "_".join([str(metal), str(eqlig_ind), str(axlig1_ind), str(axlig2_ind)])
    return gene, gen, slot, metal, ox, eqlig, axlig1, axlig2, eqlig_ind, axlig1_ind, axlig2_ind, spin, spin_cat, ahf, basename, basegene


########################
def renameHFX(job, newHFX):
    # renames job to a new HFX fraction
    base = os.path.basename(job)
    base = base.strip("\n")
    basename = base.strip(".in")
    basename = base.strip(".xyz")
    basename = base.strip(".out")
    ll = (str(basename)).split("_")
    ## replace alpha
    ll[9] = newHFX
    new_name = "_".join(ll)
    return new_name


########################
def stripName(job):
    # gets base job name
    base = os.path.basename(job)
    base = base.strip("\n")
    basename = base.strip(".in")
    basename = basename.strip(".xyz")
    basename = basename.strip(".out")
    return basename


#######################
def renameOxoEmpty(job):
    # renames Oxo job to empty job
    base = os.path.basename(job)
    base = base.strip("\n")
    basename = base.strip(".in")
    basename = base.strip(".xyz")
    basename = base.strip(".out")
    ll = (str(basename)).split("_")
    ligs = get_ligands()
    for i, item in enumerate(ligs):
        if 'x' in item:
            value = str(i)
    ## replace ax2 with x index
    ll[8] = value
    ## replace metal oxidation with 2 less
    ll[5] = str(int(ll[5]) - 2)
    new_name = "_".join(ll)
    return new_name, basename


#######################
def to_decimal_string(inp):
    # nusiance function to convert
    # int strings (in %) to decimal strings
    out = str(float(inp) / 100)
    return out


def HFXordering():
    # this function returns the dictionary
    # of HFX fractions used, where the keys
    # represent the just finshed calculation
    # and the values are:
    # [next job to run, guess for next job]
    GA_run = get_current_GA()
    if not GA_run.config['HFXsample']:
        HFXdictionary = dict()
    else:
        HFXdictionary = {"20": ["25", "20"],
                         "25": ["30", "25"],
                         "30": ["15", "20"],
                         "15": ["10", "15"],
                         "10": ["05", "10"],
                         "5": ["00", "05"]}
    return (HFXdictionary)


########################

def setup_paths():
    working_dir = get_run_dir()
    path_dictionary = {
        "geo_out_path": working_dir + "geo_outfiles/",
        "sp_out_path": working_dir + "sp_outfiles/",
        "sp_in_path": working_dir + "sp_infiles/",
        "scr_path": working_dir + "scr/geo/",
        "queue_output": working_dir + "queue_output/",
        "thermo_out_path": working_dir + "thermo_outfiles/",
        "solvent_out_path": working_dir + "solvent_outfiles/",
        "job_path": working_dir + "jobs/",
        "done_path": working_dir + "completejobs/",
        "initial_geo_path": working_dir + "initial_geo/",
        "optimial_geo_path": working_dir + "optimized_geo/",
        "prog_geo_path": working_dir + "prog_geo/",
        "stalled_jobs": working_dir + "stalled_jobs/",
        "archive_path": working_dir + "archive_resub/",
        "state_path": working_dir + "statespace/",
        "molsimplify_inps": working_dir + "ms_inps/",
        "infiles": working_dir + "infiles/",
        "mopac_path": working_dir + "mopac/",
        "ANN_output": working_dir + "ANN_ouput/",
        "ms_reps": working_dir + "ms_reps/",
        "good_reports": working_dir + "reports/good_geo/",
        "bad_reports": working_dir + "reports/bad_geo/",
        "other_reports": working_dir + "reports/other/",
        "pdb_path": working_dir + "pdb/"}

    #    shutil.copyfile(get_source_dir()+'wake.sh',get_run_dir()+'wake.sh')
    ## set scr path to scr/sp for single points
    if not isOptimize():
        path_dictionary.update({"scr_path": working_dir + "scr/geo/"})
    GA_run = get_current_GA()
    if "DLPNO" in GA_run.config.keys():
        if GA_run.config["DLPNO"]:
            path_dictionary.update({"DLPNO_path": working_dir + "DLPNO_files/"})

    for keys in path_dictionary.keys():
        ensure_dir(path_dictionary[keys])
    return path_dictionary


########################

def advance_paths(path_dictionary, generation):
    new_dict = dict()
    for keys in path_dictionary.keys():
        if not (keys in ["molsimp_path", "DLPNO_path", "good_reports", "other_reports", "bad_reports", "pdb_path"]):
            new_dict[keys] = path_dictionary[keys] + "gen_" + str(generation) + "/"
            ensure_dir(new_dict[keys])
    return new_dict


########################

def get_ligands():
    GA_run = GA_run_defintion()
    GA_run.deserialize('.madconfig')
    ligands_list = GA_run.config['liglist']
    return ligands_list


########################

def write_dictionary(dictionary, path, force_append=False):
    emsg = False
    if force_append:
        write_control = 'a'
    else:
        write_control = 'w'
    try:
        with open(path, write_control) as f:
            for keys in dictionary.keys():
                f.write(str(keys).strip("\n") + ',' + str(dictionary[keys]) + '\n')
    except:
        emsg = "Error, could not write state space: " + path
    return emsg


########################

def find_split_fitness(split_energy, split_parameter):
    en = -1 * numpy.power((float(split_energy) / split_parameter), 2.0)
    fitness = numpy.exp(en)
    return fitness


########################

def find_split_dist_fitness(split_energy, split_parameter, distance, distance_parameter):
    ##FITNESS DEBUGGING: print "scoring function: split+dist YAY"

    en = -1 * (numpy.power((float(split_energy) / split_parameter), 2.0) + numpy.power(
        (float(distance) / distance_parameter), 2.0))
    fitness = numpy.exp(en)
    return fitness


########################

def write_summary_list(outcome_list, path):
    emsg = False
    try:
        with open(path, 'w') as f:
            for tups in outcome_list:
                for items in tups:
                    f.write(str(items) + ',')
                f.write('\n')
    except:
        emsg = "Error, could not write state space: " + path
    return emsg


########################
def read_dictionary(path):
    emsg = False
    dictionary = dict()
    try:
        with open(path, 'r') as f:
            for lines in f:
                ll = lines.split(",")
                key = ll[0]
                value = (",".join(ll[1:])).rstrip("\n")
                dictionary[key] = value
    except:
        emsg = "Error, could not read state space: " + path
    return emsg, dictionary


########################
def logger(path, message):
    ensure_dir(path)
    with open(path + '/log.txt', 'a') as f:
        f.write(message + "\n")


########################
def log_bad_initial(job):
    path = get_run_dir() + 'bad_initgeo_log.txt'
    if os.path.isfile(path):
        with open(path, 'a') as f:
            f.write(job + "\n")
    else:
        with open(path, 'w') as f:
            f.write(job + "\n")


########################
def add_to_outstanding_jobs(job):
    current_outstanding = get_outstanding_jobs()
    if job in current_outstanding:
        print('*** att skipping ' + str(job) + ' since it is in list')
    else:
        current_outstanding.append(job)
        print('*** att adding ' + str(job) + ' since it is not in list')
    set_outstanding_jobs(current_outstanding)


######################
def check_job_converged_dictionary(job):
    converged_job_dictionary = find_converged_job_dictionary()
    this_status = 'unknown'
    try:
        this_status = int(converged_job_dictionary[job])
    except:
        print('could not find status for  ' + str(job) + '\n')
        pass
    return this_status


########################
def get_outstanding_jobs():
    path_dictionary = setup_paths()
    path = path_dictionary['job_path']
    ensure_dir(path)
    list_of_jobs = list()
    if os.path.exists(path + '/outstanding_job_list.txt'):
        with open(path + '/outstanding_job_list.txt', 'r') as f:
            for lines in f:
                list_of_jobs.append(lines.strip('\n'))
    return list_of_jobs


########################
def set_outstanding_jobs(list_of_jobs):
    path_dictionary = setup_paths()
    path = path_dictionary['job_path']
    ensure_dir(path)
    with open(path + '/outstanding_job_list.txt', 'w') as f:
        for jobs in list_of_jobs:
            f.write(jobs.strip("\n") + "\n")
    print('written\n')


########################
def remove_outstanding_jobs(job):
    print('removing job: ' + job)
    path_dictionary = setup_paths()
    path = path_dictionary["job_path"]
    current_outstanding = get_outstanding_jobs()
    if job in current_outstanding:
        print(str(job) + ' removed since it is in list')
        current_outstanding.remove(job)
    else:
        print(str(job) + ' not removed since it is not in list')
    with open(path + '/outstanding_job_list.txt', 'w') as f:
        for jobs in current_outstanding:
            f.write(jobs + "\n")


########################
def purge_converged_jobs(job):
    print('removing job: ' + job)
    path_dictionary = setup_paths()
    path = path_dictionary["job_path"]

    converged_job_dictionary = find_converged_job_dictionary()
    this_status = 'unknown'
    if job in converged_job_dictionary.keys():
        this_status = int(converged_job_dictionary[job])
        print(' removing job with status  ' + str(this_status) + '\n')
        converged_job_dictionary.pop(job)
        write_dictionary(converged_job_dictionary, path_dictionary["job_path"] + "/converged_job_dictionary.csv")
    else:
        print(str(job) + ' not removed since it is not in conv keys')


########################
def find_converged_job_dictionary():
    path_dictionary = setup_paths()
    converged_job_dictionary = dict()
    if os.path.exists(path_dictionary["job_path"] + "/converged_job_dictionary.csv"):
        emsg, converged_job_dictionary = read_dictionary(path_dictionary["job_path"] + "/converged_job_dictionary.csv")
    else:
        converged_job_dictionary = dict()
    return converged_job_dictionary


########################
def update_converged_job_dictionary(jobs, status):
    path_dictionary = setup_paths()
    converged_job_dictionary = find_converged_job_dictionary()
    converged_job_dictionary.update({jobs: status})
    if status != 0:
        print(' wrtiting ' + str(jobs) + ' as status ' + str(status))
    write_dictionary(converged_job_dictionary, path_dictionary["job_path"] + "/converged_job_dictionary.csv")


########################
def find_submitted_jobs():
    path_dictionary = setup_paths()
    if os.path.exists(path_dictionary["job_path"] + "/submitted_jobs.csv"):
        emsg, submitted_job_dictionary = read_dictionary(path_dictionary["job_path"] + "/submitted_jobs.csv")
    else:
        submitted_job_dictionary = dict()

    return submitted_job_dictionary


########################
def purge_submitted_jobs(job):
    print('removing job: ' + job)
    path_dictionary = setup_paths()
    path = path_dictionary["job_path"]

    submitted_job_dictionary = find_submitted_jobs()

    if job in submitted_job_dictionary.keys():
        this_status = int(submitted_job_dictionary[job])
        print(' removing job with sub number  ' + str(this_status) + '\n')
        submitted_job_dictionary.pop(job)
        write_dictionary(submitted_job_dictionary, path_dictionary["job_path"] + "/submitted_jobs.csv")
    else:
        print(str(job) + ' not removed since it is not in subm keys')


########################
def writeprops(extrct_props, newfile):
    string_to_write = ','.join([str(word) for word in extrct_props])
    newfile.write(string_to_write)
    newfile.write("\n")
    return


########################
def atrextract(a_run, list_of_props):
    extrct_props = []
    for props in list_of_props:
        extrct_props.append(getattr(a_run, props))
    return extrct_props


########################
def write_descriptor_csv(list_of_runs, file_handle, append=False):
    print('writing a file a new descriptor file')
    if list_of_runs:
        nl = len(list_of_runs[0].descriptor_names)
        file_handle.write('runs,')
        n_cols = len(list_of_runs[0].descriptor_names)
        if not append:
            print('first element has ' + str(n_cols) + ' columns')
            if n_cols == 0:
                file_handle.write('\n')
            for i, names in enumerate(list_of_runs[0].descriptor_names):
                if i < (n_cols - 1):
                    file_handle.write(names + ',')
                else:
                    file_handle.write(names + '\n')
        for runs in list_of_runs:
            try:
                file_handle.write(runs.name)
                counter = 0
                print('found ' + str(len(runs.descriptors)) + ' descriptors ')
                for properties in runs.descriptors:
                    file_handle.write(',' + str(properties))
                file_handle.write('\n')
            except:
                pass
    else:
        pass


########################
def write_output(name, list_of_things_with_props, list_of_props, base_path_dictionary=False, rdir=False, postall=False):
    ## this function flexibly writes output files
    # for both the run and comparison classes
    # this fuinction supports overloading the default run directories through
    # optional arguments in order to be useable in environments where
    # no .madconfig is available and should only be used for this purpose
    if not base_path_dictionary:
        base_path_dictionary = setup_paths()
    if not rdir:
        rdir = get_run_dir()
    if not postall:
        postall = isall_post()

    output_path = rdir + '/' + name + '_results_post.csv'
    descriptor_path = rdir + '/' + name + '_descriptor_file.csv'

    if (not postall) and os.path.isfile(output_path):
        with open(output_path, 'a') as f:
            for thing in list_of_things_with_props:
                values = atrextract(thing, list_of_props)
                writeprops(values, f)
    else:
        with open(output_path, 'w') as f:
            writeprops(list_of_props, f)
            for thing in list_of_things_with_props:
                values = atrextract(thing, list_of_props)
                writeprops(values, f)

    if (not postall) and os.path.isfile(descriptor_path):
        with open(descriptor_path, 'a') as f:
            write_descriptor_csv(list_of_things_with_props, f, append=True)
    else:
        with open(descriptor_path, 'w') as f:
            write_descriptor_csv(list_of_things_with_props, f, append=False)
    return output_path, descriptor_path


########################
def write_run_reports(all_runs):
    print('writing outpickle and reports! patience is a virtue')
    path_dictionary = setup_paths()
    for runClass in all_runs.values():
        if runClass.status in [0, 1, 2, 7, 8, 12, 13, 14] and runClass.alpha == 20.0:
            if runClass.status in [0]:
                runClass.reportpath = path_dictionary["good_reports"] + runClass.name + ".pdf"
            elif runClass.status in [1, 8]:
                runClass.reportpath = path_dictionary["bad_reports"] + runClass.name + ".pdf"
            else:
                runClass.reportpath = path_dictionary["other_reports"] + runClass.name + ".pdf"
            runClass.DFTRunToReport()


########################
def write_run_pickle(final_results):
    output = open('final_runs_pickle.pkl', 'wb')
    pickle.dump(final_results, output)
    output.close()


########################
def process_run_post(filepost, filedescriptors):
    geo_flags = ['flag_oct', 'flag_list']
    geo_metrics = ['num_coord_metal', 'rmsd_max', 'atom_dist_max',
                   'oct_angle_devi_max', 'max_del_sig_angle',
                   'dist_del_eq', 'dist_del_all',
                   'devi_linear_avrg', 'devi_linear_max']
    prog_geo_flags = ['%s_loose' % x for x in geo_flags]
    prog_geo_metrics = ['prog_%s' % x for x in geo_metrics]
    file_prefix = filepost.split('.')[0]
    df1 = pd.read_csv(filepost)
    df2 = pd.read_csv(filedescriptors)
    df2 = df2.rename(index=str, columns={'runs': 'name'})
    df = pd.merge(df1, df2, how='right', on=['name'])
    df_conv = df[df['converged'] == True]
    df_unconv = df[df['converged'] == False]
    df_conv = df_conv.drop(columns=prog_geo_metrics, axis=1)
    df_conv = df_conv.drop(columns=prog_geo_flags, axis=1)
    df_unconv = df_unconv.drop(columns=geo_metrics, axis=1)
    df_unconv = df_unconv.drop(columns=geo_flags, axis=1)
    df_conv.to_csv('%s_converged.csv' % file_prefix)
    df_unconv = df_unconv[df_unconv['status'] != 3]
    df_unconv.to_csv('%s_unconverged.csv' % file_prefix)
    df_noprog = df_unconv[df_unconv['status'] == 3]
    df_noprog.to_csv('%s_noprogress.csv' % file_prefix)
