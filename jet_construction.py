import os
import subprocess
import itertools
import time
import numpy as np
from rich.progress import track

# ----------------------------------------------------
# Simple setup
# ----------------------------------------------------
delphes_dir = "/home/ammelsayed/MG5_aMC_v3_7_0/Delphes"
root_files_dir = "/data/ammelsayed/Framework/Samples/Temp/Ready"
preprocessed_root_files_dir = "/data/ammelsayed/Framework/Samples/TempResults/RootFiles"
analysis_results_dir = "/data/ammelsayed/Framework/Samples/TempResults/Analysis"
shell_files_dir = "/data/ammelsayed/Framework/Samples/Temp/Shell"

# Path to the template and optional output directory
delphes_card_template_path = "/data/ammelsayed/Framework/Samples/Cards/delphes_card_ATLAS_FatJets_Default_Template.tcl"
delphes_cards_output_dir = "/data/ammelsayed/Framework/Samples/Temp/Cards"
overwrite_root_files = False

# Make sure directories exist
os.makedirs(delphes_cards_output_dir, exist_ok=True)
os.makedirs(root_files_dir, exist_ok=True)
os.makedirs(preprocessed_root_files_dir, exist_ok=True)
os.makedirs(analysis_results_dir, exist_ok=True)
os.makedirs(shell_files_dir, exist_ok=True)

# run with:
# nohup python3 jet_construction.py > jet_construction.log 2>&1 &
# ----------------------------------------------------
# Setting up Delphes libraries
# ----------------------------------------------------
import ROOT
DELPHES_PATH = os.environ.get("DELPHES_HOME", delphes_dir)
ROOT.gInterpreter.AddIncludePath(DELPHES_PATH)
ROOT.gInterpreter.AddIncludePath(f"{DELPHES_PATH}/classes")
ROOT.gInterpreter.AddIncludePath(f"{DELPHES_PATH}/external")
ROOT.gSystem.Load("libDelphes")
ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')
ROOT.gInterpreter.Declare('#include "classes/SortableObject.h"')
ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"')
ROOT.gROOT.SetBatch(True) 
ROOT.gROOT.SetStyle("ATLAS")
print("Using ROOT version:", ROOT.__version__)
print("Using Delphes libraries found at:", DELPHES_PATH)
# ----------------------------------------------------

def build_cards():
    start_time = time.time()
    input_arrays = ["EFlowMerger/eflow", "EFlowFilter/eflow"]
    jet_algorithms = [6]
    par_R = [0.8, 1.0, 1.2]
    nsubjettiness_beta = [1.0, 2.0]
    nsubjettiness_axismodes = [2, 4]
    mit_pt = [150.0]

    card_options = [
        {
            "InputArray": opt1,
            "JetAlgorithm": opt2,
            "ParameterR": opt3, 
            "Nsubjettiness_Beta": opt4, 
            "Nsubjettiness_AxisMode": opt5, 
            "JetPTMin": opt6
        }
        for opt1, opt2, opt3, opt4, opt5, opt6 in itertools.product(
            input_arrays, jet_algorithms, par_R, 
            nsubjettiness_beta, nsubjettiness_axismodes, mit_pt
        )
    ]

    # write suffx and card_name
    for opts in card_options:
        parts = []
        for key, value in opts.items():
            if key == "InputArray":
                parts.append(f"FatJetInputArray_{str(value).replace('/','_')}")
            else:
                parts.append(f"{key}_{value}")
        suffix = "__".join(parts)
        opts["suffix"] = suffix
        opts["card_path"] = f"DelphesCard_ATLAS_{suffix}.tcl"


    # Check if template exists and is readable
    if not os.path.isfile(delphes_card_template_path):
        raise FileNotFoundError(f"Template file not found: {delphes_card_template_path}")

    # Read the entire template
    with open(delphes_card_template_path, "r") as template:
        template_text = template.read()

    # Generate each card
    for opts in card_options:
        # Replace placeholders with actual values
        new_card = template_text
        new_card = new_card.replace("__INPUTOPTION1__", str(opts["InputArray"]))
        new_card = new_card.replace("__INPUTOPTION2__", str(opts["JetAlgorithm"]))
        new_card = new_card.replace("__INPUTOPTION3__", str(opts["ParameterR"]))
        new_card = new_card.replace("__INPUTOPTION4__", str(opts["Nsubjettiness_Beta"]))
        new_card = new_card.replace("__INPUTOPTION5__", str(opts["Nsubjettiness_AxisMode"]))
        new_card = new_card.replace("__INPUTOPTION6__", str(opts["JetPTMin"]))

        # Write the new card to the output directory
        output_path = os.path.join(delphes_cards_output_dir, opts["card_path"])
        with open(output_path, "w") as out_file:
            out_file.write(new_card)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Generated {len(card_options)} card files in {delphes_cards_output_dir} within {elapsed:.3f} s.")
    return card_options


def is_file_healthy(file_path):
    """Check if file exists, is accessible, and can be opened for reading."""
    try:
        # Check existence first
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return False
        
        if not os.path.isfile(file_path):
            print(f"Path exists but is not a file: {file_path}")
            return False
        
        # Check read permission
        if not os.access(file_path, os.R_OK):
            print(f"No read permission: {file_path}")
            return False
            
        return True
        
    except Exception as e:
        print(f"File is not healthy: {e}")
        return False

def write_shell_file(path, name, commands):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{name}.sh")
    with open(file_path, "w") as out_file:
        out_file.write("\n".join(commands))
    os.chmod(file_path, 0o755)
    print(f"Written: {file_path}")


def build_shell_files(cards_data, procs_data):
    start_time = time.time()
    generated_procs = []
    missing_procs = []
    shells_files_for_gunzip = {}
    shells_files_for_delphes = {}
    for info in procs_data:
        proc = info["proc_name"]
        hepmc_file_path = info["hepmc_file_path"]
        hepmc_gz_path = f"{hepmc_file_path}.gz"
        proc_shell_files_dir = os.path.join(shell_files_dir, proc.upper())
        proc_root_files_dir = os.path.join(root_files_dir, proc.upper())
        os.makedirs(proc_root_files_dir, exist_ok=True)
        shells_files_for_gunzip[proc] = []
        shells_files_for_delphes[proc] = []

        # check if the hepmc file exists and is readable
        if is_file_healthy(hepmc_file_path):
            print(f"Found uncompressed HEPMC file: {hepmc_file_path}")

        # check if the file exists in .hepmc.gz format, it needs to be unzipped first
        # write the gunzip shell file & IT MUST RUN FIRST !
        elif is_file_healthy(hepmc_gz_path):
            print(f"Found compressed HEPMC file: {hepmc_gz_path}")
            gunzip_shell_commands = [
                f"#!/bin/bash",
                f"# Auto-generated at: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}",
                f"gunzip {hepmc_file_path}.gz"
            ]
            write_shell_file(proc_shell_files_dir, "UnzipHEPMC", gunzip_shell_commands)
            shells_files_for_gunzip[proc].append(os.path.join(proc_shell_files_dir, "UnzipHEPMC.sh"))

        # The processes not generated
        else:
            print(f"ERROR: Neither {hepmc_file_path} nor {hepmc_gz_path} found.")
            print(f"Please check your process generation pipeline for {proc}.")
            print(f"Skipping this processes ....")
            missing_procs.append(proc)
            continue

        generated_procs.append(proc)
        for card_opts in cards_data:
            output_root_file_path = os.path.join(proc_root_files_dir, f'{card_opts["suffix"]}.root')
            output_log_file_path = os.path.join(proc_root_files_dir, f'{card_opts["suffix"]}.log')
            delphes_card_path = os.path.join(delphes_cards_output_dir, card_opts["card_path"])
            delphes_command = f'{delphes_dir}/DelphesHepMC2 {delphes_card_path} {output_root_file_path} {hepmc_file_path}'
            delphes_shell_commands = [
                f"#!/bin/bash",
                f"# Auto-generated at: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}",
                f'cd "{delphes_dir}" || exit 1',
                f'nohup {delphes_command} > {output_log_file_path} 2>&1 &',
                f'echo "Delphes job started with PID: $!"',
                f'echo "  > cat {output_log_file_path}"',
                f'echo "  > tail -f {output_log_file_path}"',
            ]
            script_name = f'DelphesRunOPTION_{card_opts["suffix"]}'
            write_shell_file(proc_shell_files_dir, script_name, delphes_shell_commands)
            delphes_info = (os.path.join(proc_shell_files_dir, script_name + ".sh"), output_root_file_path)
            shells_files_for_delphes[proc].append(delphes_info)

    # Print summary
    end_time = time.time()
    elapsed = end_time - start_time
    print("\n" + "="*60)
    print(f"Generated shell scripts for: {', '.join(generated_procs)}")
    if missing_procs:
        print(f"Missing HEPMC files for: {', '.join(missing_procs)}")

    print(f"gunzip shell executables: {sum(len(items) for items in shells_files_for_gunzip.values())}")
    print(f"delphes shell executables: {sum(len(items) for items in shells_files_for_delphes.values())}")
    print(f"Built all shell executables within {elapsed:.3f} s.")
    print("="*60)

    return shells_files_for_gunzip, shells_files_for_delphes


def run_shell_scripts(gunzip_shell_files, delphes_shell_files):
    """
    Execute gunzip scripts (if any) synchronously, then launch all Delphes scripts.
    There is no need for check if .root files already exists because Delphes does this.
    """
    all_errors = []
    for proc in gunzip_shell_files.keys():
        print(f"\n--- Processing {proc} ---")
        gunzip_scripts = gunzip_shell_files[proc]
        delphes_info_list = delphes_shell_files[proc]

        # Run gunzip script if it exists (must finish before running Delphes)
        if gunzip_scripts:
            gunzip_script = gunzip_scripts[0]
            print(f"Running gunzip: {gunzip_script}")
            try:
                start_time = time.time()
                result = subprocess.run(["sh", gunzip_script], check=True, capture_output=True, text=True)
                end_time = time.time()
                print(result.stdout)
                print(f"Finished in {(end_time-start_time):.3f} s.")
            except subprocess.CalledProcessError as e:
                err_msg = f"- ERROR:{e.stderr}"
                print(err_msg)
                print(f"Finished in {(end_time-start_time):.3f} s.")
                all_errors.append(err_msg)
                continue # If gunzip fails, skip Delphes for this process

        # Launch all Delphes scripts (they background themselves via nohup) 
        for script, root_file in delphes_info_list:

            # Check if ROOT file already exists
            if os.path.exists(root_file) and (not overwrite_root_files):
                print(f"SKIP (already exists): {root_file}")
                continue

            print(f"Launching Delphes: {script}")
            try:
                result = subprocess.run(["sh", script], check=True, capture_output=True, text=True)
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                err_msg = f"- ERROR:{e.stderr}"
                print(err_msg)
                all_errors.append(err_msg)

    print("\n" + "="*50)
    if all_errors:
        print(f"\n{len(all_errors)} Errors occured.")
    else:
        print("\nAll shell scripts executed successfully.")
    print("\n" + "="*50)


def book_branches():
    Objects = {
        "Lepton":               {"counts":    1},
        "Muon":                 {"counts":    1},    
        "Electron":             {"counts":    1},    
        "FatJet":               {"counts":    2},
    }
    # Create complete list of all particles
    all_particles = []
    for obj_type, properties in Objects.items():
        count = properties["counts"]
        for i in range(count):
            all_particles.append(f"{obj_type}{i}")
    # k-body combinations for multi-body masses
    def combos_Nbody(N):
        return list(itertools.combinations(all_particles, N))
    def kinematics_Nbody(N):
        return ["M", "DeltaR", "DeltaPhi", "DeltaEta"]

    lepton_kinematics = ["PT", "Eta", "Phi"]
    fatjet_kinematics = ["PT", "Eta", "Phi", "Mass"]
    tau_indices = [1, 2, 3]
    tau_ratios = ["Tau21", "Tau32"]
    lepton_instances = [p for p in all_particles if p.startswith(("Muon","Electron", "Lepton"))]
    fatjet_instances = [p for p in all_particles if p.startswith(("FatJet"))]
    branch_names = [f"{k}_{inst}" for inst in lepton_instances for k in lepton_kinematics]
    branch_names += [f"{k}_{inst}" for inst in fatjet_instances for k in fatjet_kinematics]
    branch_names += [f"Tau{i}_FatJet{j}" for j in range(Objects["FatJet"]["counts"]) for i in tau_indices] 
    branch_names += [f"{ratio}_FatJet{j}" for j in range(Objects["FatJet"]["counts"]) for ratio in tau_ratios] 
    branch_names += [f"{k}_{'_'.join(p)}" for p in combos_Nbody(2) for k in kinematics_Nbody(2)]
    branch_names += ["weight"]

    print("\n" + "="*80)
    print("Booked branches:\n")
    ncol=2
    nrows=-(-len(branch_names)//ncol)
    print('\n'.join('  '.join(f"{j+i*nrows:>2}: {branch_names[j+i*nrows]:<30}" for i in range(ncol) if j+i*nrows < len(branch_names)) for j in range(nrows)))
    print("\n" + "="*80)

    return branch_names

def preprocess_root_file(inputFile, proc_name):

    # Read the input file
    Chain = ROOT.TChain("Delphes")
    Chain.Add(inputFile)
    TreeReader = ROOT.ExRootTreeReader(Chain)
    numberOfEntries = TreeReader.GetEntries()

    # Branches to read
    FatJet_branch   = TreeReader.UseBranch("FatJet")
    Electron_branch = TreeReader.UseBranch("Electron")
    Muon_branch     = TreeReader.UseBranch("Muon")
    Event_branch    = TreeReader.UseBranch("Event")

    tree = ROOT.TTree(proc_name, proc_name)
    tree.SetDirectory(0)

    ################################################
    # Book the branches
    ################################################
    Objects = {
        "Lepton":               {"counts":    1},
        "Muon":                 {"counts":    1},    
        "Electron":             {"counts":    1},    
        "FatJet":               {"counts":    2},
    }
    # Create complete list of all particles
    all_particles = []
    for obj_type, properties in Objects.items():
        count = properties["counts"]
        for i in range(count):
            all_particles.append(f"{obj_type}{i}")
    # k-body combinations for multi-body masses
    def combos_Nbody(N):
        return list(itertools.combinations(all_particles, N))
    def kinematics_Nbody(N):
        return ["M", "DeltaR", "DeltaPhi", "DeltaEta"]

    lepton_kinematics = ["PT", "Eta", "Phi"]
    fatjet_kinematics = ["PT", "Eta", "Phi", "Mass"]
    tau_indices = [1, 2, 3]
    tau_ratios = ["Tau21", "Tau32"]
    lepton_instances = [p for p in all_particles if p.startswith(("Muon","Electron", "Lepton"))]
    fatjet_instances = [p for p in all_particles if p.startswith(("FatJet"))]
    branch_names = [f"{k}_{inst}" for inst in lepton_instances for k in lepton_kinematics]
    branch_names += [f"{k}_{inst}" for inst in fatjet_instances for k in fatjet_kinematics]
    branch_names += [f"Tau{i}_FatJet{j}" for j in range(Objects["FatJet"]["counts"]) for i in tau_indices] 
    branch_names += [f"{ratio}_FatJet{j}" for j in range(Objects["FatJet"]["counts"]) for ratio in tau_ratios] 
    branch_names += [f"{k}_{'_'.join(p)}" for p in combos_Nbody(2) for k in kinematics_Nbody(2)]
    branch_names += ["weight"]
    b = {}   
    for name in branch_names:
        b[name] = np.zeros(1, dtype=np.float64)
        tree.Branch(name, b[name], f"{name}/D")  # /D for Double (64-bit float)

    # Loop over all events
    for entry in track(range(0, numberOfEntries)):
        TreeReader.ReadEntry(entry)

        # main MC event weight
        w = Event_branch[0].Weight

        # Reset branches to np.nan
        for name in b:
            b[name][0] = np.nan
        b["weight"][0] = 0.0

        P4s = {} # Dictionary to cache 4-momenta for combinations

        ################################################
        # Quality Selections
        ################################################

        # Collect good leptons
        goodLeptons, goodMuons, goodElectrons = [], [], []
        IsoCutMuon     = 0.1 ## looseCut = 0.3, mediumCut = 0.2, tightCut = 0.1
        IsoCutElectron = 0.2 ## looseCut = 0.3, mediumCut = 0.2, tightCut = 0.1
        for i in range(Muon_branch.GetEntries()):
            muon = Muon_branch.At(i)
            if (muon.IsolationVar < IsoCutMuon) and muon.PT > 25 and abs(muon.Eta) <= 2.5:
                goodLeptons.append(muon)
                goodMuons.append(muon)
        for i in range(Electron_branch.GetEntries()):
            electron = Electron_branch.At(i)
            if (electron.IsolationVar < IsoCutElectron) and electron.PT > 30 and abs(electron.Eta) <= 2.5:
                goodLeptons.append(electron)
                goodElectrons.append(electron)
        if len(goodLeptons) == 0:
            continue
        goodLeptons.sort(key=lambda lep: lep.PT, reverse=True) 

        # Collect good fatjets
        goodFatJets = []
        mW, mZ, mH = 80.4, 91.2, 125 # GeV
        for i in range(FatJet_branch.GetEntries()):
            fatjet = FatJet_branch.At(i)
            # if hasattr(fatjet, 'SoftDroppedP4[0]'):
            #     mWindow = 20
            #     MassOk = mW - mWindow <= fatjet.SoftDroppedP4[0].M() <= mH + mWindow
            # else:
            #     mWindow = 40
            #     MassOk = mW - mWindow <= fatjet.Mass <= mH + mWindow
            mWindow = 20
            MassOk = mW - mWindow <= fatjet.SoftDroppedP4[0].M() <= mH + mWindow
            DeltaROk = min(fatjet.P4().DeltaR(lep.P4()) for lep in goodLeptons) > 0.2
            if DeltaROk and MassOk and fatjet.PT > 100.0 and abs(fatjet.Eta) <= 2.5:
                goodFatJets.append(fatjet)
        if len(goodFatJets) == 0:
            continue
        goodFatJets.sort(key=lambda fj: fj.PT, reverse=True)

        ################################################
        # Signal Region Selections
        ################################################
        if not (len(goodLeptons) >= 1 and len(goodFatJets) >= 1):
            continue
        
        ################################################
        # Fill the branches
        ################################################

        # Fill lepton kinematics -----------------------
        for lep_idx, lepton in enumerate(goodLeptons[:1]):
            classname = lepton.ClassName()
            for lepType in [classname, "Lepton"]:
                inst = f"{lepType}{lep_idx}"
                p4 = lepton.P4()
                P4s[inst] = p4
                for k in lepton_kinematics:
                    try:
                        b[f"{k}_{inst}"][0] = getattr(lepton, k) # example: lep.PT
                    except AttributeError:
                        try:
                            b[f"{k}_{inst}"][0] = getattr(p4, k)() # example: lep.P4().Px()
                        except AttributeError as e:
                            print(f"Delphes {classname} object has no attribute {k} or .P4().{k}(). Error: {e}")


        # Fill fajet kinematics -----------------------
        for fj_idx, fatjet in enumerate(goodFatJets[:2]):
            classname = fatjet.ClassName()
            p4 = fatjet.P4()
            P4s[f"FatJet{fj_idx}"] = p4
            for k in fatjet_kinematics:
                try:
                    b[f"{k}_FatJet{fj_idx}"][0] = getattr(fatjet, k)
                except AttributeError:
                    try:
                        b[f"{k}_FatJet{fj_idx}"][0] = getattr(p4, k)()
                    except AttributeError as e:
                        print(f"Delphes {classname} object has no attribute {k} or .P4().{k}(). Error: {e}")

            # Tau substructures
            for t_idx in tau_indices:
                b[f"Tau{t_idx}_FatJet{fj_idx}"][0] = fatjet.Tau[t_idx - 1]
            t1, t2, t3 = fatjet.Tau[0], fatjet.Tau[1], fatjet.Tau[2]
            b[f"Tau21_FatJet{fj_idx}"][0] = t2 / t1 if t1 > 0 else np.nan
            b[f"Tau32_FatJet{fj_idx}"][0] = t3 / t2 if t2 > 0 else np.nan


        # Fill 2-body relations and masses -------------------------
        for p1_name, p2_name in combos_Nbody(2):
            p1, p2 = P4s.get(p1_name), P4s.get(p2_name)
            if p1 and p2:
                sufx = f"{p1_name}_{p2_name}"
                b[f"DeltaR_{sufx}"][0] = p1.DeltaR(p2)
                b[f"DeltaPhi_{sufx}"][0] = p1.DeltaPhi(p2)
                b[f"DeltaEta_{sufx}"][0] = p1.Eta() - p2.Eta()
                b[f"M_{sufx}"][0] = (p1 + p2).M()

        # Weight
        b["weight"][0] = w
        tree.Fill()

    return tree

def preprocess_root_files(cards_data, procs_data):
    """
    Loop over all the possible card obtions, read the root files of all the processes corresponding to each card option, and preprocess them to get the branches needed for the analysis, each process gets a corresponding tree inside the output root file.
    """
    start_time = time.time()
    branch_names = book_branches()
    for card_opts in cards_data:
        # book a root file for each card option
        output_root_file_path = os.path.join(preprocessed_root_files_dir, f'{card_opts["suffix"]}.root')
        outFile = ROOT.TFile(output_root_file_path, "RECREATE")
        for proc in procs_data:
            proc_name = proc["proc_name"]
            input_root_file_path = os.path.join(root_files_dir, proc_name.upper(), f'{card_opts["suffix"]}.root')
            print(f"processing {input_root_file_path}")
            tree = preprocess_root_file(input_root_file_path, proc_name.upper())
            nEntries = tree.GetEntries()
            outFile.cd()
            tree.Write()
            print(f"Written {nEntries} entries to {output_root_file_path}\n")
        outFile.Close()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Preprocessed {len(cards_data)} root files in {elapsed:.3f} s.")


import uproot
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import math

def analyze_data():
    for card_opts in cards_data:
        input_root_file_path = os.path.join(preprocessed_root_files_dir, f'{card_opts["suffix"]}.root')
        dataFrames = {}
        features = (
        ["DeltaR_Lepton0_FatJet0", "DeltaR_Lepton0_FatJet1", "DeltaPhi_Lepton0_FatJet0", "DeltaPhi_Lepton0_FatJet1"] +
        ["M_Lepton0_FatJet0", "M_Lepton0_FatJet1"] +
        ["Mass_FatJet0", "Mass_FatJet1", "Tau21_FatJet0", "Tau32_FatJet0", "Tau21_FatJet1", "Tau32_FatJet1"] +
        ["weight"]
        )
        with uproot.open(input_root_file_path) as inputFile:
            for key, obj in inputFile.items():
                if isinstance(obj, uproot.TTree):
                    df = obj.arrays(features, library="pd")
                    dataFrames[key] = df
                

        # plot
        for tree_name, dataframe in dataFrames.items():
            dataframe = dataframe.dropna(axis=1, how='all', ignore_index=True)
            dataframe = dataframe.dropna(axis=0, how='any', ignore_index=True)

            N = len(features)
            n_cols = math.ceil(math.sqrt(N))
            n_rows = math.ceil(N / n_cols)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
            axes_flat = axes.flatten()
            for i, feature in enumerate(features):
                if feature == "weight": continue
                array_B = dataframe[feature].values.astype(np.float32)
                weights_B = dataframe["weight"].values.astype(np.float64)
                axes_flat[i].hist(array_B, bins=30, weights=weights_B, alpha=0.7, color='blue')
                axes_flat[i].set_xlabel(feature)
                axes_flat[i].set_ylabel('Events')
            
            # Hide unused subplots if any
            for j in range(N, n_rows * n_cols):
                axes_flat[j].set_visible(False)

            plt.tight_layout()
            plt.savefig(os.path.join(analysis_results_dir, f"{tree_name}_{card_opts["suffix"]}.png"), dpi=500)
            plt.close()
            


                
if __name__ == '__main__':
    #--------------------------------------------------------
    # Step (1) generate .root files with different cards
    # -------------------------------------------------------

    procs_data = []
    for proc in ["tt", "WW", "WZ", "ZZ", "sig1000"]: # , "tt", "WW", "WZ", "ZZ" and signal!
        procs_data.append({
            "proc_name" : proc,
            "hepmc_file_path" : f"/data/ammelsayed/Framework/Samples/TempBkg/{proc.lower()}/Events/run_01/tag_1_pythia8_events.hepmc"
        })


    cards_data = build_cards()
    # gunzip_exec, delphes_exec = build_shell_files(cards_data, procs_data)
    # run_shell_scripts(gunzip_exec, delphes_exec)

    #--------------------------------------------------------
    # Step (2) read all the root files and compare
    # -------------------------------------------------------
    preprocess_root_files(cards_data, procs_data)
    # analyze_data()
