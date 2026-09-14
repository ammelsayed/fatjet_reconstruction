import os
import array
import ROOT
from tqdm import tqdm
from math import fabs

DELPHES_PATH = os.environ.get("DELPHES_HOME", "/home/ammelsayed/softwares/MG5_aMC_v3_5_15/Delphes")
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

debug = 0

def fatjet_efficiency(inputRootFile):

    # read file
    Chain = ROOT.TChain("Delphes")
    if isinstance(inputRootFile, list):
        for file in inputRootFile:
            Chain.Add(file)
    elif isinstance(inputRootFile, str):
        Chain.Add(inputRootFile)
    else:
        raise TypeError(f"inputRootFile must be str or list, got {type(inputRootFile).__name__}")
    
    TreeReader = ROOT.ExRootTreeReader(Chain)
    FatJet_branch    = TreeReader.UseBranch("FatJet")
    Particle_branch  = TreeReader.UseBranch("Particle")
    Weight_branch     = TreeReader.UseBranch("Weight")

    DeltaRCut = 1.0
    pTRatioLow = 0.5
    pTRatioHigh = 2.0
    mWindow = 25
    mW, mZ, mH = 80.4, 91.2, 125

    bin_width = 50
    pt_bins = list(range(100, 1000+bin_width, bin_width))
    n_bins = len(pt_bins) - 1

    total_w, matched_w = 0.0, 0.0
    total_w_bin, matched_w_bin = [0] * n_bins, [0] * n_bins
       
    numberOfEntries = TreeReader.GetEntries()
    for entry in tqdm(range(numberOfEntries)):

        TreeReader.ReadEntry(entry)
        w_gen = Weight_branch.At(0).Weight

        if entry > 3000: break
        
        # Collect hadronically decaying W bosons in this event
        w_list, z_list, h_list, t_list = [], [], [], []
        for p in Particle_branch:
            apid = fabs(p.PID)
            # W bosons
            if apid == 24: 
                d1 = Particle_branch.At(int(p.D1)) if p.D1 >= 0 else None
                d2 = Particle_branch.At(int(p.D2)) if p.D2 >= 0 else None
                if d1 is not None and d2 is not None:
                    all_quarks = True
                    for d_idx in range(int(p.D1), int(p.D2) + 1):
                        dau_p = Particle_branch.At(d_idx)
                        if not (1 <= fabs(dau_p.PID) <= 5):
                            all_quarks = False
                            break
                    if all_quarks:
                        w_list.append(p.P4())
                        total_w += w_gen
                        pt_w = p.PT
                        for i in range(n_bins):
                            if pt_bins[i] <= pt_w < pt_bins[i+1]:
                                total_w_bin[i] += w_gen
                                break
                        if debug:
                            print(f"W boson found: PID={p.PID}, Status={p.Status}, Mass={p.Mass:.3f} GeV")
                            print(f"    Daughter1: PID={d1.PID}, Status={d1.Status}")
                            print(f"    Daughter2: PID={d2.PID}, Status={d2.Status}")
        
        # get list of fatjets in this event
        fj_list = []
        for fj_idx in range(FatJet_branch.GetEntries()):
            fj = FatJet_branch.At(fj_idx)
            fj_list.append((fj.P4(), fj.SoftDroppedP4[0].M()))
        used = [False] * len(fj_list)


        # W to fatjet matching
        if w_list:
            w_list = sorted(w_list, key=lambda w: w.Pt(), reverse=True)
            for w in w_list:
                best_idx = -1
                best_dr = 1e9
                for fj_idx, (fj_p4, fj_m) in enumerate(fj_list):
                    dR = w.DeltaR(fj_p4)
                    dR_OK = (dR <= DeltaRCut)
                    ptRatioOk = (pTRatioLow <= (fj_p4.Pt()/w.Pt() if w.Pt() > 0.0 else 0.0) <= pTRatioHigh)
                    massOk = (mW - mWindow <= fj_m <= mW + mWindow)
                    if used[fj_idx]: continue
                    if dR_OK and ptRatioOk and massOk:
                        if dR < best_dr:
                            best_dr = dR
                            best_idx = fj_idx

                if best_idx != -1:
                    used[best_idx] = True
                    matched_w += w_gen
                    pt_w = w.Pt()
                    for i in range(n_bins):
                        if pt_bins[i] <= pt_w < pt_bins[i+1]:
                            matched_w_bin[i] += w_gen
                            break

    print("\nEfficiency vs pT:")
    print("pT bin [GeV]   N_total   N_matched   Eff (%)")
    for i in range(n_bins):
        n_tot = total_w_bin[i]
        n_mat = matched_w_bin[i]
        if n_tot == 0:
            continue
        eff = n_mat / n_tot * 100
        print(f"{pt_bins[i]:.2f}-{pt_bins[i+1]:.2f}     {n_tot:.2f}      {n_mat:.2f}      {eff:6.2f}")
    print(f"\nTotal (all bins): {total_w} W bosons, {matched_w} matched, {matched_w/total_w*100:.2f}%" if total_w else "No W bosons found.")

    # ------------------------------------------------------------
    # Clopper‑Pearson intervals with ROOT’s TEfficiency (native)
    # ------------------------------------------------------------
    bins_arr = array.array('d', pt_bins)
    h_total = ROOT.TH1D("h_total", "Total W bosons", n_bins, bins_arr)
    h_matched = ROOT.TH1D("h_matched", "Matched W bosons", n_bins, bins_arr)
    for i in range(n_bins):
        h_total.SetBinContent(i+1, total_w_bin[i])
        h_matched.SetBinContent(i+1, matched_w_bin[i])

    eff_hist = ROOT.TEfficiency(h_matched, h_total)
    eff_hist.SetStatisticOption(0) # Clopper-Pearson interval (recommended by PDG)
    eff_hist.SetConfidenceLevel(0.68) 
    eff_hist.SetMarkerStyle(20)
    eff_hist.SetMarkerSize(1.2)
    eff_hist.SetLineWidth(2)
    eff_hist.SetFillColor(0) 



    # Plot and save
    c = ROOT.TCanvas("c_eff", "W FatJet Efficiency", 800, 600)
    eff_hist.SetTitle("W reconstruction efficiency; Truth p_{T} [GeV]; Ak10-Jet Reconstruction Efficiency")
    eff_hist.Draw("AP")

    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextSize(0.032)
    label.DrawLatex(0.2, 0.86, "#Delta R(J_{tr.}, J_{re.}) <= 1.0")
    label.DrawLatex(0.2, 0.80, "0.5 <= p_{T}(J_{re.}) / p_{T}(J_{tr.}) <= 2.0")
    label.DrawLatex(0.2, 0.74, "m_{re.}^{SD}, m_{tr.} #in [m_{W/Z/H} - 20 GeV, m_{W/Z/H} + 20 GeV]")

    c.SetGrid()
    c.SaveAs("w_efficiency.png")

    print("\nClopper‑Pearson 68% confidence intervals:")
    for i in range(n_bins):
        if total_w_bin[i] == 0:
            continue
        eff = eff_hist.GetEfficiency(i+1)
        low_err = eff_hist.GetEfficiencyErrorLow(i+1)   # absolute lower error
        up_err  = eff_hist.GetEfficiencyErrorUp(i+1)    # absolute upper error
        print(f"Bin {pt_bins[i]:.2f}-{pt_bins[i+1]:.2f}: eff = {eff*100:.2f}%  [{eff*100 - low_err*100:.2f}%, {eff*100 + up_err*100:.2f}%]")

if __name__ == "__main__":
    fatjet_efficiency("/data/ammelsayed/Framework/Samples/signal/fdm/Events/sig1000/tag_1_delphes_events.root")