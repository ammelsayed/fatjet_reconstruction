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


debug = 1

def find_bosons(Particle_branch, boson_pid, daughter_pids):
    bosons = []
    seen = set()
    n_particles = Particle_branch.GetEntries()
    for i in range(n_particles):
        p = Particle_branch.At(i)
        if fabs(p.PID) != boson_pid: continue
        lc_idx = i
        visited = set()
        while lc_idx not in visited:
            visited.add(lc_idx)
            p_lc = Particle_branch.At(lc_idx)
            d1, d2 = int(p_lc.D1), int(p_lc.D2)
            if d1 < 0 or d1 >= n_particles: break
            found_next = False
            for d_idx in range(d1, d2 + 1):
                if fabs(Particle_branch.At(d_idx).PID) == boson_pid and d_idx > lc_idx:
                    lc_idx = d_idx
                    found_next = True
                    break
            if not found_next: break
        if lc_idx in seen: continue
        seen.add(lc_idx)
        p_lc = Particle_branch.At(lc_idx)
        d1, d2 = int(p_lc.D1), int(p_lc.D2)
        if d1 < 0 or d2 < 0: continue
        all_match = True
        for d_idx in range(d1, d2 + 1):
            if fabs(Particle_branch.At(d_idx).PID) not in daughter_pids:
                all_match = False
                break
        if all_match:
            bosons.append(p_lc)
    return bosons

def get_fatjets(FatJet_branch):
    return [(FatJet_branch.At(i).P4(), FatJet_branch.At(i).SoftDroppedP4[0].M()) for i in range(FatJet_branch.GetEntries())]

def fatjet_efficiency(inputRootFile):

    print("Reading MC events from the root file:")
    print(f" > {inputRootFile}")

    Chain = ROOT.TChain("Delphes")
    if isinstance(inputRootFile, list):
        for file in inputRootFile: Chain.Add(file)
    elif isinstance(inputRootFile, str):
        Chain.Add(inputRootFile)
    else:
        raise TypeError(f"inputRootFile must be str or list, got {type(inputRootFile).__name__}")
    
    TreeReader = ROOT.ExRootTreeReader(Chain)
    FatJet_branch = TreeReader.UseBranch("FatJet")
    Particle_branch = TreeReader.UseBranch("Particle")
    Weight_branch = TreeReader.UseBranch("Weight")

    DeltaRCut = 1.0
    pTRatioLow = 0.5
    pTRatioHigh = 2.0
    mWindow = 25

    bin_width = 50
    pt_bins = list(range(100, 1000+bin_width, bin_width))
    n_bins = len(pt_bins) - 1
    bins_arr = array.array('d', pt_bins)

    boson_defs = {
        24: {"name": "W", "mass": 80.4, "daughter_pids": {1,2,3,4,5}, "color": 2},
        23: {"name": "Z", "mass": 91.2, "daughter_pids": {1,2,3,4,5}, "color": 4},
        25: {"name": "H", "mass": 125.0, "daughter_pids": {5}, "color": 3}
    }
    
    hists = {}
    for pid, props in boson_defs.items():
        name = props["name"]
        hists[pid] = {
            "total": ROOT.TH1D(f"h_total_{name}", f"Total {name}", n_bins, bins_arr),
            "matched": ROOT.TH1D(f"h_matched_{name}", f"Matched {name}", n_bins, bins_arr)
        }
       
    numberOfEntries = TreeReader.GetEntries()
    for entry in tqdm(range(numberOfEntries)):
        TreeReader.ReadEntry(entry)
        w_gen = Weight_branch.At(0).Weight

        if entry > 1000: break

        all_gen = []
        for pid, props in boson_defs.items():
            bosons = find_bosons(Particle_branch, pid, props["daughter_pids"])
            for p in bosons:
                hists[pid]["total"].Fill(p.PT, w_gen)
                all_gen.append((pid, p))
        
        fj_list = get_fatjets(FatJet_branch)
        used = [False] * len(fj_list)
        
        all_gen.sort(key=lambda x: x[1].PT, reverse=True)
        for pid, p in all_gen:
            mass = boson_defs[pid]["mass"]
            best_idx = -1
            best_dr = 1e9
            for fj_idx, (fj_p4, fj_m) in enumerate(fj_list):
                if used[fj_idx]: continue
                dR = p.P4().DeltaR(fj_p4)
                ptRatio = fj_p4.Pt() / p.PT if p.PT > 0 else 0
                if dR <= DeltaRCut and pTRatioLow <= ptRatio <= pTRatioHigh and (mass - mWindow <= fj_m <= mass + mWindow):
                    if dR < best_dr:
                        best_dr = dR
                        best_idx = fj_idx
            if best_idx != -1:
                used[best_idx] = True
                hists[pid]["matched"].Fill(p.PT, w_gen)

    c = ROOT.TCanvas("c_eff", "FatJet Efficiency", 800, 600)
    
    eff_hists = []
    first = True
    for pid, props in boson_defs.items():
        h_tot = hists[pid]["total"]
        h_mat = hists[pid]["matched"]
        total = h_tot.Integral()
        matched = h_mat.Integral()
        
        print(f"\nEfficiency vs pT for {props['name']}:")
        print("pT bin [GeV]   N_total     N_matched   Eff (%)")
        for i in range(n_bins):
            n_tot = h_tot.GetBinContent(i+1)
            n_mat = h_mat.GetBinContent(i+1)
            if n_tot == 0: continue
            eff = n_mat / n_tot * 100
            print(f"{pt_bins[i]:3d}-{pt_bins[i+1]:3d}     {n_tot:8.1f}      {n_mat:8.1f}      {eff:6.2f}")
        if total > 0:
            print(f"Total: {total:.1f} {props['name']} bosons, {matched:.1f} matched, {matched/total*100:.2f}%")
        else:
            print(f"No {props['name']} bosons found.")

        eff_hist = ROOT.TEfficiency(h_mat, h_tot)
        eff_hist.SetConfidenceLevel(0.68)
        eff_hist.SetMarkerStyle(20)
        eff_hist.SetMarkerSize(1.2)
        eff_hist.SetLineWidth(2)
        eff_hist.SetFillColor(0)
        eff_hist.SetMarkerColor(props["color"])
        eff_hist.SetLineColor(props["color"])
        eff_hists.append(eff_hist)

        if first:
            eff_hist.SetTitle("Reconstruction efficiency; Truth p_{T} [GeV]; Ak10-Jet Reconstruction Efficiency")
            eff_hist.Draw("AP")
            ROOT.gPad.Update()
            eff_hist.GetPaintedGraph().SetMaximum(1.1)
            first = False
        else:
            eff_hist.Draw("same P")

    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextSize(0.032)
    label.DrawLatex(0.18, 0.90, "#DeltaR(J_{tr.}, J_{re.}) #leq 1.0")
    label.DrawLatex(0.18, 0.84, "0.5 #leq p_{T}(J_{re.}) / p_{T}(J_{tr.}) #leq 2.0")
    label.DrawLatex(0.18, 0.78, "m_{re.}^{SD} & m_{tr.} #in [m_{W/Z/H}-20 GeV, m_{W/Z/H}+20 GeV]")

    leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    for i, pid in enumerate(boson_defs):
        leg.AddEntry(eff_hists[i], boson_defs[pid]["name"], "lp")
    leg.Draw()

    c.SetGrid()
    c.SaveAs("boson_efficiency_2.png")
    c.SaveAs("boson_efficiency_2.pdf")

if __name__ == "__main__":
    fatjet_efficiency("/data/ammelsayed/Framework/Samples/signal/fdm/Events/sig1000/tag_1_delphes_events.root")