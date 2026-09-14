import os
import array
import ROOT
from tqdm import tqdm
from math import fabs
from parallelization import parallel_runs
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
overwrite_results = True
max_entries = None
results_file = "fatjet_reconstruction_efficiency_results.root"

xsec = 2.4410449915414665
lumi = 400

DeltaRCut = 1.0
pTRatioLow = 0.5
pTRatioHigh = 2.0
mWindow = 25

bin_width = 50
pt_bins = list(range(100, 2000+bin_width, bin_width))
n_bins = len(pt_bins) - 1
bins_arr = array.array('d', pt_bins)

boson_defs = {
    24: {"name": "W", "mass": 80.4, "daughter_pids": {1,2,3,4,5}, "color": 2},
    23: {"name": "Z", "mass": 91.2, "daughter_pids": {1,2,3,4,5}, "color": 4},
    25: {"name": "H", "mass": 125.0, "daughter_pids": {5}, "color": 3}
}

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
                    if debug: print(f"  [Debug] Tracing PID {boson_pid} idx {lc_idx} -> {d_idx}")
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
            if debug: print(f"  [Debug] Found hadronic {boson_pid} idx {lc_idx}: pt={p_lc.PT:.1f}, eta={p_lc.Eta:.2f}, phi={p_lc.Phi:.2f}, mass={p_lc.Mass:.1f}")
            bosons.append(p_lc)
    return bosons

def get_fatjets(FatJet_branch):
    return [(FatJet_branch.At(i).P4(), FatJet_branch.At(i).SoftDroppedP4[0].M()) for i in range(FatJet_branch.GetEntries())]

def build_chain(inputRootFile):
    chain = ROOT.TChain("Delphes")
    if isinstance(inputRootFile, list):
        for file in inputRootFile:
            chain.Add(file)
    elif isinstance(inputRootFile, str):
        chain.Add(inputRootFile)
    else:
        raise TypeError(f"inputRootFile must be str or list, got {type(inputRootFile).__name__}")
    return chain

def count_entries(inputRootFile):
    return build_chain(inputRootFile).GetEntries()

def fatjet_efficiency(inputRootFile, start_entry=0, end_entry=None, progress=True):

    if progress:
        print("Reading MC events from the root file:")
        print(f" > {inputRootFile}")

    Chain = build_chain(inputRootFile)
    TreeReader = ROOT.ExRootTreeReader(Chain)
    FatJet_branch = TreeReader.UseBranch("FatJet")
    Particle_branch = TreeReader.UseBranch("Particle")

    hists = {}
    for pid, props in boson_defs.items():
        name = props["name"]
        hists[pid] = {
            "total": ROOT.TH1D(f"h_total_{name}", f"Total {name}", n_bins, bins_arr),
            "matched": ROOT.TH1D(f"h_matched_{name}", f"Matched {name}", n_bins, bins_arr)
        }

    numberOfEntries = TreeReader.GetEntries()
    if end_entry is None or end_entry > numberOfEntries:
        end_entry = numberOfEntries

    w_gen = xsec * lumi / numberOfEntries

    rng = range(start_entry, end_entry)
    for entry in (tqdm(rng) if progress else rng):
        TreeReader.ReadEntry(entry)

        if max_entries is not None and entry > max_entries:
            break

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
                if debug:
                    fj_p4, fj_m = fj_list[best_idx]
                    print(f"  [Debug] MATCH: PID {pid} pt={p.PT:.1f} with FatJet pt={fj_p4.Pt():.1f}, dR={best_dr:.2f}, mSD={fj_m:.1f}")
                used[best_idx] = True
                hists[pid]["matched"].Fill(p.PT, w_gen)
            else:
                if debug:
                    closest_dr = 999
                    closest_info = "None"
                    for fj_idx, (fj_p4, fj_m) in enumerate(fj_list):
                        dR = p.P4().DeltaR(fj_p4)
                        if dR < closest_dr:
                            closest_dr = dR
                            ptRatio = fj_p4.Pt() / p.PT if p.PT > 0 else 0
                            closest_info = f"dR={dR:.2f}, ptRatio={ptRatio:.2f}, mSD={fj_m:.1f}"
                    print(f"  [Debug] NO MATCH: PID {pid} pt={p.PT:.1f}. Closest FatJet: {closest_info}")

    return hists

def split_range(total, n):
    n = max(1, min(n, total))
    base, rem = divmod(total, n)
    out, start = [], 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        out.append((start, start + size))
        start += size
    return out

def merge_outputs(hists_list, out_path=results_file):
    merged = hists_list[0]
    for hists in hists_list[1:]:
        for pid in boson_defs:
            merged[pid]["total"].Add(hists[pid]["total"])
            merged[pid]["matched"].Add(hists[pid]["matched"])
    f_out = ROOT.TFile(out_path, "RECREATE")
    for pid in boson_defs:
        merged[pid]["total"].Write()
        merged[pid]["matched"].Write()
    f_out.Close()
    return merged

def fatjet_efficiency_advanced(inputRootFile, run_parallel=False, n_chunks=None, max_workers=None):
    if not run_parallel:
        return merge_outputs([fatjet_efficiency(inputRootFile)])

    total = count_entries(inputRootFile)
    n_chunks = n_chunks or (max_workers or os.cpu_count())
    chunks = [(inputRootFile, s, e, False)  for s, e in split_range(total, n_chunks)]
    results = parallel_runs(fatjet_efficiency, chunks, max_workers=max_workers, info="Getting fatjet efficiencies", mpContext="fork")
    for r in results:
        if isinstance(r, Exception):
            raise r

    return merge_outputs(results)

def calc_and_plot(hists):
    c = ROOT.TCanvas("c_eff", "FatJet Efficiency", 650, 600)

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
        eff_hist.SetStatisticOption(ROOT.TEfficiency.kFNormal)
        # eff_hist.SetStatisticOption(0) # Clopper-Pearson interval (recommended by PDG)
        eff_hist.SetConfidenceLevel(0.95)
        eff_hist.SetMarkerStyle(20)
        eff_hist.SetMarkerSize(0.8) # Reduced to make error bars visible
        eff_hist.SetLineWidth(2)
        eff_hist.SetFillColor(0)
        eff_hist.SetMarkerColor(props["color"])
        eff_hist.SetLineColor(props["color"])
        eff_hists.append(eff_hist)


        if first:
            eff_hist.SetTitle("Reconstruction efficiency; Truth p_{T} [GeV]; Ak10-Jet Reconstruction Efficiency")
            eff_hist.Draw("AP")
            ROOT.gPad.Update()
            eff_hist.GetPaintedGraph().SetMaximum(1.4)
            eff_hist.GetPaintedGraph().GetXaxis().SetRangeUser(0, 2000)
            eff_hist.GetPaintedGraph().GetXaxis().SetNdivisions(505, True)
            eff_hist.GetPaintedGraph().GetYaxis().SetNdivisions(1010, True)
            first = False
        else:
            eff_hist.Draw("same P")

    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextSize(0.032)
    label.DrawLatex(0.19, 0.88, "#scale[1.1]{#bf{#sqrt{s} = %s TeV, %s fb^{-1}}}" % (13.6, 400))
    label.DrawLatex(0.19, 0.82, "#it{m_{N}} = 1.0 TeV")
    label.DrawLatex(0.19, 0.78, "#DeltaR(J_{tr.}, J_{re.}) #leq 1.0")
    label.DrawLatex(0.19, 0.74, "0.5 #leq p_{T}(J_{re.}) / p_{T}(J_{tr.}) #leq 2.0")
    label.DrawLatex(0.19, 0.70, "Mass Windows: 25 GeV")
    label.DrawLatex(0.19, 0.66, "Normal Approx. @ 95% CL")
    label.DrawLatex(0.19, 0.62, "Considering H #rightarrow b#bar{b} only.")
    

    leg = ROOT.TLegend(0.60, 0.73, 0.93, 0.92)
    leg.SetFillColor(0)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.035)
    for i, pid in enumerate(boson_defs):
        leg.AddEntry(eff_hists[i], f'{boson_defs[pid]["name"]} (#it{{m}} = {boson_defs[pid]["mass"]} GeV)', "lp")
    leg.Draw()

    # c.SetGrid()
    c.Update()
    c.SaveAs("fatjet_reconstruction_efficincy.png")
    c.SaveAs("fatjet_reconstruction_efficincy.pdf")
    c.Close()

def main(inputRootFile, rp=True):
    if os.path.exists(results_file) and not overwrite_results:
        print(f"Loading histograms from {results_file}")
        f_in = ROOT.TFile(results_file)
        hists = {}
        for pid, props in boson_defs.items():
            name = props["name"]
            hists[pid] = {
                "total": f_in.Get(f"h_total_{name}"),
                "matched": f_in.Get(f"h_matched_{name}")
            }
    else:
        hists = fatjet_efficiency_advanced(inputRootFile, run_parallel=rp)

    calc_and_plot(hists)

if __name__ == "__main__":
    main("/data/ammelsayed/Framework/Samples/signal/FDM/Events/sig1000/tag_1_delphes_events.root")
