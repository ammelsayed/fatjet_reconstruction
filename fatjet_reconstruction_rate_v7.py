import os
import array
import ROOT
import numpy as np
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
overwrite_results = False
max_entries = None
results_file = "fatjet_reconstruction_efficiency_results.root"
histograms_file = "fatjet_histograms_results.root"

DeltaRCut = 0.75
apply_fiducial_cuts_at_filling = True

# Groomed (soft-drop) mass window around the parent boson mass. This is applied as a
# *second* stage on top of the DeltaR matching, and is reported separately: ATLAS and
# CMS quote the fatjet reconstruction efficiency both with and without a mass
# requirement (ATLAS arXiv:1906.11005 Fig. 6, CMS arXiv:1612.09159 Fig. 4).
#   CMS arXiv:1410.4227 (CMS-PAS-JME-13-006): 65-85 GeV (W), 85-105 GeV (Z) on m_pruned
#   v5 of this script used the same half-width of 25 GeV around the nominal mass
mass_window_halfwidth = 25.0

# Plot layout. The annotation block lives in the left margin and the legend in the
# right margin, so that neither of them can overlap the data points. The canvas is
# made wider to fit both of them.
CANVAS_W = 1050
CANVAS_H = 680
MARGIN_LEFT = 0.135
MARGIN_RIGHT = 0.035
MARGIN_TOP = 0.045
MARGIN_BOTTOM = 0.125
TEXT_SIZE = 0.022
LEGEND_TEXT_SIZE = 0.024

# The annotation block and the legend are drawn inside the frame, in the empty band
# between the top of the frame and the largest efficiency. The y axis of every plot
# is therefore given a maximum of Y_MAX, so that this band is free of data points.
Y_MAX = 1.45
TEXT_X = MARGIN_LEFT + 0.040
TEXT_Y_TOP = 1.0 - MARGIN_TOP - 0.052
TEXT_DY = 0.036
LEGEND_X1 = 1.0 - MARGIN_RIGHT - 0.46
LEGEND_X2 = 1.0 - MARGIN_RIGHT - 0.02
LEGEND_Y1 = 1.0 - MARGIN_TOP - 0.180
LEGEND_Y2 = 1.0 - MARGIN_TOP - 0.040

# The two selection requirements of the same boson share the boson colour and are
# told apart by the line and marker style only, so that they read as a pair and stay
# distinguishable when the figure is printed in black and white.
LINE_STYLE_PLAIN = 1
LINE_STYLE_MASSWINDOW = 2
MARKER_STYLE_PLAIN = {24: 20, 23: 22, 25: 24}
MARKER_STYLE_MASSWINDOW = {24: 21, 23: 23, 25: 25}

boson_defs = {
    24: {"name": "W", "mass": 80.4, "daughter_pids": {1,2,3,4,5}, "color": 2, "mass_window": (55.4, 105.4)},
    23: {"name": "Z", "mass": 91.2, "daughter_pids": {1,2,3,4,5}, "color": 4, "mass_window": (66.2, 116.2)},
    25: {"name": "H", "mass": 125.0, "daughter_pids": {5}, "color": 3, "mass_window": (100.0, 150.0)}
}

BRANCHES = []
for pid, props in boson_defs.items():
    name = props["name"]
    BRANCHES.extend([
        f"{name}_truth_pt", f"{name}_truth_eta", f"{name}_truth_phi", f"{name}_truth_m",
        f"{name}_reco_pt", f"{name}_reco_eta", f"{name}_reco_phi", f"{name}_reco_m",
        f"{name}_matched",
        f"{name}_m_matched"
    ])

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

def get_fatjets(FatJet_branch):
    ak10jets = []
    for i in range(FatJet_branch.GetEntries()):
        fj = FatJet_branch.At(i)
        if apply_fiducial_cuts_at_filling:
            if not (fj.PT > 200 and abs(fj.Eta) <= 2.5):
                continue
        ak10jets.append((fj.P4(),fj.SoftDroppedP4[0].M()))
    return ak10jets

def draw_label_block(lines):
    """Draw the annotation block inside the frame, below its top edge."""
    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextSize(TEXT_SIZE)
    y = TEXT_Y_TOP
    for text in lines:
        label.DrawLatex(TEXT_X, y, text)
        y -= TEXT_DY

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

    tree = ROOT.TTree("results", "fatjet reconstruction efficiency")
    tree.SetDirectory(0)
    b = {name: np.zeros(1, dtype=np.float64) for name in BRANCHES}
    for name in BRANCHES:
        tree.Branch(name, b[name], f"{name}/D")

    numberOfEntries = TreeReader.GetEntries()
    if end_entry is None or end_entry > numberOfEntries:
        end_entry = numberOfEntries

    rng = range(start_entry, end_entry)
    for entry in (tqdm(rng) if progress else rng):
        TreeReader.ReadEntry(entry)

        if max_entries is not None and entry > max_entries:
            break

        all_gen = []
        for pid, props in boson_defs.items():
            bosons = find_bosons(Particle_branch, pid, props["daughter_pids"])
            for p in bosons:
                if apply_fiducial_cuts_at_filling:
                    if not (p.PT > 200 and abs(p.Eta) <= 2.5):
                        continue
                all_gen.append((pid, p))

        fj_list = get_fatjets(FatJet_branch)
        used = [False] * len(fj_list)

        all_gen.sort(key=lambda x: x[1].PT, reverse=True)
        for pid, p in all_gen:
            name = boson_defs[pid]["name"]

            for branch_name in BRANCHES:
                b[branch_name][0] = 0.0

            b[f"{name}_truth_pt"][0] = p.PT
            b[f"{name}_truth_eta"][0] = p.Eta
            b[f"{name}_truth_phi"][0] = p.Phi
            b[f"{name}_truth_m"][0] = p.Mass

            b[f"{name}_reco_pt"][0] = 0
            b[f"{name}_reco_eta"][0] = 0
            b[f"{name}_reco_phi"][0] = 0
            b[f"{name}_reco_m"][0] = 0
            b[f"{name}_matched"][0] = 0
            b[f"{name}_m_matched"][0] = 0

            best_idx = -1
            best_dr = 1e9
            for fj_idx, (fj_p4, fj_m) in enumerate(fj_list):
                if used[fj_idx]:
                    continue
                dR = p.P4().DeltaR(fj_p4)
                if dR <= DeltaRCut and dR < best_dr:
                    best_dr = dR
                    best_idx = fj_idx

            if best_idx != -1:
                fj_p4, fj_m = fj_list[best_idx]
                b[f"{name}_reco_pt"][0] = fj_p4.Pt()
                b[f"{name}_reco_eta"][0] = fj_p4.Eta()
                b[f"{name}_reco_phi"][0] = fj_p4.Phi()
                b[f"{name}_reco_m"][0] = fj_m
                b[f"{name}_matched"][0] = 1
                used[best_idx] = True

                m_lo, m_hi = boson_defs[pid]["mass_window"]
                if m_lo <= fj_m <= m_hi:
                    b[f"{name}_m_matched"][0] = 1
                if debug:
                    print(f"  [Debug] MATCH: PID {pid} pt={p.PT:.1f} with FatJet pt={fj_p4.Pt():.1f}, dR={best_dr:.2f}, mSD={fj_m:.1f}")
            else:
                if debug:
                    closest_dr = 1e9
                    for fj_idx, (fj_p4, fj_m) in enumerate(fj_list):
                        dR = p.P4().DeltaR(fj_p4)
                        if dR < closest_dr:
                            closest_dr = dR
                    print(f"  [Debug] NO MATCH: PID {pid} pt={p.PT:.1f}. Closest FatJet dR={closest_dr:.2f}")

            tree.Fill()

    return tree

def split_range(total, n):
    n = max(1, min(n, total))
    base, rem = divmod(total, n)
    out, start = [], 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        out.append((start, start + size))
        start += size
    return out

def merge_outputs(trees, out_path=results_file):
    tl = ROOT.TList()
    for t in trees:
        tl.Add(t)
    f_out = ROOT.TFile(out_path, "RECREATE")
    merged = ROOT.TTree.MergeTrees(tl)
    merged.SetName("results")
    merged.SetTitle("fatjet reconstruction efficiency")
    merged.Write("results", ROOT.TObject.kOverwrite)
    f_out.Close()
    f_in = ROOT.TFile.Open(out_path)
    tree = f_in.Get("results")
    tree._file = f_in
    print(f"Saved merged tree 'results' -> {out_path} ({tree.GetEntries()} entries)")
    return tree

def fatjet_efficiency_advanced(inputRootFile, run_parallel=False, n_chunks=None, max_workers=None):
    if not run_parallel:
        return merge_outputs([fatjet_efficiency(inputRootFile)])

    total = count_entries(inputRootFile)
    n_chunks = n_chunks or (max_workers or os.cpu_count())
    chunks = [(inputRootFile, s, e, False) for s, e in split_range(total, n_chunks)]
    results = parallel_runs(fatjet_efficiency, chunks, max_workers=max_workers, info="Getting fatjet efficiencies", mpContext="fork")
    for r in results:
        if isinstance(r, Exception):
            raise r

    return merge_outputs(results)


PLOT_SETTINGS = {
    "pt": {
        "x_title": "Truth p_{T} [GeV]",
        "x_range": (150, 1150),
        "x_divisions": 505,
        "draw_option": "APL",
        "same_option": "same PL",
        "legend_option": "lp",
        "canvas_title": "FatJet Efficiency vs pT",
        "output": "./plots/fatjet_reconstruction_efficiency_vs_pt",
    },
    "eta": {
        "x_title": "Truth #eta",
        "x_range": (-2.5, 2.5),
        "x_divisions": 1010,
        "draw_option": "AP",
        "same_option": "same P",
        "legend_option": "p",
        "canvas_title": "FatJet Efficiency vs eta",
        "output": "./plots/fatjet_reconstruction_efficiency_vs_eta",
    },
}

def make_efficiency(h_mat, h_tot, pid, with_mass_window):
    """Efficiency of the reconstruction of one boson, with or without mass window."""
    props = boson_defs[pid]
    eff = ROOT.TEfficiency(h_mat, h_tot)
    eff.SetStatisticOption(ROOT.TEfficiency.kFCP)
    eff.SetConfidenceLevel(0.95)
    eff.SetMarkerStyle(MARKER_STYLE_MASSWINDOW[pid] if with_mass_window else MARKER_STYLE_PLAIN[pid])
    eff.SetMarkerSize(1.0)
    eff.SetLineWidth(2)
    eff.SetLineStyle(LINE_STYLE_MASSWINDOW if with_mass_window else LINE_STYLE_PLAIN)
    eff.SetFillColor(0)
    eff.SetMarkerColor(props["color"])
    eff.SetLineColor(props["color"])
    return eff

def print_efficiency_table(all_hists, kind, bins):
    """Print the efficiency of every boson in the given binning."""
    for pid, props in boson_defs.items():
        name = props["name"]
        h_tot = all_hists[pid]["total_" + kind]
        h_mat = all_hists[pid]["matched_" + kind]
        h_mat_mw = all_hists[pid]["matched_" + kind + "_mw"]
        total = h_tot.Integral()
        matched = h_mat.Integral()
        matched_mw = h_mat_mw.Integral()
        print(f"\nEfficiency vs {kind} for {name}:")
        for i in range(h_tot.GetNbinsX()):
            n_tot = h_tot.GetBinContent(i + 1)
            n_mat = h_mat.GetBinContent(i + 1)
            n_mw = h_mat_mw.GetBinContent(i + 1)
            if n_tot == 0:
                continue
            eff = n_mat / n_tot * 100
            eff_mw = n_mw / n_tot * 100
            print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f})   N={n_tot:.0f}  matched={n_mat:.0f}  eff={eff:.2f}%  matched_mw={n_mw:.0f}  eff_mw={eff_mw:.2f}%")
        if total > 0:
            print(f"Total: {total:.1f} {name} bosons, {matched:.1f} matched ({matched/total*100:.2f}%), {matched_mw:.1f} matched within mass window ({matched_mw/total*100:.2f}%)")
        else:
            print(f"No {name} bosons found.")

def plot_efficiency(all_hists, pids, kind, output=None, canvas_name=None, n_columns=2):
    """Draw the reconstruction efficiency of the given bosons and save the figure.

    Every boson is drawn twice: on its own and with the mass window of the channel
    applied. The two are told apart by the line and marker style, and they share
    the colour of the boson.
    """
    settings = PLOT_SETTINGS[kind]
    if output is None:
        output = settings["output"]
    if canvas_name is None:
        canvas_name = "c_eff_" + kind

    canvas = ROOT.TCanvas(canvas_name, settings["canvas_title"], CANVAS_W, CANVAS_H)
    canvas.SetLeftMargin(MARGIN_LEFT)
    canvas.SetRightMargin(MARGIN_RIGHT)
    canvas.SetTopMargin(MARGIN_TOP)
    canvas.SetBottomMargin(MARGIN_BOTTOM)
    eff_hists = []
    eff_hists_mw = []
    first = True
    for pid in pids:
        h_tot = all_hists[pid]["total_" + kind]
        h_mat = all_hists[pid]["matched_" + kind]
        h_mat_mw = all_hists[pid]["matched_" + kind + "_mw"]

        eff_hist = make_efficiency(h_mat, h_tot, pid, False)
        eff_hists.append(eff_hist)
        eff_hist_mw = make_efficiency(h_mat_mw, h_tot, pid, True)
        eff_hists_mw.append(eff_hist_mw)

        if first:
            eff_hist.SetTitle("Reconstruction efficiency; " + settings["x_title"] + "; #epsilon_{reco.} (%)")
            eff_hist.Draw(settings["draw_option"])
            ROOT.gPad.Update()
            eff_hist.GetPaintedGraph().SetMinimum(0.0)
            eff_hist.GetPaintedGraph().SetMaximum(Y_MAX)
            eff_hist.GetPaintedGraph().GetXaxis().SetRangeUser(*settings["x_range"])
            eff_hist.GetPaintedGraph().GetXaxis().SetNdivisions(settings["x_divisions"], True)
            eff_hist.GetPaintedGraph().GetYaxis().SetNdivisions(1010, True)
            eff_hist_mw.Draw(settings["same_option"])
            ROOT.gPad.Update()
            first = False
        else:
            eff_hist.Draw(settings["same_option"])
            ROOT.gPad.Update()
            eff_hist_mw.Draw(settings["same_option"])
            ROOT.gPad.Update()
        ROOT.gPad.Update()

    draw_label_block([
        "#scale[1.1]{#bf{#sqrt{s} = %s TeV, %s fb^{-1}}}" % (13.6, 400),
        "Ak10-Jet Reco. Efficiency (%)",
        "#it{m_{N}} = 1.0 TeV",
        "#DeltaR(J_{truth}, J_{reco.}) #leq 0.75",
        "Considering H #rightarrow b#bar{b} only.",
        "Clopper-Pearson @ 95% CL",
    ])

    legend_width = 0.46 if len(pids) > 1 else 0.26
    leg = ROOT.TLegend(1.0 - MARGIN_RIGHT - legend_width, LEGEND_Y1, 1.0 - MARGIN_RIGHT - 0.02, LEGEND_Y2)
    leg.SetFillColor(0)
    leg.SetBorderSize(0)
    leg.SetTextSize(LEGEND_TEXT_SIZE)
    leg.SetNColumns(n_columns)
    for i, pid in enumerate(pids):
        name = boson_defs[pid]["name"]
        leg.AddEntry(eff_hists[i], f'{name} (#it{{m}} = {boson_defs[pid]["mass"]} GeV)', settings["legend_option"])
        leg.AddEntry(eff_hists_mw[i], f'{name} + #it{{m_{{SD}}}} window', settings["legend_option"])
    leg.Draw()

    canvas.Update()
    canvas.SaveAs(output + ".png")
    canvas.SaveAs(output + ".pdf")
    canvas.Close()

def calc_and_plot(tree, verbose=False):

    ROOT.gStyle.SetEndErrorSize(2)

    os.makedirs("./plots", exist_ok=True)
    pt_bin_width = 20
    pt_bins = list(range(200, 1000 + pt_bin_width, pt_bin_width))
    n_pt_bins = len(pt_bins) - 1
    pt_bins_arr = array.array('d', pt_bins)

    n_eta_bins = 50
    eta_lo, eta_hi = -2.5, 2.5
    eta_bins = [eta_lo + i * (eta_hi - eta_lo) / n_eta_bins for i in range(n_eta_bins + 1)]
    eta_bins_arr = array.array('d', eta_bins)

    all_hists = {}
    for pid, props in boson_defs.items():
        name = props["name"]
        all_hists[pid] = {
            "total_pt": ROOT.TH1D(f"h_total_pt_{name}", "", n_pt_bins, pt_bins_arr),
            "matched_pt": ROOT.TH1D(f"h_matched_pt_{name}", "", n_pt_bins, pt_bins_arr),
            "matched_pt_mw": ROOT.TH1D(f"h_matched_pt_mw_{name}", "", n_pt_bins, pt_bins_arr),
            "total_eta": ROOT.TH1D(f"h_total_eta_{name}", "", n_eta_bins, eta_bins_arr),
            "matched_eta": ROOT.TH1D(f"h_matched_eta_{name}", "", n_eta_bins, eta_bins_arr),
            "matched_eta_mw": ROOT.TH1D(f"h_matched_eta_mw_{name}", "", n_eta_bins, eta_bins_arr),
            "pt_response": ROOT.TH2D(f"h_pt_resp_{name}", "", n_pt_bins, pt_bins_arr, 100, 0, 2),
            "mass_response": ROOT.TH2D(f"h_mass_resp_{name}", "", 100, 0, 200, 100, 0, 200),
        }
        tree.Project(f"h_total_pt_{name}", f"{name}_truth_pt", f"{name}_truth_pt > 0")
        tree.Project(f"h_matched_pt_{name}", f"{name}_truth_pt", f"{name}_matched")
        tree.Project(f"h_matched_pt_mw_{name}", f"{name}_truth_pt", f"{name}_m_matched")
        tree.Project(f"h_total_eta_{name}", f"{name}_truth_eta", f"{name}_truth_pt > 0")
        tree.Project(f"h_matched_eta_{name}", f"{name}_truth_eta", f"{name}_matched")
        tree.Project(f"h_matched_eta_mw_{name}", f"{name}_truth_eta", f"{name}_m_matched")
        tree.Project(f"h_pt_resp_{name}", f"{name}_reco_pt/{name}_truth_pt:{name}_truth_pt", f"{name}_matched")
        tree.Project(f"h_mass_resp_{name}", f"{name}_reco_m:{name}_truth_m", f"{name}_matched")

    if verbose:
        print_efficiency_table(all_hists, "pt", pt_bins)
        print_efficiency_table(all_hists, "eta", eta_bins)

    # all the bosons on one figure, and then one figure per boson
    plot_efficiency(all_hists, list(boson_defs), "pt")
    plot_efficiency(all_hists, list(boson_defs), "eta")
    for pid, props in boson_defs.items():
        name = props["name"]
        plot_efficiency(all_hists, [pid], "pt",
                        output="./plots/fatjet_reconstruction_efficiency_vs_pt_" + name,
                        canvas_name="c_eff_pt_" + name, n_columns=1)
        plot_efficiency(all_hists, [pid], "eta",
                        output="./plots/fatjet_reconstruction_efficiency_vs_eta_" + name,
                        canvas_name="c_eff_eta_" + name, n_columns=1)

    # set titles on response hists and save all to file
    for pid, props in boson_defs.items():
        name = props["name"]
        all_hists[pid]["pt_response"].GetXaxis().SetTitle("Truth p_{T} [GeV]")
        all_hists[pid]["pt_response"].GetYaxis().SetTitle("Reco p_{T} / Truth p_{T}")
        all_hists[pid]["mass_response"].GetXaxis().SetTitle("Truth mass [GeV]")
        all_hists[pid]["mass_response"].GetYaxis().SetTitle("Reco mass (SoftDrop) [GeV]")

    f_out = ROOT.TFile(histograms_file, "RECREATE")
    for pid in all_hists:
        for h in all_hists[pid].values():
            h.Write()
    f_out.Close()
    print(f"Saved histograms -> {histograms_file}")

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute fatjet reconstruction efficiency from Delphes ROOT files."
    )
    parser.add_argument(
        "inputRootFiles",
        nargs="+",
        help="One or more Delphes ROOT files (space-separated), e.g. f1.root f2.root f3.root",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing over entry ranges.",
    )
    args = parser.parse_args()

    inputRootFiles = args.inputRootFiles
    rp = not args.no_parallel

    if not overwrite_results and os.path.exists(results_file):
        f_in = ROOT.TFile.Open(results_file)
        tree = f_in.Get("results")
        tree._file = f_in
    else:
        tree = fatjet_efficiency_advanced(inputRootFiles, run_parallel=rp)
    calc_and_plot(tree)

if __name__ == "__main__":
    main()