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
results_file = "dijet_reconstruction_efficiency_results.root"

DeltaRCut = 0.75
apply_fiducial_cuts_at_filling = True

# Truth parton -> reconstructed jet matching in the eta-phi plane. DeltaR = 0.4 is the
# ATLAS/CMS standard for parton-to-jet matching (equal to the anti-kt R = 0.4 radius):
#   ATLAS: arXiv:2308.00716, arXiv:1809.08597, ATLFAST 2.0 (cds.cern.ch/record/683751)
#   CMS:   CMS-PAS-JME-16-003 (R_min = 0.4), arXiv:1912.06046
DeltaRQuarkJet = 0.4

# Reco-side (AK4) jet fiducial cuts, as used by ATLAS for a resolved V->qq dijet:
#   ATLAS arXiv:1706.01702: pT > 25 GeV, |eta| < 2.5
JetPTCut = 25.0
JetEtaCut = 2.5

# Dijet invariant mass window around the parent boson mass. This is applied as a
# *second* stage on top of the parton-level matching, and is reported separately:
# ATLAS/CMS quote the dijet reconstruction efficiency both with and without it
# (ATLAS arXiv:1906.11005 Fig. 6, ATLFAST 2.0 Tables 3-1/3-2).
#   ATLAS arXiv:1905.07714 (STDM-2017-20): 64 < m_jj < 106 GeV, resolved W/Z -> qq
#   ATLAS arXiv:1712.06518 (JHEP 03 (2018) 174): 110 < m_bb < 140 GeV, H -> bb
boson_defs = {
    24: {"name": "W", "mass": 80.4, "daughter_pids": {1,2,3,4,5}, "color": 2, "mass_window": (64.0, 106.0)},
    23: {"name": "Z", "mass": 91.2, "daughter_pids": {1,2,3,4,5}, "color": 4, "mass_window": (64.0, 106.0)},
    25: {"name": "H", "mass": 125.0, "daughter_pids": {5}, "color": 3, "mass_window": (110.0, 140.0)}
}


BRANCHES = []
for pid, props in boson_defs.items():
    name = props["name"]
    BRANCHES.extend([
        f"{name}_truth_pt", f"{name}_truth_eta", f"{name}_truth_phi", f"{name}_truth_m",
        f"{name}_reco_pt", f"{name}_reco_eta", f"{name}_reco_phi", f"{name}_reco_m",
        f"{name}_matched",
        f"{name}_m_matched",
        f"{name}_truth_dR"
    ])

def draw_label_block(lines):
    """Draw the annotation block inside the frame, below its top edge."""
    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextSize(0.022)
    y = 0.903
    for text in lines:
        label.DrawLatex(0.175, y, text)
        y -= 0.036
        
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

def get_jets(Jet_branch):
    ak4jets = []
    for i in range(Jet_branch.GetEntries()):
        j = Jet_branch.At(i)
        if apply_fiducial_cuts_at_filling:
            if not (j.PT > JetPTCut and abs(j.Eta) <= JetEtaCut):
                continue
        ak4jets.append((j.P4(), j.Mass))
    return ak4jets

def get_daughter_quarks(Particle_branch, boson):
    d1, d2 = int(boson.D1), int(boson.D2)
    if d1 < 0 or d2 < 0:
        return []
    return [Particle_branch.At(d_idx) for d_idx in range(d1, d2 + 1)]

def match_dijet(Particle_branch, boson, jets, used, dR_quark_jet=DeltaRQuarkJet):
    """Match the two truth decay quarks of the boson onto two distinct reco jets.

    Closest match in the eta-phi plane, one-to-one, as done by ATLAS and CMS
    (DeltaR = 0.4 for parton-to-jet matching, e.g. arXiv:2308.00716,
    CMS-PAS-JME-16-003). The jet pair is the one minimising the largest quark-jet
    DeltaR over the two possible quark assignments, so that a merged (single-jet)
    topology fails the two-jet requirement instead of assigning both quarks to the
    same jet.
    """
    quarks = get_daughter_quarks(Particle_branch, boson)
    if len(quarks) != 2:
        return None

    avail = [j_idx for j_idx in range(len(jets)) if not used[j_idx]]
    if len(avail) < 2:
        return None

    q1_p4 = quarks[0].P4()
    q2_p4 = quarks[1].P4()
    d_q1 = [q1_p4.DeltaR(jets[j_idx][0]) for j_idx in avail]
    d_q2 = [q2_p4.DeltaR(jets[j_idx][0]) for j_idx in avail]

    best_dr = None
    best_pair = None
    for a in range(len(avail)):
        for b in range(len(avail)):
            if a == b:
                continue
            dr = max(d_q1[a], d_q2[b])
            if best_dr is None or dr < best_dr:
                best_dr = dr
                best_pair = (avail[a], avail[b])
            dr = max(d_q2[a], d_q1[b])
            if best_dr is None or dr < best_dr:
                best_dr = dr
                best_pair = (avail[b], avail[a])

    if best_dr is None or best_dr > dR_quark_jet:
        return None
    return best_pair

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

def dijet_efficiency(inputRootFile, start_entry=0, end_entry=None, progress=True):

    if progress:
        print("Reading MC events from the root file:")
        print(f" > {inputRootFile}")

    Chain = build_chain(inputRootFile)
    TreeReader = ROOT.ExRootTreeReader(Chain)
    Jet_branch = TreeReader.UseBranch("Jet")
    Particle_branch = TreeReader.UseBranch("Particle")

    tree = ROOT.TTree("results", "dijet reconstruction efficiency")
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

        jet_list = get_jets(Jet_branch)
        used = [False] * len(jet_list)

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

            quarks = get_daughter_quarks(Particle_branch, p)
            if len(quarks) == 2:
                b[f"{name}_truth_dR"][0] = quarks[0].P4().DeltaR(quarks[1].P4())

            matched_jets = match_dijet(Particle_branch, p, jet_list, used)

            if matched_jets is not None:
                j1_idx, j2_idx = matched_jets
                j1_p4, j1_m = jet_list[j1_idx]
                j2_p4, j2_m = jet_list[j2_idx]
                dijet_p4 = j1_p4 + j2_p4
                b[f"{name}_reco_pt"][0] = dijet_p4.Pt()
                b[f"{name}_reco_eta"][0] = dijet_p4.Eta()
                b[f"{name}_reco_phi"][0] = dijet_p4.Phi()
                b[f"{name}_reco_m"][0] = dijet_p4.M()
                b[f"{name}_matched"][0] = 1
                used[j1_idx] = True
                used[j2_idx] = True

                m_lo, m_hi = boson_defs[pid]["mass_window"]
                if m_lo <= dijet_p4.M() <= m_hi:
                    b[f"{name}_m_matched"][0] = 1
                if debug:
                    print(f"  [Debug] MATCH: PID {pid} pt={p.PT:.1f} -> dijet pt={dijet_p4.Pt():.1f}, mjj={dijet_p4.M():.1f}, dRqq={b[f'{name}_truth_dR'][0]:.2f}")
            else:
                if debug:
                    print(f"  [Debug] NO MATCH: PID {pid} pt={p.PT:.1f} (quark-jet DeltaR cut {DeltaRQuarkJet})")

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
    merged.SetTitle("dijet reconstruction efficiency")
    merged.Write("results", ROOT.TObject.kOverwrite)
    f_out.Close()
    f_in = ROOT.TFile.Open(out_path)
    tree = f_in.Get("results")
    tree._file = f_in
    print(f"Saved merged tree 'results' -> {out_path} ({tree.GetEntries()} entries)")
    return tree

def dijet_efficiency_advanced(inputRootFile, run_parallel=False, n_chunks=None, max_workers=None):
    if not run_parallel:
        return merge_outputs([dijet_efficiency(inputRootFile)])

    total = count_entries(inputRootFile)
    n_chunks = n_chunks or (max_workers or os.cpu_count())
    chunks = [(inputRootFile, s, e, False) for s, e in split_range(total, n_chunks)]
    results = parallel_runs(dijet_efficiency, chunks, max_workers=max_workers, info="Getting dijet efficiencies", mpContext="fork")
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
        "canvas_title": "Dijet Efficiency vs pT",
        "output": "./plots/dijet_reconstruction_efficiency_vs_pt",
    },
    "eta": {
        "x_title": "Truth #eta",
        "x_range": (-2.5, 2.5),
        "x_divisions": 1010,
        "draw_option": "AP",
        "same_option": "same P",
        "legend_option": "p",
        "canvas_title": "Dijet Efficiency vs eta",
        "output": "./plots/dijet_reconstruction_efficiency_vs_eta",
    },
}

def make_efficiency(h_mat, h_tot, pid, with_mass_window):
    """Efficiency of the reconstruction of one boson, with or without mass window."""
    props = boson_defs[pid]
    eff = ROOT.TEfficiency(h_mat, h_tot)
    eff.SetStatisticOption(ROOT.TEfficiency.kFCP)
    eff.SetConfidenceLevel(0.95)
    eff.SetMarkerStyle({24: 21, 23: 23, 25: 25}[pid] if with_mass_window else {24: 20, 23: 22, 25: 24}[pid])
    eff.SetMarkerSize(1.0)
    eff.SetLineWidth(2)
    eff.SetLineStyle(2 if with_mass_window else 1)
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

    canvas = ROOT.TCanvas(canvas_name, settings["canvas_title"], 1050, 680)
    canvas.SetLeftMargin(0.135)
    canvas.SetRightMargin(0.035)
    canvas.SetTopMargin(0.045)
    canvas.SetBottomMargin(0.125)
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
            eff_hist.GetPaintedGraph().SetMaximum(1.45)
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
        "Ak4-Dijet Reco. Efficiency (%)",
        "#it{m_{N}} = 1.0 TeV",
        "#DeltaR(q, j) #leq 0.4",
        "Considering H #rightarrow b#bar{b} only.",
        "Clopper-Pearson @ 95% CL",
    ])

    legend_width = 0.46 if len(pids) > 1 else 0.26
    leg = ROOT.TLegend(1.0 - 0.035 - legend_width, 0.775, 0.945, 0.915)
    leg.SetFillColor(0)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.024)
    leg.SetNColumns(n_columns)
    for i, pid in enumerate(pids):
        name = boson_defs[pid]["name"]
        leg.AddEntry(eff_hists[i], f'{name} (#it{{m}} = {boson_defs[pid]["mass"]} GeV)', settings["legend_option"])
        leg.AddEntry(eff_hists_mw[i], f'{name} + #it{{m_{{jj}}}} window', settings["legend_option"])
    leg.Draw()

    canvas.Update()
    canvas.SaveAs(output + ".png")
    canvas.SaveAs(output + ".pdf")
    canvas.Close()

def build_hists(tree, kind, pids):
    """Bin the raw results tree for the given bosons. The binning is decided
    here, in the drawing stage, and is never written to disk: change it freely,
    the event loop never has to be re-run."""
    if kind == "pt":
        pt_bin_width = 20
        bins = list(range(200, 1000 + pt_bin_width, pt_bin_width))
    else:
        n_bins = 50
        eta_lo, eta_hi = -2.5, 2.5
        bins = [eta_lo + i * (eta_hi - eta_lo) / n_bins for i in range(n_bins + 1)]
    bins_arr = array.array('d', bins)

    all_hists = {}
    for pid in pids:
        name = boson_defs[pid]["name"]
        n_bins = len(bins) - 1
        all_hists[pid] = {
            "total_" + kind: ROOT.TH1D(f"h_total_{kind}_{name}", "", n_bins, bins_arr),
            "matched_" + kind: ROOT.TH1D(f"h_matched_{kind}_{name}", "", n_bins, bins_arr),
            "matched_" + kind + "_mw": ROOT.TH1D(f"h_matched_{kind}_mw_{name}", "", n_bins, bins_arr),
        }
        tree.Project(f"h_total_{kind}_{name}", f"{name}_truth_{kind}", f"{name}_truth_pt > 0")
        tree.Project(f"h_matched_{kind}_{name}", f"{name}_truth_{kind}", f"{name}_matched")
        tree.Project(f"h_matched_{kind}_mw_{name}", f"{name}_truth_{kind}", f"{name}_m_matched")
    return all_hists, bins


def calc_and_plot(tree, verbose=False):

    ROOT.gStyle.SetEndErrorSize(2)

    os.makedirs("./plots", exist_ok=True)

    for kind in ["pt", "eta"]:
        all_hists, bins = build_hists(tree, kind, list(boson_defs))

        if verbose:
            print_efficiency_table(all_hists, kind, bins)

        # all the bosons on one figure, and then one figure per boson
        plot_efficiency(all_hists, list(boson_defs), kind)
        for pid, props in boson_defs.items():
            name = props["name"]
            plot_efficiency(all_hists, [pid], kind,
                            output=f"./plots/dijet_reconstruction_efficiency_vs_{kind}_" + name,
                            canvas_name=f"c_eff_{kind}_" + name, n_columns=1)

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute dijet reconstruction efficiency from Delphes ROOT files."
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
        tree = dijet_efficiency_advanced(inputRootFiles, run_parallel=rp)
    calc_and_plot(tree)

if __name__ == "__main__":
    main()