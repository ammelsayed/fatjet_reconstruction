import os
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gROOT.SetStyle("ATLAS")
print("Using ROOT version:", ROOT.__version__)

fatjet_histograms_file = "fatjet_histograms_results.root"
dijet_histograms_file = "dijet_histograms_results.root"

BOSONS = ["W", "Z", "H"]

def draw_label_block(lines):
    """Draw the annotation block inside the frame, below its top edge."""
    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextSize(0.028)
    y = 0.88
    for text in lines:
        label.DrawLatex(0.2, y, text)
        y -= 0.04

def load_histograms(path):
    """Read the efficiency input histograms of one reconstruction method.

    Both the fatjet and the dijet script save h_total_pt_{name}, h_matched_pt_
    {name} and h_matched_pt_mw_{name} with the same binning, so the curves can
    be built straight from the saved files without re-running either script.
    """
    f_in = ROOT.TFile.Open(path)
    histograms = {name: {} for name in BOSONS}
    for name in BOSONS:
        for stage in ["total_pt", "matched_pt", "matched_pt_mw"]:
            h = f_in.Get("h_" + stage + "_" + name)
            h.GetXaxis().SetTitle(f"Truth p_{{T}}^{{#it{name}}} [GeV]")
            h.SetDirectory(0)
            histograms[name][stage] = h
    f_in.Close()
    return histograms

def make_efficiency(h_mat, h_tot, color, with_mass_window):
    """Efficiency curve of one reconstruction, with or without mass window."""
    eff = ROOT.TEfficiency(h_mat, h_tot)
    eff.SetStatisticOption(ROOT.TEfficiency.kFCP)
    eff.SetConfidenceLevel(0.95)
    eff.SetMarkerStyle(21 if with_mass_window else 20)
    eff.SetMarkerSize(0.8)
    eff.SetLineWidth(2)
    # eff.SetLineStyle(2 if with_mass_window else 1)
    eff.SetFillColor(0)
    eff.SetMarkerColorAlpha(color, 0.5 if with_mass_window else 1.0)
    eff.SetLineColorAlpha(color, 0.5 if with_mass_window else 1.0)
    return eff

def plot_efficiency(name, fatjet_hists, dijet_hists):
    """Draw the FatJet and dijet reconstruction efficiency of one boson (each
    with and without the mass window) on one canvas and save the figure."""
    canvas = ROOT.TCanvas("c_eff_pt_" + name, "FatJet vs Dijet Efficiency vs pT", 800, 600)

    curricula = []
    for hists, color in [[fatjet_hists, ROOT.kBlack], [dijet_hists, 2]]:
        curricula.append(make_efficiency(hists["matched_pt"], hists["total_pt"], color, False))
        curricula.append(make_efficiency(hists["matched_pt_mw"], hists["total_pt"], color, True))

    curricula[0].SetTitle("Reconstruction efficiency; Truth p_{T} [GeV]; #epsilon_{reco.} (%)")
    curricula[0].Draw("APL")
    ROOT.gPad.Update()
    graph = curricula[0].GetPaintedGraph()
    graph.SetMinimum(0.0)
    graph.SetMaximum(1.65)
    graph.GetXaxis().SetRangeUser(150, 1150)
    graph.GetXaxis().SetNdivisions(505, True)
    graph.GetYaxis().SetNdivisions(1010, True)
    for eff in curricula[1:]:
        eff.Draw("same PL")
        ROOT.gPad.Update()

    draw_label_block([
        "#scale[1.1]{#bf{#sqrt{s} = %s TeV, %s fb^{-1}}}" % (13.6, 500),
        "FatJet vs Dijet Reco. Efficiency (%)",
        # "#it{m_{N}} = 1.0 TeV",
        "FatJet: #DeltaR(J_{truth}, J_{reco.}) #leq 0.75",
        "Dijet: #DeltaR(q, j) #leq 0.4",
        # "Considering H #rightarrow b#bar{b} only.",
        "Clopper-Pearson @ 95% CL"
    ])

    leg = ROOT.TLegend(0.6, 0.7, 0.88, 0.92)
    leg.SetFillColor(0)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.03)
    leg.AddEntry(curricula[0], "FatJet (angular match)", "lp")
    leg.AddEntry(curricula[1], "FatJet + #it{m_{SD}} window", "lp")
    leg.AddEntry(curricula[2], "Dijet (angular match)", "lp")
    leg.AddEntry(curricula[3], "Dijet + #it{m_{jj}} window", "lp")
    leg.Draw()

    output = "./plots/reconstruction_efficiency_comparison_vs_pt_" + name
    canvas.Update()
    canvas.SaveAs(output + ".png")
    canvas.SaveAs(output + ".pdf")
    canvas.Close()
    return output

def main():
    ROOT.gStyle.SetEndErrorSize(2)

    os.makedirs("./plots", exist_ok=True)
    fatjet_hists = load_histograms(fatjet_histograms_file)
    dijet_hists = load_histograms(dijet_histograms_file)

    for name in BOSONS:
        output = plot_efficiency(name, fatjet_hists[name], dijet_hists[name])
        print(f"Saved {output}.png/.pdf")

if __name__ == "__main__":
    main()
