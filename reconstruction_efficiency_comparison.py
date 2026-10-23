import os
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gROOT.SetStyle("ATLAS")
print("Using ROOT version:", ROOT.__version__)

fatjet_histograms_file = "fatjet_histograms_results.root"
dijet_histograms_file = "dijet_histograms_results.root"

BOSONS = ["W", "Z", "H"]

def suppress_x_errors(eff_hists):
    """Drop the horizontal (bin width) error bars, keeping the vertical ones.

    ROOT draws them by default, which turns a binned efficiency plot into a
    staircase of overlapping bars. Every draw of a TEfficiency puts the bin width
    back, and TLegend draws the object again for each legend entry, so this has
    to be done once all the curves and the legend are on the canvas.
    """
    for eff_hist in eff_hists:
        graph = eff_hist.GetPaintedGraph()
        if not graph:
            continue
        ex_low = graph.GetEXlow()
        ex_high = graph.GetEXhigh()
        if not ex_low or not ex_high:
            continue
        for i in range(graph.GetN()):
            ex_low[i] = 0.0
            ex_high[i] = 0.0

def draw_label_block(lines):
    """Draw the annotation block inside the frame, below its top edge."""
    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextSize(0.022)
    y = 0.903
    for text in lines:
        label.DrawLatex(0.175, y, text)
        y -= 0.036

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
    eff.SetMarkerSize(1.0)
    eff.SetLineWidth(2)
    eff.SetLineStyle(2 if with_mass_window else 1)
    eff.SetFillColor(0)
    eff.SetMarkerColor(color)
    eff.SetLineColor(color)
    return eff

def plot_efficiency(name, fatjet_hists, dijet_hists):
    """Draw the FatJet and dijet reconstruction efficiency of one boson (each
    with and without the mass window) on one canvas and save the figure."""
    canvas = ROOT.TCanvas("c_eff_pt_" + name, "FatJet vs Dijet Efficiency vs pT", 1050, 680)
    canvas.SetLeftMargin(0.135)
    canvas.SetRightMargin(0.035)
    canvas.SetTopMargin(0.045)
    canvas.SetBottomMargin(0.125)

    curricula = []
    for hists, color in [[fatjet_hists, ROOT.kBlack], [dijet_hists, 2]]:
        curricula.append(make_efficiency(hists["matched_pt"], hists["total_pt"], color, False))
        curricula.append(make_efficiency(hists["matched_pt_mw"], hists["total_pt"], color, True))

    curricula[0].SetTitle("Reconstruction efficiency; Truth p_{T} [GeV]; #epsilon_{reco.} (%)")
    curricula[0].Draw("APL")
    ROOT.gPad.Update()
    graph = curricula[0].GetPaintedGraph()
    graph.SetMinimum(0.0)
    graph.SetMaximum(1.45)
    graph.GetXaxis().SetRangeUser(150, 1150)
    graph.GetXaxis().SetNdivisions(505, True)
    graph.GetYaxis().SetNdivisions(1010, True)
    for eff in curricula[1:]:
        eff.Draw("same PL")
        ROOT.gPad.Update()

    draw_label_block([
        "#scale[1.1]{#bf{#sqrt{s} = %s TeV, %s fb^{-1}}}" % (13.6, 400),
        "FatJet vs Dijet Reco. Efficiency (%)",
        "#it{m_{N}} = 1.0 TeV",
        "FatJet: #DeltaR(J_{truth}, J_{reco.}) #leq 0.75",
        "Dijet: #DeltaR(q, j) #leq 0.4",
        "Considering H #rightarrow b#bar{b} only. Clopper-Pearson @ 95% CL",
    ])

    leg = ROOT.TLegend(0.505, 0.775, 0.945, 0.915)
    leg.SetFillColor(0)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.024)
    leg.SetNColumns(2)
    leg.AddEntry(curricula[0], "FatJet (angular match)", "lp")
    leg.AddEntry(curricula[1], "FatJet + #it{m_{SD}} window", "lp")
    leg.AddEntry(curricula[2], "Dijet (angular match)", "lp")
    leg.AddEntry(curricula[3], "Dijet + #it{m_{jj}} window", "lp")
    leg.Draw()

    suppress_x_errors(curricula)

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
