import os
import array
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gROOT.SetStyle("ATLAS")
print("Using ROOT version:", ROOT.__version__)

fatjet_results_file = "fatjet_reconstruction_efficiency_results.root"
dijet_results_file = "dijet_reconstruction_efficiency_results.root"
correlations_results_file = "/data/ammelsayed/mother_decay_correlations/truth_decay_correlations_results_v3.root"

BOSONS = ["W", "Z", "H"]

def draw_label_block(lines, text_size=0.028, x=0.2, step = 0.04):
    """Draw the annotation block inside the frame, below its top edge. A
    text_size >= 1 is taken in pixels (font precision 3), which keeps the same
    absolute size across pads of different heights."""
    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextSize(text_size)
    y = 0.88
    for text in lines:
        label.DrawLatex(x, y, text)
        y -= step

def load_tree(path):
    """Read the raw results tree written by one of the reconstruction scripts."""
    f_in = ROOT.TFile.Open(path)
    tree = f_in.Get("results")
    tree._file = f_in
    return tree

def build_hists(tree, name, tag):
    """Bin the raw results of one boson for one reconstruction method. The
    binning is decided here, in the drawing stage, and is never written to
    disk: change it freely, the efficiency scripts never have to be re-run."""
    pt_bin_width = 10
    bins = list(range(0, 1200 + pt_bin_width, pt_bin_width))
    n_bins = len(bins) - 1
    bins_arr = array.array('d', bins)

    hists = {
        "total_pt": ROOT.TH1D(f"h_total_pt_{tag}_{name}", "", n_bins, bins_arr),
        "matched_pt": ROOT.TH1D(f"h_matched_pt_{tag}_{name}", "", n_bins, bins_arr),
        "matched_pt_mw": ROOT.TH1D(f"h_matched_pt_mw_{tag}_{name}", "", n_bins, bins_arr),
    }
    tree.Project(f"h_total_pt_{tag}_{name}", f"{name}_truth_pt", f"{name}_truth_pt > 0")
    tree.Project(f"h_matched_pt_{tag}_{name}", f"{name}_truth_pt", f"{name}_matched")
    tree.Project(f"h_matched_pt_mw_{tag}_{name}", f"{name}_truth_pt", f"{name}_m_matched")
    return hists

def build_corr_hist(tree):
    """Bin the truth W -> qq deltaR vs truth W pT from the decay correlations
    results, with the same binning and smoothing as truth_decay_correlations_v3."""
    h2 = ROOT.TH2D("h2_W_dR", "", 200, 0.0, 1200.0, 200, 0.0, 3.5)
    tree.Project("h2_W_dR", "W_dR:W_pt", "weight")
    h2.Smooth(option="k5b")
    return h2

def set_pixel_text(axis, label_px, title_px):
    """Draw the axis labels and title with font precision 3, where the size is
    in pixels and so keeps the same absolute size across pads of different heights."""
    axis.SetLabelFont(43)
    axis.SetLabelSize(label_px)
    axis.SetTitleFont(43)
    axis.SetTitleSize(title_px)

def make_efficiency(h_mat, h_tot, color, with_mass_window):
    """Efficiency curve of one reconstruction, with or without mass window. The
    Clopper-Pearson intervals are carried by the painted TGraphAsymmErrors and
    drawn as a filled uncertainty band."""
    eff = ROOT.TEfficiency(h_mat, h_tot)
    eff.SetStatisticOption(ROOT.TEfficiency.kFCP)
    eff.SetConfidenceLevel(0.95)
    eff.SetLineWidth(2)
    eff.SetLineStyle(2 if with_mass_window else 1)
    eff.SetLineColor(color)
    eff.SetFillColorAlpha(color, 0.1 if with_mass_window else 0.2)
    return eff

def plot_efficiency(name, fatjet_hists, dijet_hists):
    """Draw the FatJet and dijet reconstruction efficiency of one boson (each
    with and without the mass window) on one canvas and save the figure."""
    canvas = ROOT.TCanvas("c_eff_pt_" + name, "FatJet vs Dijet Efficiency vs pT", 800, 600)

    curricula = []
    for hists, color in [[fatjet_hists, ROOT.kBlack], [dijet_hists, 2]]:
        curricula.append(make_efficiency(hists["matched_pt"], hists["total_pt"], color, False))
        curricula.append(make_efficiency(hists["matched_pt_mw"], hists["total_pt"], color, True))

    # the painted TGraphAsymmErrors of each efficiency carries the asymmetric
    # Clopper-Pearson intervals; it is materialized on a scratch canvas and
    # split into a filled band and the central line drawn here
    scratch = ROOT.TCanvas("scratch_eff_" + name, "", 10, 10)
    bands, curves = [], []
    for eff in curricula:
        eff.Draw("a2")
        ROOT.gPad.Update()
        graph = eff.GetPaintedGraph()
        graph.SetLineColor(eff.GetLineColor())
        graph.SetLineStyle(eff.GetLineStyle())
        graph.SetLineWidth(eff.GetLineWidth())
        band = graph.Clone()
        band.SetLineColorAlpha(eff.GetLineColor(), 0.0)   # hide the band border
        bands.append(band)
        curves.append(graph)
    canvas.cd()
    frame = canvas.DrawFrame(150, 0.0, 1150, 1.65)
    frame.SetTitle("Reconstruction efficiency; Truth p_{T} [GeV]; #epsilon_{reco.} (%)")
    frame.GetXaxis().SetNdivisions(505, True)
    frame.GetYaxis().SetNdivisions(1010, True)
    for band in bands:
        band.Draw("same 2")
        ROOT.gPad.Update()
    for curve in curves:
        curve.Draw("same L")
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
    leg.AddEntry(curricula[0], "FatJet (angular match)", "l")
    leg.AddEntry(curricula[1], "FatJet + #it{m_{SD}} window", "l")
    leg.AddEntry(curricula[2], "Dijet (angular match)", "l")
    leg.AddEntry(curricula[3], "Dijet + #it{m_{jj}} window", "l")
    leg.Draw()

    output = "./plots/reconstruction_efficiency_comparison_vs_pt_" + name
    canvas.Update()
    canvas.SaveAs(output + ".png")
    canvas.SaveAs(output + ".pdf")
    canvas.Close()
    return output

def plot_efficiency_with_correlations(name, fatjet_hists, dijet_hists, corr_tree):
    """Draw the W -> qq deltaR 2D histogram below the FatJet vs dijet efficiency
    plot of one boson, with both panels sharing the same x axis."""
    canvas = ROOT.TCanvas("c_eff_corr_" + name, "FatJet vs Dijet Efficiency vs pT with W Decay Correlations", 800, 800)
    ROOT.gStyle.SetPalette(ROOT.kRainbow)

    pad_top = ROOT.TPad("pad_top_" + name, "", 0.0, 0.4, 1.0, 1.0)
    pad_top.SetBottomMargin(0.01)
    pad_top.Draw()
    pad_bot = ROOT.TPad("pad_bot_" + name, "", 0.0, 0.0, 1.0, 0.4)
    pad_bot.SetLogz(True)
    pad_bot.SetTopMargin(0.02)
    pad_bot.SetBottomMargin(0.22)   # room for the enlarged x tick labels and title
    pad_bot.Draw()

    pad_top.cd()
    curricula = []
    for hists, color in [[fatjet_hists, ROOT.kBlack], [dijet_hists, 2]]:
        curricula.append(make_efficiency(hists["matched_pt"], hists["total_pt"], color, False))
        curricula.append(make_efficiency(hists["matched_pt_mw"], hists["total_pt"], color, True))

    # the painted TGraphAsymmErrors of each efficiency carries the asymmetric
    # Clopper-Pearson intervals; it is materialized on a scratch canvas and
    # split into a filled band and the central line drawn here
    scratch = ROOT.TCanvas("scratch_corr_" + name, "", 10, 10)
    bands, curves = [], []
    for eff in curricula:
        eff.Draw("a2")
        ROOT.gPad.Update()
        graph = eff.GetPaintedGraph()
        graph.SetLineColor(eff.GetLineColor())
        graph.SetLineStyle(eff.GetLineStyle())
        graph.SetLineWidth(eff.GetLineWidth())
        band = graph.Clone()
        band.SetLineColorAlpha(eff.GetLineColor(), 0.0)   # hide the band border
        bands.append(band)
        curves.append(graph)
    pad_top.cd()

    # a TEfficiency axis cannot span beyond its own data, so the frame is drawn
    # explicitly to get the same 0-1200 GeV range as the 2D histogram below.
    frame = pad_top.DrawFrame(0.0, 0.0, 1200.0, 1.65)
    frame.GetYaxis().SetTitle("#epsilon_{reco.} (%)")
    frame.GetXaxis().SetLabelSize(0)   # shared axis: labelled by the 2D histogram below
    frame.GetXaxis().SetNdivisions(506, True)
    for band in bands:
        band.Draw("same 2")
        ROOT.gPad.Update()
    for curve in curves:
        curve.Draw("same L")
        ROOT.gPad.Update()

    draw_label_block([
        "#scale[1.1]{#bf{#sqrt{s} = %s TeV, %s fb^{-1}}}" % (13.6, 500),
        "Ak10-jet vs Dijet (Ak4) Reco. Efficiency (%)",
        "Ak10-jet: #DeltaR(W, J_{reco.}) #leq 1.0",
        "Dijet (Ak4): #DeltaR(q, j) #leq 0.4",
        "Clopper-Pearson @ 95% CL"
    ], text_size=0.04, step=0.06)

    leg = ROOT.TLegend(0.6, 0.65, 0.75, 0.92)
    leg.SetFillColor(0)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.04)
    leg.AddEntry(curricula[0], "Ak10-jet (angular match)", "l")
    leg.AddEntry(curricula[1], "Ak10-jet + #it{m_{SD}} window", "l")
    leg.AddEntry(curricula[2], "Dijet (Ak4) (angular match)", "l")
    leg.AddEntry(curricula[3], "Dijet (Ak4) + #it{m_{jj}} window", "l")
    leg.Draw()

    pad_bot.cd()
    h2 = build_corr_hist(corr_tree)
    h2.GetXaxis().SetRangeUser(0.0, 1200.0)
    h2.GetXaxis().SetNdivisions(506)
    # the bottom panel text is enlarged to the same absolute size as the top
    # panel, where the ATLAS style uses 0.05 of the pad height
    h2.GetXaxis().SetTitle("Truth p_{T} [GeV]")
    set_pixel_text(h2.GetXaxis(), 23, 23)
    h2.GetYaxis().SetTitle("#it{#DeltaR}(#it{q_{1}, q_{2}})")
    h2.GetYaxis().SetNdivisions(1010)
    set_pixel_text(h2.GetYaxis(), 23, 23)
    h2.GetYaxis().SetTitleOffset(2.6)   # same visual title offset as the top panel
    h2.GetZaxis().SetTitle("Weighted Events")
    h2.GetZaxis().SetNdivisions(1010)
    h2.GetZaxis().SetTitleOffset(1.1) 
    set_pixel_text(h2.GetZaxis(), 23, 23)
    # a TPaletteAxis wider than it is tall is drawn horizontally (ROOT >= 6.26),
    # here inside the frame over the empty top-right corner of the map
    palette = ROOT.TPaletteAxis(600.0, 2.0, 1130.0, 2.5, h2)
    h2.GetListOfFunctions().Add(palette)
    h2.Draw("COLZS")
    draw_label_block([
        "#it{pp #rightarrow N^{#pm}N^{0}, N^{0} #rightarrow W^{#pm}l^{#mp}, W^{#pm} #rightarrow q_{1}q_{2}}",
        "#it{m_{N}} = 1.2 TeV",
    ], text_size=0.055, x=0.56, step=0.08)

    output = "./plots/reconstruction_efficiency_comparison_vs_pt_" + name + "_with_w_correlations"
    canvas.Update()
    canvas.SaveAs(output + ".png")
    canvas.SaveAs(output + ".pdf")
    canvas.Close()
    return output


def main():
    ROOT.gStyle.SetEndErrorSize(2)

    os.makedirs("./plots", exist_ok=True)
    fatjet_tree = load_tree(fatjet_results_file)
    dijet_tree = load_tree(dijet_results_file)
    correlations_tree = load_tree(correlations_results_file)

    for name in BOSONS:
        fatjet_hists = build_hists(fatjet_tree, name, "fatjet")
        dijet_hists = build_hists(dijet_tree, name, "dijet")
        output = plot_efficiency(name, fatjet_hists, dijet_hists)
        print(f"Saved {output}.png/.pdf")
        if name == "W":
            output = plot_efficiency_with_correlations(name, fatjet_hists, dijet_hists, correlations_tree)
            print(f"Saved {output}.png/.pdf")

if __name__ == "__main__":
    main()
