# FatJet reconstruction efficiency scripts (v1-v6)

- **v1**: Baseline version. Reads one Delphes file, identifies hadronic W bosons from the Particle branch, matches them to fatjets using a simple selection in $\Delta R$, transverse-momentum ratio, and soft-drop mass, then prints a basic efficiency summary and saves a simple plot.
- **v2**: Generalizes the logic to W, Z, and H bosons. Introduces helper functions for boson finding and fatjet handling, improves the matching workflow, and produces a more polished multi-boson efficiency plot.
- **v3**: Adds ROOT output persistence, better plotting labels, and a reusable `main()` entry point. The purpose is to save results and avoid rerunning the full analysis from scratch.
- **v4**: Makes the workflow more scalable. Adds configurable options such as overwrite behavior, entry limits, output filenames, event-weight handling, and **support for parallel processing** over chunks of the input sample.
- **v5**: Tightens the matching cone to $\Delta R \le 0.75$ and improves the plotting style. It also uses a more explicit statistical treatment in the efficiency plots and adds cleaner annotations for the analysis setup.
- **v6**: Moves from simple histogram-based outputs to an event-level `TTree` structure with per-object branches such as truth pT, reco pT, reco mass, and match flags. It also adds eta-efficiency plots and pT/mass response histograms, making it much more useful for deeper physics studies.
- **v7** (`fatjet_reconstruction_rate_v7.py`): Same truth-level boson selection and the same `DeltaR` matching as v6, but the efficiency is now reported **twice**: once from the geometric matching alone, and once with the additional requirement that the groomed (soft-drop) mass of the fatjet lies within a window around the parent boson mass. It is the version to use from now on: it writes to the same output file names as v6 (`fatjet_reconstruction_efficiency_results.root`, `fatjet_histograms_results.root`, `plots/fatjet_reconstruction_efficiency_vs_pt.*` and `..._vs_eta.*`), so it is a drop-in replacement. The mass window is applied as a *second* stage, exactly as in the dijet study below, and is the same requirement that v5 of this script used as a hard cut in the matching loop. The window used here is `m_V +/- 25 GeV` (W: 55.4-105.4, Z: 66.2-116.2, H: 100.0-150.0); the CMS equivalent for a boosted W/Z tagger is 65-85 GeV (W) and 85-105 GeV (Z) on the pruned mass (arXiv:1410.4227, CMS-PAS-JME-13-006).
- **dijet** (`dijet_reconstruction_rate.py`): Same truth-level boson selection as the fatjet case (W/Z/H decaying hadronically), but the reco object is a **dijet** built from two AK4 jets instead of a single AK10 fatjet. The matching is done at parton level: the two truth decay quarks of the boson are matched to two distinct reconstructed jets by `DeltaR < 0.4`, which is the ATLAS/CMS standard for parton-to-jet matching (ATLAS arXiv:2308.00716, arXiv:1809.08597, ATLFAST 2.0; CMS CMS-PAS-JME-16-003 with `R_min = 0.4`, arXiv:1912.06046). The reco-side jet fiducial cuts are `pT > 25 GeV`, `|eta| < 2.5`, as used by ATLAS for a resolved V->qq dijet (arXiv:1706.01702). The efficiency is reported **twice**: once from the parton-level matching alone, and once with the additional requirement that the reconstructed dijet invariant mass falls in a window around the parent boson mass.
- **Possible future updates**: Write the loop method in C/C++ and use Cython for more speed improvements.

## Plot formatting

All the efficiency plots (fatjet v7 and dijet) share the same layout:

- The canvas is 1400 x 700. The annotation block sits in the **left margin** and the legend in the **right margin**, so that neither of them can ever overlap the data points.
- The two selection requirements of the same boson share the boson colour and are told apart by the **line and marker style only**: solid line with a filled marker for the plain reconstruction efficiency, dashed line with an open marker for the efficiency with the mass window. They therefore read as a pair, and stay distinguishable when the figure is printed in black and white.
- The legend uses two columns, one per requirement.
- The y-axis range is set explicitly. ROOT auto-ranges from the first graph that is drawn, which would clip the mass-window curve below the bottom of the frame.
- Each boson has its own marker shape (W circle, Z square, H triangle) on top of its colour, and the efficiency with the mass window uses the open version of that marker with a dashed line, so the figure stays readable in black and white.
- The points are connected by a line in the pT plot, where the efficiency varies smoothly, but not in the eta plot, where it oscillates from bin to bin.
- The exported PDF is cropped to the drawing. ROOT always writes the PDF on a fixed A4 page and leaves the drawing in a corner of it, so the page is cropped with ghostscript afterwards.

## Does the dijet efficiency need an invariant mass requirement?

No — the reconstruction efficiency itself is defined **without** it, and the mass requirement is a second, separate stage. This is what ATLAS and CMS do, and both publish the efficiency with and without the mass cut:

- The reconstruction efficiency is a purely geometric quantity: the fraction of truth bosons for which the required reco object(s) are found. ATLAS quotes the Higgs-boson acceptance "with the tight, loose and no mass window requirements" side by side (arXiv:1906.11005, Fig. 6), and ATLFAST 2.0 reports the b-jet reconstruction efficiency (epsilon_b-jet_recon ~ 80-83 %) separately from the efficiency in a mass bin around `m_bb` (Tables 3-1/3-2).
- The dijet invariant mass window is a **selection** that belongs to the analysis, not to the definition of the efficiency. The windows used for resolved dijets from a boson decay are:
  - W/Z: `64 < m_jj < 106 GeV` (ATLAS arXiv:1905.07714, STDM-2017-20)
  - H(125) -> bb: `110 < m_bb < 140 GeV` (ATLAS arXiv:1712.06518, JHEP 03 (2018) 174)
  - W (aTGC-style): `75 < m_jj < 95 GeV` (CMS arXiv:1210.7544)
- Note that matching a dijet to the truth boson by an angular `DeltaR` between the boson and the dijet has no precedent in the public ATLAS/CMS record (a dijet has no well-defined axis); the parton-level matching used here is the standard alternative.
- Caveat on the low-mass tail: the reconstructed dijet mass peak is shifted low for b-jets (ATLFAST 2.0 quotes -12.5 GeV from final-state radiation/hadronisation and -4.9 GeV from the B-field), so a symmetric window around `m_V` loses efficiency from the low-mass side. In this sample the H -> bb peak sits at ~110 GeV for a true 125 GeV Higgs, and the matched dijet efficiency is 65% without the mass window versus 33% with it.
- The script also stores the truth-level `DeltaR(q1, q2)` between the two decay quarks. The efficiency drops sharply below `DeltaR(q1, q2) ~ 0.4`, where the two jets merge into one (ATLAS arXiv:1506.00285; `DeltaR(q1,q2) ~ 2 m_V / pT`); this is the resolved/boosted transition and is the main limitation of the dijet reconstruction at high boson `pT`.

## Why these updates matter for the physics study

The goal of these scripts is to measure how often a hadronically decaying boson at truth level is recovered by a reconstructed fatjet in the Delphes sample. That is a standard efficiency study, and the progression from v1 to v6 reflects how the analysis became more robust and more suited to larger-scale physics work.

The main changes were driven by the needs of the study rather than by code convenience. In v4, the workflow was expanded to use parallel processing because studying larger samples and larger sets of input files quickly becomes computationally expensive. In v5, the analysis moved away from event-weighted handling and toward counting-based efficiency estimates, since the quantity of interest is fundamentally a ratio of matched to total cases and the event weights are not essential to the reconstruction performance measurement. In v6, the structure was changed to a tree-based workflow so that the study becomes more scalable, easier to extend, and better suited to producing both efficiency plots and more detailed reconstruction diagnostics.

