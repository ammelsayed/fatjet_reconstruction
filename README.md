# FatJet reconstruction efficiency scripts (v1-v6)

- **v1**: Baseline version. Reads one Delphes file, identifies hadronic W bosons from the Particle branch, matches them to fatjets using a simple selection in $\Delta R$, transverse-momentum ratio, and soft-drop mass, then prints a basic efficiency summary and saves a simple plot.
- **v2**: Generalizes the logic to W, Z, and H bosons. Introduces helper functions for boson finding and fatjet handling, improves the matching workflow, and produces a more polished multi-boson efficiency plot.
- **v3**: Adds ROOT output persistence, better plotting labels, and a reusable `main()` entry point. The purpose is to save results and avoid rerunning the full analysis from scratch.
- **v4**: Makes the workflow more scalable. Adds configurable options such as overwrite behavior, entry limits, output filenames, event-weight handling, and **support for parallel processing** over chunks of the input sample.
- **v5**: Tightens the matching cone to $\Delta R \le 0.75$ and improves the plotting style. It also uses a more explicit statistical treatment in the efficiency plots and adds cleaner annotations for the analysis setup.
- **v6**: Moves from simple histogram-based outputs to an event-level `TTree` structure with per-object branches such as truth pT, reco pT, reco mass, and match flags. It also adds eta-efficiency plots and pT/mass response histograms, making it much more useful for deeper physics studies.
- **dijet** (`dijet_reconstruction_rate.py`): Same truth-level boson selection as the fatjet case (W/Z/H decaying hadronically), but the reco object is a **dijet** built from two AK4 jets instead of a single AK10 fatjet. The matching is done at parton level: the two truth decay quarks of the boson are matched to two distinct reconstructed jets by `DeltaR < 0.4`, which is the ATLAS/CMS standard for parton-to-jet matching (ATLAS arXiv:2308.00716, arXiv:1809.08597, ATLFAST 2.0; CMS CMS-PAS-JME-16-003 with `R_min = 0.4`, arXiv:1912.06046). The reco-side jet fiducial cuts are `pT > 25 GeV`, `|eta| < 2.5`, as used by ATLAS for a resolved V->qq dijet (arXiv:1706.01702). The efficiency is reported **twice**: once from the parton-level matching alone, and once with the additional requirement that the reconstructed dijet invariant mass falls in a window around the parent boson mass.
- **Possible future updates**: Write the loop method in C/C++ and use Cython for more speed improvements.

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

