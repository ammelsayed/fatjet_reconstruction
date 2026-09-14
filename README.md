# FatJet reconstruction efficiency scripts (v1-v6)

- **v1**: Baseline version. Reads one Delphes file, identifies hadronic W bosons from the Particle branch, matches them to fatjets using a simple selection in $\Delta R$, transverse-momentum ratio, and soft-drop mass, then prints a basic efficiency summary and saves a simple plot.
- **v2**: Generalizes the logic to W, Z, and H bosons. Introduces helper functions for boson finding and fatjet handling, improves the matching workflow, and produces a more polished multi-boson efficiency plot.
- **v3**: Adds ROOT output persistence, better plotting labels, and a reusable `main()` entry point. The purpose is to save results and avoid rerunning the full analysis from scratch.
- **v4**: Makes the workflow more scalable. Adds configurable options such as overwrite behavior, entry limits, output filenames, event-weight handling, and **support for parallel processing** over chunks of the input sample.
- **v5**: Tightens the matching cone to $\Delta R \le 0.75$ and improves the plotting style. It also uses a more explicit statistical treatment in the efficiency plots and adds cleaner annotations for the analysis setup.
- **v6**: Moves from simple histogram-based outputs to an event-level `TTree` structure with per-object branches such as truth pT, reco pT, reco mass, and match flags. It also adds eta-efficiency plots and pT/mass response histograms, making it much more useful for deeper physics studies.
- **Possible future updates**: Write the loop method in C/C++ and use Cython for more speed improvements.

## Why these updates matter for the physics study

The goal of these scripts is to measure how often a hadronically decaying boson at truth level is recovered by a reconstructed fatjet in the Delphes sample. That is a standard efficiency study, and the progression from v1 to v6 reflects how the analysis became more robust and more suited to larger-scale physics work.

The main changes were driven by the needs of the study rather than by code convenience. In v4, the workflow was expanded to use parallel processing because studying larger samples and larger sets of input files quickly becomes computationally expensive. In v5, the analysis moved away from event-weighted handling and toward counting-based efficiency estimates, since the quantity of interest is fundamentally a ratio of matched to total cases and the event weights are not essential to the reconstruction performance measurement. In v6, the structure was changed to a tree-based workflow so that the study becomes more scalable, easier to extend, and better suited to producing both efficiency plots and more detailed reconstruction diagnostics.

