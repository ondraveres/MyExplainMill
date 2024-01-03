using Pkg
using Documenter, DocumenterCitations
using ExplainMill

makedocs(
         sitename = "ExplainMill.jl",
         format = Documenter.HTML(sidebar_sitename=false,
                                  collapselevel=2,
                                  assets=["assets/favicon.ico", "assets/custom.css"]),
         warnonly = Documenter.except(:eval_block, :example_block, :meta_block, :setup_block),
         modules = [ExplainMill],
         plugins = [
             CitationBibliography(joinpath(@__DIR__, "references.bib"), style=:numeric)
         ],
         pages = [
                  "Home" => "index.md",
                  "Manual" => [],
                  "Examples" => [],
                  "Public API" => [],
                  "References" => "references.md",
                  "Citation" => "citation.md"
                 ],
        )

deploydocs(
    repo = "github.com/CTUAvastLab/ExplainMill.jl.git"
)
