using Documenter
using NaNTracker

DocMeta.setdocmeta!(NaNTracker, :DocTestSetup, :(using NaNTracker); recursive=true)

makedocs(;
    modules  = [NaNTracker],
    sitename = "NaNTracker.jl",
    repo     = Documenter.Remotes.GitHub("mashu", "NaNTracker.jl"),
    pages    = [
        "Home"      => "index.md",
        "Guide"     => "guide.md",
        "Reference" => "reference.md",
    ],
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://mashu.github.io/NaNTracker.jl",
    ),
    warnonly = [:missing_docs],
)

deploydocs(;
    repo = "github.com/mashu/NaNTracker.jl.git",
    devbranch = "main",
)
