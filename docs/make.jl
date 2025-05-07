using Documenter
using Attention

# Get the package version from the Project.toml
using Pkg
project_toml_path = joinpath(pkgdir(Attention), "Project.toml")
project_info = Pkg.TOML.parsefile(project_toml_path)
current_version = project_info["version"]


makedocs(
    sitename = "Attention.jl",
    authors = "Mateusz Kaduk <mateusz.kaduk@gmail.com>",
    version = current_version,
    format = Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mashu.github.io/Attention.jl/stable/",
        repolink = "https://github.com/mashu/Attention.jl",
        assets=String[],
    ),
    modules = [Attention],
    pages = [
        "Home" => "index.md",
        "API" => [
            "Public API" => "api/public.md",
        ]
    ],
    repo = "https://github.com/mashu/Attention.jl/blob/{commit}{path}#{line}",
    doctest = true,
)

deploydocs(
    repo = "github.com/mashu/Attention.jl.git",
    devbranch = "main",
    push_preview = true
) 