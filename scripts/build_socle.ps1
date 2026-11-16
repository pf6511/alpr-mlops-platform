# Prépare le socle d'entraînement du classificateur de marque à partir du jeu
# VMMRdb_europe : un dossier par marque, dont les noms sont exactement les 22 classes
# du modèle (audi, bmw, ..., « mercedes benz », ..., volvo).
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\build_socle.ps1 -Source "C:\...\VMMRdb_europe" -Destination "C:\...\socle"
#
# - chaque dossier garde son nom (= classe du modèle) ;
# - les $PerBrand premières images, triées par nom : un socle reproductible ;
# - le dossier de destination est vidé au préalable.
#
# Ensuite : docker cp <socle> alpr-engine:/app/data/base
#           python -m src.training.train --base-dir data/base
param(
    [Parameter(Mandatory = $true)] [string] $Source,
    [Parameter(Mandatory = $true)] [string] $Destination,
    [int] $PerBrand = 15
)

if (-not (Test-Path $Source)) { throw "Dossier source introuvable : $Source" }
if (Test-Path $Destination) { Remove-Item $Destination -Recurse -Force }

Get-ChildItem $Source -Directory | ForEach-Object {
    $target = Join-Path $Destination $_.Name.ToLower()
    New-Item -ItemType Directory -Force $target | Out-Null
    Get-ChildItem $_.FullName -File |
        Where-Object { $_.Extension -match '^\.(jpe?g|png)$' } |
        Sort-Object Name |
        Select-Object -First $PerBrand |
        Copy-Item -Destination $target
}

# Bilan : nombre d'images par classe
Get-ChildItem $Destination -Directory | Sort-Object Name | ForEach-Object {
    "{0,-14} {1}" -f $_.Name, (Get-ChildItem $_.FullName -File).Count
}
