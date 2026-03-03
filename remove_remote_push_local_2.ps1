param(
    [string]$Remote = "origin",
    [string]$Branch = "main"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$LocalPath = (Get-Location).Path
Write-Host "Working in: $LocalPath" -ForegroundColor Cyan

if (-not (Test-Path ".git")) {
    Write-Error "No .git folder found in '$LocalPath'. Aborting."
    exit 1
}

# -- 1. Get the remote URL
$remoteUrl = git remote get-url $Remote
Write-Host "Remote URL: $remoteUrl" -ForegroundColor Cyan

# -- 2. Clone remote into a temp folder
$tempDir = Join-Path $env:TEMP ("remote_clone_" + [System.IO.Path]::GetRandomFileName().Replace(".", ""))
Write-Host "Cloning remote into: $tempDir" -ForegroundColor Cyan
git clone --branch $Branch $remoteUrl $tempDir
if ($LASTEXITCODE -ne 0) {
    Write-Error "Clone failed. Aborting."
    exit 1
}

# -- 3. Delete everything in the clone except .git and .github
Write-Host "Deleting all remote files (keeping .github)..." -ForegroundColor Cyan
Get-ChildItem -Path $tempDir -Force | Where-Object {
    $_.Name -ne ".git" -and $_.Name -ne ".github"
} | Remove-Item -Recurse -Force

# -- 4. Copy local files into the clone (skip .git only)
Write-Host "Copying local files..." -ForegroundColor Cyan
Get-ChildItem -Path $LocalPath -Force | Where-Object {
    $_.Name -ne ".git"
} | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination (Join-Path $tempDir $_.Name) -Recurse -Force
}

# -- 5. Commit and push
Set-Location $tempDir
git add --all
$status = git status --porcelain
if ($status) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    git commit -m "Replace remote contents with local files [$timestamp]"
    git push $Remote $Branch --force
    Write-Host "Push complete." -ForegroundColor Green
} else {
    Write-Host "Nothing to commit - remote already matches local." -ForegroundColor Yellow
}

# -- 6. Clean up
Set-Location $LocalPath
Remove-Item $tempDir -Recurse -Force

Write-Host ""
Write-Host "Done! Remote '$Remote/$Branch' now has your local files. Local files unchanged." -ForegroundColor Green
Write-Host ""