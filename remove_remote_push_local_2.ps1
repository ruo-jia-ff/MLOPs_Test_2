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
# -- 2. Clone the remote into a temp folder
$tempDir = Join-Path $env:TEMP ("remote_clone_" + [System.IO.Path]::GetRandomFileName().Replace(".", ""))
Write-Host "Cloning remote into temp folder: $tempDir" -ForegroundColor Cyan
git clone --branch $Branch $remoteUrl $tempDir
if ($LASTEXITCODE -ne 0) {
    Write-Error "Clone failed. Aborting."
    exit 1
}
# -- 3. Backup remote .github contents into memory
$githubFiles = @{}
$entries = git -C $tempDir ls-tree -r --name-only $Branch -- .github 2>$null
foreach ($file in $entries) {
    $gitRef = "${Branch}:${file}"
    $content = git -C $tempDir show $gitRef
    $githubFiles[$file] = $content
}
Write-Host "Saved $($githubFiles.Count) file(s) from remote .github" -ForegroundColor Green
# -- 4. Delete everything in the temp clone EXCEPT .git and .github
Write-Host "Clearing remote contents (keeping .github)..." -ForegroundColor Cyan
Get-ChildItem -Path $tempDir -Force | Where-Object {
    $_.Name -ne ".git" -and $_.Name -ne ".github"
} | Remove-Item -Recurse -Force
# -- 5. Copy local files into temp clone EXCEPT .git and .github
Write-Host "Copying local files into temp clone..." -ForegroundColor Cyan
Get-ChildItem -Path $LocalPath -Force | Where-Object {
    $_.Name -ne ".git" -and $_.Name -ne ".github"
} | ForEach-Object {
    $dest = Join-Path $tempDir $_.Name
    Copy-Item -Path $_.FullName -Destination $dest -Recurse -Force
}
# -- 6. Overlay remote .github (only if it existed remotely)
if ($githubFiles.Count -gt 0) {
    Write-Host "Restoring remote .github files..." -ForegroundColor Cyan
    foreach ($file in $githubFiles.Keys) {
        $dest = Join-Path $tempDir $file
        $dir = Split-Path $dest -Parent
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
        $githubFiles[$file] | Set-Content -Path $dest -Encoding UTF8
    }
}
# -- 7. Commit and push
Write-Host "Committing and pushing to $Remote/$Branch..." -ForegroundColor Cyan
Set-Location $tempDir
git add --all
$status = git status --porcelain
if ($status) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    git commit -m "Sync local files to remote, preserve remote .github [$timestamp]"
    git push $Remote $Branch --force
    Write-Host "Push complete." -ForegroundColor Green
} else {
    Write-Host "Nothing to commit - remote already matches local." -ForegroundColor Yellow
}
# -- 8. Clean up
Set-Location $LocalPath
Remove-Item $tempDir -Recurse -Force
Write-Host ""
Write-Host "Done! Remote '$Remote/$Branch' now mirrors your local files (remote .github preserved). Local files unchanged." -ForegroundColor Green
Write-Host ""