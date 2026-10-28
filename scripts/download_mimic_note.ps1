# Download the MIMIC-IV-Note v2.2 discharge files from PhysioNet, resuming after network drops.
# Run in your own PowerShell window (you type your PhysioNet password once; it stays in memory only):
#   powershell -ExecutionPolicy Bypass -File scripts\download_mimic_note.ps1 -User <physionet user> -Dest <folder>
# Requires a signed MIMIC-IV-Note data use agreement. Files land OUTSIDE this repository.
#
# PhysioNet only accepts password (HTTP Basic) logins from wget-style clients: other user agents
# get 403 before the password is checked. So curl identifies as wget here. A login error (401/403)
# stops the script at once, so a wrong password is never retried (no account lockout).
param(
    [Parameter(Mandatory)][string]$User,
    [Parameter(Mandatory)][string]$Dest
)

$base = "https://physionet.org/files/mimic-iv-note/2.2/note"
$files = @("discharge_detail.csv.gz", "discharge.csv.gz")
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

$secure = Read-Host "PhysioNet password for $User" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
# curl reads the credentials from stdin (--config -), so they never appear on the command line or on disk
$escaped = $plain.Replace('\', '\\').Replace('"', '\"')
$plain = $null

foreach ($f in $files) {
    $out = Join-Path $Dest $f
    for ($i = 1; $i -le 1000; $i++) {
        $code = "user = `"$User`:$escaped`"" | curl.exe --config - --fail --location --continue-at - `
            --user-agent "Wget/1.21.4" --retry 5 --retry-delay 15 --speed-time 120 --speed-limit 200 `
            --write-out "%{http_code}" --output $out "$base/$f"
        if ($LASTEXITCODE -eq 0) { Write-Host "`ndone: $f"; break }
        if ($code -eq "401" -or $code -eq "403") {
            $escaped = $null
            Write-Host "`nLogin refused (HTTP $code). Check the password and that the MIMIC-IV-Note DUA is signed, then run again."
            exit 1
        }
        if ($code -eq "416") { Write-Host "`n$f is already complete"; break }
        Write-Host "`nattempt $i stopped (HTTP $code, curl exit $LASTEXITCODE); resuming in 30 s..."
        Start-Sleep -Seconds 30
    }
}
$escaped = $null
Write-Host "Finished. Files are in $Dest (sizes below). Tell Claude to verify the checksums."
Get-ChildItem $Dest | Select-Object Name, Length
