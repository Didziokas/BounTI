# Installs (or removes) the BounTI plugin for every Dragonfly installation
# on this user account. Per-user location: no admin rights required and
# nothing under "C:\Program Files\Dragonfly" is touched.
#
#   Install:     powershell -ExecutionPolicy Bypass -File Install.ps1
#   Uninstall:   powershell -ExecutionPolicy Bypass -File Install.ps1 -Uninstall
#
param([switch]$Uninstall)

$pluginName = 'BounTI'
$src = Join-Path $PSScriptRoot $pluginName

function Stop-WithError([string]$message) {
    Write-Warning $message
    # Launched via right-click "Run with PowerShell", the console closes the
    # instant the script ends; hold it open so the error is readable.
    Write-Host 'Press Enter to close this window...'
    [void](Read-Host)
    exit 1
}

$parentDirs = @(
    (Join-Path $env:LOCALAPPDATA 'Comet'),
    (Join-Path $env:LOCALAPPDATA 'ORS'),
    $env:LOCALAPPDATA
)
$dataRoots = $parentDirs | ForEach-Object {
    Get-ChildItem $_ -Directory -Filter 'Dragonfly*' -ErrorAction SilentlyContinue
} | Sort-Object -Property FullName -Unique
if (-not $dataRoots) {
    Stop-WithError 'No Dragonfly data folder found under %LOCALAPPDATA% (checked %LOCALAPPDATA%\Comet\Dragonfly*, %LOCALAPPDATA%\ORS\Dragonfly* and %LOCALAPPDATA%\Dragonfly*). Open Dragonfly once (it creates the folder), then run this script again.'
}

foreach ($root in $dataRoots) {
    $pluginsDir = Join-Path $root.FullName 'pythonUserExtensions\Plugins'
    $dest = Join-Path $pluginsDir $pluginName

    if ($Uninstall) {
        if (Test-Path $dest) {
            try {
                Remove-Item -Recurse -Force -ErrorAction Stop $dest
                Write-Host "Removed: $dest"
            } catch {
                Stop-WithError "Could not remove $dest - is Dragonfly running? ($($_.Exception.Message))"
            }
        } else {
            Write-Host "Not installed (nothing removed): $dest"
        }
        continue
    }

    if (-not (Test-Path (Join-Path $src 'BounTI.py'))) {
        Stop-WithError "Plugin source not found next to this script (expected folder '$pluginName' containing BounTI.py): $src"
    }
    if (-not (Test-Path $pluginsDir)) {
        try {
            New-Item -ItemType Directory -Path $pluginsDir -ErrorAction Stop | Out-Null
        } catch {
            Stop-WithError "Could not create $pluginsDir ($($_.Exception.Message))"
        }
    }
    if (Test-Path $dest) {
        try {
            Remove-Item -Recurse -Force -ErrorAction Stop $dest
        } catch {
            Stop-WithError "Could not remove the previous install at $dest - is Dragonfly running? ($($_.Exception.Message))"
        }
    }
    try {
        Copy-Item -Recurse -ErrorAction Stop $src $dest
    } catch {
        Stop-WithError "Could not copy to $dest ($($_.Exception.Message))"
    }
    $pycache = Join-Path $dest '__pycache__'
    if (Test-Path $pycache) { Remove-Item -Recurse -Force $pycache }
    Write-Host "Installed: $dest"
}

Write-Host ''
Write-Host 'Restart Dragonfly, then open Utilities > Plugins > Run BounTI'
Write-Host '(also right-click a channel image > "Segment with BounTI...").'
