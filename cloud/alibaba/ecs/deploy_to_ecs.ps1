param(
  [Parameter(Mandatory = $true)]
  [string]$PublicIp,

  [Parameter(Mandatory = $true)]
  [string]$FlowPilotToken,

  [string]$SshUser = "root",
  [string]$SshKey = "",
  [string]$PublicBaseUrl = "",
  [string]$QwenApiKey = "",
  [string]$QwenModel = "qwen3.7-plus",
  [string]$NudgeMode = "auto",

  [Parameter(Mandatory = $true)]
  [string]$BasicAuthUser,

  [Parameter(Mandatory = $true)]
  [string]$BasicAuthPassword,
  [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..\..\..").Path
)

$ErrorActionPreference = "Stop"

function Convert-ToShellSingleQuoted {
  param([string]$Value)
  return "'" + $Value.Replace("'", "'\''") + "'"
}

if (-not $PublicBaseUrl) {
  $PublicBaseUrl = "http://$PublicIp"
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$staging = Join-Path $env:TEMP "bamboo-focus-ecs-$stamp"
$zipPath = Join-Path $env:TEMP "bamboo-focus-ecs-$stamp.zip"

$skipDirs = @(
  ".git",
  ".venv-pi",
  "venv",
  "__pycache__",
  "monitor_data",
  "object_monitor_data",
  "nudge_agent_data",
  "models",
  ".superpowers"
)

$skipFiles = @(
  ".env",
  "Qwen Cloud Proof of Deployment.md",
  "webcam_event_queue.jsonl"
)

New-Item -ItemType Directory -Force -Path $staging | Out-Null

Get-ChildItem -LiteralPath $ProjectRoot -Force | ForEach-Object {
  if ($skipDirs -contains $_.Name) { return }
  if ($skipFiles -contains $_.Name) { return }
  Copy-Item -LiteralPath $_.FullName -Destination $staging -Recurse -Force
}

Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $zipPath -Force

$target = "$SshUser@$PublicIp"
$sshArgs = @()
if ($SshKey) {
  $sshArgs += @("-i", $SshKey)
}

Write-Host "Uploading $zipPath to $target..."
scp @sshArgs $zipPath "${target}:/tmp/flowpilot.zip"

Write-Host "Installing on Alibaba Cloud ECS..."
$remoteFlowPilotToken = Convert-ToShellSingleQuoted $FlowPilotToken
$remotePublicBaseUrl = Convert-ToShellSingleQuoted $PublicBaseUrl
$remoteQwenApiKey = Convert-ToShellSingleQuoted $QwenApiKey
$remoteQwenModel = Convert-ToShellSingleQuoted $QwenModel
$remoteNudgeMode = Convert-ToShellSingleQuoted $NudgeMode
$remoteBasicAuthUser = Convert-ToShellSingleQuoted $BasicAuthUser
$remoteBasicAuthPassword = Convert-ToShellSingleQuoted $BasicAuthPassword

$remote = @"
set -euxo pipefail
apt-get update
apt-get install -y unzip
rm -rf /opt/flowpilot
mkdir -p /opt/flowpilot
unzip -q /tmp/flowpilot.zip -d /opt/flowpilot
rm -f /tmp/flowpilot.zip
chmod +x /opt/flowpilot/cloud/alibaba/ecs/install_bamboo_focus_ecs.sh
FLOWPILOT_TOKEN=$remoteFlowPilotToken PUBLIC_BASE_URL=$remotePublicBaseUrl QWEN_API_KEY=$remoteQwenApiKey QWEN_MODEL=$remoteQwenModel NUDGE_MODE=$remoteNudgeMode BASIC_AUTH_USER=$remoteBasicAuthUser BASIC_AUTH_PASSWORD=$remoteBasicAuthPassword bash /opt/flowpilot/cloud/alibaba/ecs/install_bamboo_focus_ecs.sh
"@

ssh @sshArgs $target $remote

Write-Host "Done. Open: $PublicBaseUrl/app/"
