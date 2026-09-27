param(
  [Parameter(Mandatory = $true)]
  [string]$SeedCsv,

  [Parameter(Mandatory = $true)]
  [string]$Model,

  [string]$OutDir = "data",
  [string]$Device = "auto",
  [string]$TorchDtype = "auto",
  [switch]$UseChatTemplate,
  [string]$HfEndpoint = "",
  [int]$NumCounters = 1,
  [int]$Seed = 42,
  [string]$DiscoveryTemplates = "main_v1,discovery_v2,discovery_v3,discovery_v4",
  [string]$HeldoutTemplates = "heldout_v1,heldout_v2,heldout_v3,heldout_v4",
  [string]$GroupKey = "subject_id"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$env:PYTHONPATH = "src"
if ($HfEndpoint -ne "") {
  $env:HF_ENDPOINT = $HfEndpoint
}

$IntermediateDir = Join-Path $OutDir "01_intermediate"
$PromptDir = Join-Path $OutDir "02_prompts"
$SplitDir = Join-Path $OutDir "03_splits"
$ReportDir = Join-Path $OutDir "04_reports"

New-Item -ItemType Directory -Force -Path $IntermediateDir, $PromptDir, $SplitDir, $ReportDir | Out-Null

Write-Host "[clean-micro] build"
py -3.11 -m screscomp.cli.build_dataset `
  --seed_csv $SeedCsv `
  --out_dir $OutDir `
  --num_counters $NumCounters `
  --seed $Seed `
  --template_family main_v1 `
  --out_funnel_csv (Join-Path $ReportDir "sample_funnel_build.csv")

Write-Host "[clean-micro] probe with discovery-template stability"
$ProbeArgs = @(
  "-m", "screscomp.cli.run_prior_probe",
  "--in_jsonl", (Join-Path $IntermediateDir "04_prompt_ready_pairs.jsonl"),
  "--model", $Model,
  "--device", $Device,
  "--torch_dtype", $TorchDtype,
  "--out_retained", (Join-Path $IntermediateDir "05_retained_pairs.jsonl"),
  "--out_excluded", (Join-Path $IntermediateDir "05_excluded_pairs.jsonl"),
  "--out_funnel_csv", (Join-Path $ReportDir "sample_funnel_probe.csv"),
  "--out_exclusion_stats_csv", (Join-Path $ReportDir "exclusion_stats.csv"),
  "--stability_template_families", $DiscoveryTemplates,
  "--require_template_stability",
  "--out_template_scores_jsonl", (Join-Path $ReportDir "template_prior_scores.jsonl")
)
if ($UseChatTemplate) {
  $ProbeArgs += "--use_chat_template"
}
py -3.11 @ProbeArgs

Write-Host "[clean-micro] split"
py -3.11 -m screscomp.cli.make_splits `
  --retained_jsonl (Join-Path $IntermediateDir "05_retained_pairs.jsonl") `
  --out_manifest (Join-Path $SplitDir "split_manifest.json") `
  --group_key $GroupKey `
  --seed $Seed `
  --discovery_template_families $DiscoveryTemplates `
  --heldout_template_families $HeldoutTemplates `
  --validation_template_mode discovery_only

Write-Host "[clean-micro] render all templates allowed by split"
py -3.11 -m screscomp.cli.render_prompts `
  --retained_jsonl (Join-Path $IntermediateDir "05_retained_pairs.jsonl") `
  --split_manifest (Join-Path $SplitDir "split_manifest.json") `
  --out_dir $PromptDir `
  --template_mode all_by_split

Write-Host "[clean-micro] audit"
py -3.11 -m screscomp.cli.audit_prompts `
  --rendered_jsonl (Join-Path $PromptDir "09_rendered_samples.jsonl") `
  --split_manifest (Join-Path $SplitDir "split_manifest.json") `
  --out_json (Join-Path $ReportDir "prompt_audit.json") `
  --strict

Write-Host "[clean-micro] done"
Write-Host "Rendered samples: $(Join-Path $PromptDir '09_rendered_samples.jsonl')"
Write-Host "Audit report:     $(Join-Path $ReportDir 'prompt_audit.json')"
