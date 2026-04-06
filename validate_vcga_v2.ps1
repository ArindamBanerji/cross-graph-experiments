$driveOut = "G:\My Drive\public-files\gen-ai-roi\experiments"
$file     = "$driveOut\v_cga_frozen_two_stream_v2.json"

Write-Host "=== FILE ==="
$p = Get-Item $file
Write-Host "  Name : $($p.Name)"
Write-Host "  Size : $([math]::Round($p.Length / 1024, 1)) KB"

$raw = Get-Content $file -Raw | ConvertFrom-Json

# ── Top-level keys ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== TOP-LEVEL KEYS ==="
$raw.PSObject.Properties.Name

# ── Metadata scalars ──────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== METADATA ==="
foreach ($key in $raw.metadata.PSObject.Properties.Name) {
    $val = $raw.metadata.$key
    if ($val -isnot [System.Array] -and $val -isnot [System.Management.Automation.PSCustomObject]) {
        Write-Host "  ${key}: $val"
    }
}

# ── Record counts ─────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== RECORD COUNTS ==="
Write-Host "  stream_a: $($raw.stream_a.Count)"
Write-Host "  stream_b: $($raw.stream_b.Count)"

# ── Required field check (Stream A sample) ────────────────────────────────────
Write-Host ""
Write-Host "=== REQUIRED FIELD CHECK (Stream A) ==="
$required = @('seed','stream','day','centroid_frozen','enrichment_active',
              'graph_entity_count','graph_edge_count','accuracy_rolling_10','reached_85_pct')
$firstA = $raw.stream_a[0]
foreach ($field in $required) {
    $present = $firstA.PSObject.Properties.Name -contains $field
    $status  = if ($present) { "OK" } else { "MISSING" }
    Write-Host "  $status  $field"
}

# ── Stream A boundary records ─────────────────────────────────────────────────
Write-Host ""
Write-Host "=== STREAM A BOUNDARY RECORDS ==="
$lastA = $raw.stream_a[-1]
Write-Host "  First (seed=$($firstA.seed), day=$($firstA.day)): acc=$($firstA.accuracy_rolling_10), frozen=$($firstA.centroid_frozen), entities=$($firstA.graph_entity_count)"
Write-Host "  Last  (seed=$($lastA.seed),  day=$($lastA.day)):  acc=$($lastA.accuracy_rolling_10), frozen=$($lastA.centroid_frozen), entities=$($lastA.graph_entity_count)"

# ── Stream B boundary records ─────────────────────────────────────────────────
Write-Host ""
Write-Host "=== STREAM B BOUNDARY RECORDS ==="
$firstB = $raw.stream_b[0]
$lastB  = $raw.stream_b[-1]
Write-Host "  First (seed=$($firstB.seed), day=$($firstB.day)): acc=$($firstB.accuracy_rolling_10), frozen=$($firstB.centroid_frozen), entities=$($firstB.graph_entity_count)"
Write-Host "  Last  (seed=$($lastB.seed),  day=$($lastB.day)):  acc=$($lastB.accuracy_rolling_10), frozen=$($lastB.centroid_frozen), entities=$($lastB.graph_entity_count)"

# ── Entity count variance at Day 90 (Stream B) ────────────────────────────────
Write-Host ""
Write-Host "=== STREAM B ENTITY VARIANCE AT DAY 90 ==="
$b90 = $raw.stream_b | Where-Object { $_.day -eq 90 }
$entities = $b90 | ForEach-Object { $_.graph_entity_count }
$uniqueEntities = ($entities | Sort-Object -Unique)
Write-Host "  Records at Day 90 : $($b90.Count)  (expected 30)"
Write-Host "  entity_count min  : $([math]::Round(($entities | Measure-Object -Minimum).Minimum))"
Write-Host "  entity_count max  : $([math]::Round(($entities | Measure-Object -Maximum).Maximum))"
Write-Host "  Unique values     : $($uniqueEntities.Count)  (must be > 1 for G.3)"

# ── Stream A entity count flat check ─────────────────────────────────────────
Write-Host ""
Write-Host "=== STREAM A ENTITY COUNT (should be flat at 500) ==="
$a90 = $raw.stream_a | Where-Object { $_.day -eq 90 }
$aEntities = $a90 | ForEach-Object { $_.graph_entity_count }
$aUnique = ($aEntities | Sort-Object -Unique)
Write-Host "  Unique entity values at Day 90: $($aUnique.Count)  (expected 1)"
Write-Host "  Value: $($aUnique[0])"

# ── Stream B freeze flag check ────────────────────────────────────────────────
Write-Host ""
Write-Host "=== STREAM B FREEZE FLAG CHECK ==="
$bDay1  = $raw.stream_b | Where-Object { $_.seed -eq 0 -and $_.day -eq 1  }
$bDay45 = $raw.stream_b | Where-Object { $_.seed -eq 0 -and $_.day -eq 45 }
$bDay46 = $raw.stream_b | Where-Object { $_.seed -eq 0 -and $_.day -eq 46 }
$bDay90 = $raw.stream_b | Where-Object { $_.seed -eq 0 -and $_.day -eq 90 }
Write-Host "  seed=0 day=1  : centroid_frozen=$($bDay1.centroid_frozen)   (expected True)"
Write-Host "  seed=0 day=45 : centroid_frozen=$($bDay45.centroid_frozen)  (expected True)"
Write-Host "  seed=0 day=46 : centroid_frozen=$($bDay46.centroid_frozen)  (expected False)"
Write-Host "  seed=0 day=90 : centroid_frozen=$($bDay90.centroid_frozen)  (expected False)"
