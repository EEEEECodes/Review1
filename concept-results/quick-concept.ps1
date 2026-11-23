$ErrorActionPreference = 'Stop'

# Numerical port of plugin/bif.py's bit decomposition; no ML training.
$timer = [System.Diagnostics.Stopwatch]::StartNew()
$rng = [System.Random]::new(1234)
$weights = @(-0.15, -0.13, -0.11, -0.09, -0.07, -0.05, -0.03, -0.01,
              0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15)
$bitWidth = 4
$selectedBit = 2
$mask = 1 -shl $selectedBit
$maxRange = ($weights | ForEach-Object { [Math]::Abs($_) } | Measure-Object -Maximum).Maximum
$alpha = $maxRange / [Math]::Pow(2, $bitWidth - 1)
$rows = @()

for ($i = 0; $i -lt $weights.Count; $i++) {
    $q = [int][Math]::Floor(($weights[$i] + $maxRange) / $alpha + $rng.NextDouble())
    $q = [Math]::Min([Math]::Max($q, 0), [int][Math]::Pow(2, $bitWidth) - 1)
    $selectedValue = ($q -band $mask) -shr $selectedBit
    $frozen = $q - ($selectedValue * $mask)
    $remainingWeight = ($frozen * $alpha) - $maxRange
    $reconstructed = $remainingWeight + $selectedValue * $alpha * $mask

    # Illustrate a possible learned-bit change by toggling half the selected bits.
    # This toggle is synthetic; it is not a gradient or training result.
    $newBit = if (($i % 2) -eq 0) { 1 - $selectedValue } else { $selectedValue }
    $updated = $frozen + $newBit * $mask
    $rows += [pscustomobject]@{
        original_weight = $weights[$i]
        quantized_integer = $q
        reconstructed_weight = [Math]::Round($reconstructed, 8)
        quantization_absolute_error = [Math]::Abs($reconstructed - $weights[$i])
        updated_integer = $updated
        frozen_bits_preserved = (($q -band (15 -bxor $mask)) -eq ($updated -band (15 -bxor $mask)))
        reconstruction_correct = ([Math]::Abs($reconstructed - ($q * $alpha - $maxRange)) -lt 1e-12)
    }
}

$preserved = @($rows | Where-Object { $_.frozen_bits_preserved }).Count
$reconstructedCount = @($rows | Where-Object { $_.reconstruction_correct }).Count
$changed = @($rows | Where-Object { $_.quantized_integer -ne $_.updated_integer }).Count
if ($preserved -ne $weights.Count -or $reconstructedCount -ne $weights.Count) {
    throw 'Bit-freezing or reconstruction check failed.'
}
$timer.Stop()
$result = [ordered]@{
    scope = 'PowerShell numerical concept demo of plugin/bif.py; not original Python execution or ML training'
    seed = 1234
    weights = $weights.Count
    quantization_bits = $bitWidth
    selected_trainable_bit_zero_based = $selectedBit
    quantization_step = $alpha
    reconstructed_correctly = $reconstructedCount
    weights_with_selected_bit_changed = $changed
    weights_with_all_other_bits_preserved = $preserved
    max_quantization_absolute_error = ($rows | Measure-Object -Property quantization_absolute_error -Maximum).Maximum
    elapsed_ms = [Math]::Round($timer.Elapsed.TotalMilliseconds, 2)
    training_accuracy = $null
    measured_network_savings = $null
    details = $rows
}
$json = $result | ConvertTo-Json -Depth 5
$json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'results.json') -Encoding UTF8
$result.Remove('details')
$result | ConvertTo-Json -Depth 3
