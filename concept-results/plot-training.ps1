#Requires -Version 5.1
[CmdletBinding()]
param([string]$ResultsPath = '', [string]$OutputPath = '')

$ErrorActionPreference = 'Stop'
if (-not $ResultsPath) { $ResultsPath = Join-Path $PSScriptRoot 'training-results.json' }
if (-not $OutputPath) { $OutputPath = Join-Path $PSScriptRoot 'training-comparison.png' }
$results = Get-Content -LiteralPath $ResultsPath -Raw | ConvertFrom-Json
Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing

$colors = @{
    fedavg = [System.Drawing.ColorTranslator]::FromHtml('#2463A6')
    fedbif = [System.Drawing.ColorTranslator]::FromHtml('#C25316')
    target = [System.Drawing.ColorTranslator]::FromHtml('#555F69')
}
$ink = [System.Drawing.ColorTranslator]::FromHtml('#182C3C')
$muted = [System.Drawing.ColorTranslator]::FromHtml('#536778')
$grid = [System.Drawing.ColorTranslator]::FromHtml('#DFE5EB')
$axisFont = New-Object System.Drawing.Font('Segoe UI', 11)
$titleFont = New-Object System.Drawing.Font('Segoe UI', 15, [System.Drawing.FontStyle]::Bold)
$headerFont = New-Object System.Drawing.Font('Segoe UI', 22, [System.Drawing.FontStyle]::Bold)
$noteFont = New-Object System.Drawing.Font('Segoe UI', 12)
$labelFont = New-Object System.Drawing.Font('Segoe UI', 12, [System.Drawing.FontStyle]::Bold)
$brush = New-Object System.Drawing.SolidBrush($ink)
$mutedBrush = New-Object System.Drawing.SolidBrush($muted)

function New-Plot {
    param([string]$Metric, [string]$Title, [string]$YLabel,
          [double]$Maximum, [double]$Target, [double]$Scale)
    $chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
    $chart.Width = 820
    $chart.Height = 570
    $chart.BackColor = [System.Drawing.Color]::White
    $chart.AntiAliasing = 'All'
    $chart.TextAntiAliasingQuality = 'High'
    $area = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea('plot')
    $area.BackColor = [System.Drawing.Color]::White
    $area.AxisX.Minimum = 0
    $area.AxisX.Maximum = [double]$results.settings.rounds
    $area.AxisX.Interval = 1
    $area.AxisX.Title = 'Federated round (0 = before training)'
    $area.AxisY.Minimum = 0
    $area.AxisY.Maximum = $Maximum
    $area.AxisY.Title = $YLabel
    $area.AxisY.Interval = if ($Metric -eq 'loss') { 0.1 } else { 20 }
    $area.AxisY.LabelStyle.Format = if ($Metric -eq 'loss') { '0.0' } else { '0' }
    foreach ($axis in @($area.AxisX, $area.AxisY)) {
        $axis.TitleFont = $axisFont
        $axis.LabelStyle.Font = $axisFont
        $axis.TitleForeColor = $ink
        $axis.LabelStyle.ForeColor = $ink
        $axis.LineColor = $muted
        $axis.MajorTickMark.LineColor = $muted
        $axis.MajorGrid.LineColor = $grid
        $axis.MajorGrid.LineDashStyle = 'Solid'
    }
    $area.AxisX.MajorGrid.Enabled = $false
    [void]$chart.ChartAreas.Add($area)
    $titleObject = New-Object System.Windows.Forms.DataVisualization.Charting.Title
    $titleObject.Text = $Title
    $titleObject.Font = $titleFont
    $titleObject.ForeColor = $ink
    [void]$chart.Titles.Add($titleObject)
    $legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend('legend')
    $legend.Docking = 'Bottom'
    $legend.Alignment = 'Center'
    $legend.LegendStyle = 'Column'
    $legend.Font = $axisFont
    $legend.ForeColor = $ink
    [void]$chart.Legends.Add($legend)

    # This is a mathematical limit, not a measured or invented training curve.
    $ideal = New-Object System.Windows.Forms.DataVisualization.Charting.Series('Ideal limit')
    $ideal.ChartType = 'Line'
    $ideal.Color = $colors.target
    $ideal.BorderDashStyle = 'Dash'
    $ideal.BorderWidth = 2
    $ideal.LegendText = if ($Metric -eq 'loss') { 'Ideal limit: zero loss' } else { 'Ideal ceiling: 100% accuracy' }
    [void]$ideal.Points.AddXY(0, $Target)
    [void]$ideal.Points.AddXY([double]$results.settings.rounds, $Target)
    [void]$chart.Series.Add($ideal)

    foreach ($method in @('fedavg', 'fedbif')) {
        $entry = $results.methods.$method
        $series = New-Object System.Windows.Forms.DataVisualization.Charting.Series($method)
        $series.ChartType = 'Line'
        $series.Color = $colors[$method]
        $series.BorderWidth = 3
        $series.MarkerStyle = if ($method -eq 'fedavg') { 'Circle' } else { 'Diamond' }
        $series.MarkerSize = if ($method -eq 'fedavg') { 13 } else { 7 }
        $series.MarkerColor = if ($method -eq 'fedavg') { [System.Drawing.Color]::White } else { $colors[$method] }
        $series.MarkerBorderColor = $colors[$method]
        $series.MarkerBorderWidth = 2
        $name = if ($method -eq 'fedavg') { 'FedAvg' } else { 'FedBiF' }
        $series.LegendText = if ($Metric -eq 'loss') {
            '{0}: final loss {1:0.0000}' -f $name, $entry.final.loss
        } else {
            '{0}: final accuracy {1:0}%' -f $name, ($entry.final.synthetic_accuracy * 100)
        }
        [void]$series.Points.AddXY(0, ([double]$entry.initial.$Metric * $Scale))
        foreach ($point in $entry.history) {
            [void]$series.Points.AddXY([double]$point.round, ([double]$point.$Metric * $Scale))
        }
        [void]$chart.Series.Add($series)
    }
    return ,$chart
}

$lossPlot = $null
$accuracyPlot = $null
$canvas = $null
$graphics = $null
try {
    $lossPlot = New-Plot 'loss' 'Loss falls toward the ideal limit' 'Test cross-entropy loss (lower is better)' 0.75 0 1
    $accuracyPlot = New-Plot 'synthetic_accuracy' 'Accuracy reaches the ideal ceiling' 'Synthetic test accuracy (%)' 105 100 100
    $canvas = New-Object System.Drawing.Bitmap(1680, 840)
    $canvas.SetResolution(144, 144)
    $graphics = [System.Drawing.Graphics]::FromImage($canvas)
    $graphics.Clear([System.Drawing.Color]::White)
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $graphics.DrawString('FedBiF concept run: measured learning vs ideal limits', $headerFont, $brush, 34, 22)
    $description = '{0} clients | {1} generated training images | {2} held-out images | {3} rounds | one seed' -f `
        $results.settings.clients, $results.settings.samples, $results.settings.test_samples, $results.settings.rounds
    $graphics.DrawString($description, $noteFont, $mutedBrush, 36, 73)
    $streamLoss = New-Object System.IO.MemoryStream
    $streamAccuracy = New-Object System.IO.MemoryStream
    $lossImage = $null
    $accuracyImage = $null
    try {
        $lossPlot.SaveImage($streamLoss, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
        $accuracyPlot.SaveImage($streamAccuracy, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
        $streamLoss.Position = 0
        $streamAccuracy.Position = 0
        $lossImage = [System.Drawing.Image]::FromStream($streamLoss)
        $accuracyImage = [System.Drawing.Image]::FromStream($streamAccuracy)
        $graphics.DrawImage($lossImage, 20, 112, 820, 570)
        $graphics.DrawImage($accuracyImage, 840, 112, 820, 570)
    } finally {
        if ($lossImage) { $lossImage.Dispose() }
        if ($accuracyImage) { $accuracyImage.Dispose() }
        $streamLoss.Dispose()
        $streamAccuracy.Dispose()
    }
    $graphics.DrawString('Both accuracy curves overlap at 100% from round 1; the loss curves still differ.', $labelFont, $brush, 36, 714)
    $graphics.DrawString('Dashed lines are mathematical ideals, not expected learning curves or paper benchmarks.', $noteFont, $mutedBrush, 36, 754)
    $graphics.DrawString('One small synthetic run: real-data accuracy, communication savings and reliable speedups remain untested.', $noteFont, $mutedBrush, 36, 787)
    $canvas.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
    Write-Output "Saved $OutputPath"
} finally {
    if ($graphics) { $graphics.Dispose() }
    if ($canvas) { $canvas.Dispose() }
    if ($lossPlot) { $lossPlot.Dispose() }
    if ($accuracyPlot) { $accuracyPlot.Dispose() }
    foreach ($resource in @($axisFont, $titleFont, $headerFont, $noteFont, $labelFont, $brush, $mutedBrush)) {
        $resource.Dispose()
    }
}
