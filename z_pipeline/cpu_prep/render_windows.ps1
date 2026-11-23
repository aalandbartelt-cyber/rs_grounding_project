param([Parameter(Mandatory=$true)][string]$JobsPath, [Parameter(Mandatory=$true)][string]$ReportPath, [int]$BudgetMiB=256)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$timer = [System.Diagnostics.Stopwatch]::StartNew()
$jobs = Get-Content -Raw -Encoding UTF8 -LiteralPath $JobsPath | ConvertFrom-Json
$jpegCodec = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object MimeType -EQ 'image/jpeg'
$completed = 0
foreach ($job in $jobs) {
    if ([System.Diagnostics.Process]::GetCurrentProcess().WorkingSet64 -gt $BudgetMiB * 1MB) { throw 'Image renderer exceeded its checkpoint memory budget' }
    $source = $null
    $bitmap = $null
    try {
        $source = [System.Drawing.Image]::FromFile([string]$job.source)
        $bitmap = [System.Drawing.Bitmap]::new($source.Width, $source.Height, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
        $g = [System.Drawing.Graphics]::FromImage($bitmap)
        try { $g.DrawImageUnscaled($source,0,0) } finally { $g.Dispose() }
        switch ([string]$job.operation) {
            'original' { }
            'flip_x' { $bitmap.RotateFlip([System.Drawing.RotateFlipType]::RotateNoneFlipX) }
            'rotate90_ccw' { $bitmap.RotateFlip([System.Drawing.RotateFlipType]::Rotate270FlipNone) }
            'crop_keep_all' {
                $v = $job.crop
                $rect = [System.Drawing.Rectangle]::new([int]$v[0],[int]$v[1],[int]$v[2],[int]$v[3])
                $next = $bitmap.Clone($rect,[System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
                $bitmap.Dispose(); $bitmap = $next
            }
            'brightness_095' {
                $next = [System.Drawing.Bitmap]::new($bitmap.Width,$bitmap.Height)
                $matrix = [System.Drawing.Imaging.ColorMatrix]::new()
                $matrix.Matrix00 = 0.95; $matrix.Matrix11 = 0.95; $matrix.Matrix22 = 0.95
                $attrs = [System.Drawing.Imaging.ImageAttributes]::new()
                $g = [System.Drawing.Graphics]::FromImage($next)
                try {
                    $attrs.SetColorMatrix($matrix)
                    $rect = [System.Drawing.Rectangle]::new(0,0,$next.Width,$next.Height)
                    $g.DrawImage($bitmap,$rect,0,0,$bitmap.Width,$bitmap.Height,[System.Drawing.GraphicsUnit]::Pixel,$attrs)
                } finally { $g.Dispose(); $attrs.Dispose() }
                $bitmap.Dispose(); $bitmap = $next
            }
            'resample_blur_085' {
                $small = [System.Drawing.Bitmap]::new([Math]::Max(1,[int]($bitmap.Width*0.85)),[Math]::Max(1,[int]($bitmap.Height*0.85)))
                $g = [System.Drawing.Graphics]::FromImage($small)
                try { $g.InterpolationMode=[System.Drawing.Drawing2D.InterpolationMode]::HighQualityBilinear; $g.DrawImage($bitmap,0,0,$small.Width,$small.Height) } finally { $g.Dispose() }
                $next = [System.Drawing.Bitmap]::new($bitmap.Width,$bitmap.Height)
                $g = [System.Drawing.Graphics]::FromImage($next)
                try { $g.InterpolationMode=[System.Drawing.Drawing2D.InterpolationMode]::HighQualityBilinear; $g.DrawImage($small,0,0,$next.Width,$next.Height) } finally { $g.Dispose(); $small.Dispose() }
                $bitmap.Dispose(); $bitmap=$next
            }
            'jpeg_80' { }
            default { throw ('Unsupported operation: ' + $job.operation) }
        }
        if ($bitmap.Width -ne [int]$job.width -or $bitmap.Height -ne [int]$job.height) { throw 'Pixel and coordinate dimensions differ' }
        $destination = [string]$job.destination
        $parent = [System.IO.Path]::GetDirectoryName($destination)
        [System.IO.Directory]::CreateDirectory($parent) | Out-Null
        $encoding = [System.Drawing.Imaging.EncoderParameters]::new(1)
        try {
            $quality = if ($job.operation -eq 'jpeg_80') { 80L } else { 95L }
            $encoding.Param[0] = [System.Drawing.Imaging.EncoderParameter]::new([System.Drawing.Imaging.Encoder]::Quality,$quality)
            if ($job.operation -eq 'original') { [System.IO.File]::Copy([string]$job.source,$destination,$false) }
            else { $bitmap.Save($destination,$jpegCodec,$encoding) }
        } finally { $encoding.Dispose() }
        if ($job.overlay) {
            $g = [System.Drawing.Graphics]::FromImage($bitmap)
            $pen = [System.Drawing.Pen]::new([System.Drawing.Color]::Lime,2)
            try {
                foreach ($box in $job.boxes) { $g.DrawRectangle($pen,[single]$box[0],[single]$box[1],[single]$box[2],[single]$box[3]) }
                $bitmap.Save([string]$job.overlay,[System.Drawing.Imaging.ImageFormat]::Png)
            } finally { $g.Dispose(); $pen.Dispose() }
        }
        $completed++
        if ($completed % 40 -eq 0) { Write-Output ('Rendered ' + $completed + '/' + $jobs.Count) }
    } finally { if ($bitmap) { $bitmap.Dispose() }; if ($source) { $source.Dispose() } }
    if ($completed % 20 -eq 0) { [System.GC]::Collect() }
}
$process = [System.Diagnostics.Process]::GetCurrentProcess()
$report = @{ jobs=$completed; seconds=[Math]::Round($timer.Elapsed.TotalSeconds,3); peak_rss_mib=[Math]::Round($process.PeakWorkingSet64/1MB,2); rss_mib=[Math]::Round($process.WorkingSet64/1MB,2); checkpoint_budget_mib=$BudgetMiB; renderer='Windows System.Drawing'; scope='This renderer process only; no model or GPU loaded' }
$report | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath $ReportPath
Write-Output ($report | ConvertTo-Json -Compress)
