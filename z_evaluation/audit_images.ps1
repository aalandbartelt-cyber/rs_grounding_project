param([string]$DataRoot = (Join-Path $PSScriptRoot '..\rs_grounding_train_v1'))
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$rows = [System.Collections.Generic.List[object]]::new()
foreach ($split in @('train', 'val')) {
    foreach ($file in (Get-ChildItem -LiteralPath (Join-Path $DataRoot "$split\images") -Filter '*.jpg' -File | Sort-Object Name)) {
        $original = [System.Drawing.Bitmap]::new($file.FullName)
        $pixels = [System.Drawing.Bitmap]::new($original.Width, $original.Height, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
        $small = [System.Drawing.Bitmap]::new(9, 8)
        try {
            $graphics = [System.Drawing.Graphics]::FromImage($pixels)
            try {
                $graphics.CompositingMode = [System.Drawing.Drawing2D.CompositingMode]::SourceCopy
                $graphics.DrawImageUnscaled($original, 0, 0)
            } finally { $graphics.Dispose() }
            $rect = [System.Drawing.Rectangle]::new(0, 0, $pixels.Width, $pixels.Height)
            $locked = $pixels.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::ReadOnly, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
            try {
                $bytes = [byte[]]::new([Math]::Abs($locked.Stride) * $pixels.Height)
                [System.Runtime.InteropServices.Marshal]::Copy($locked.Scan0, $bytes, 0, $bytes.Length)
                $pixelHash = [Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant()
            } finally { $pixels.UnlockBits($locked) }
            $graphics = [System.Drawing.Graphics]::FromImage($small)
            try {
                $graphics.CompositingMode = [System.Drawing.Drawing2D.CompositingMode]::SourceCopy
                $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBilinear
                $graphics.DrawImage($pixels, [System.Drawing.Rectangle]::new(0, 0, 9, 8))
            } finally { $graphics.Dispose() }
            $bitString = [System.Text.StringBuilder]::new()
            for ($y = 0; $y -lt 8; $y++) {
                for ($x = 0; $x -lt 8; $x++) {
                    $a = $small.GetPixel($x, $y)
                    $b = $small.GetPixel($x + 1, $y)
                    $left = 299 * $a.R + 587 * $a.G + 114 * $a.B
                    $right = 299 * $b.R + 587 * $b.G + 114 * $b.B
                    [void]$bitString.Append($(if ($left -gt $right) { '1' } else { '0' }))
                }
            }
            $dhash = [Convert]::ToUInt64($bitString.ToString(), 2).ToString('x16')
            $rows.Add([ordered]@{key = "$split/$($file.BaseName)"; width = $original.Width; height = $original.Height; pixel_sha256 = $pixelHash; dhash64 = $dhash})
        } finally { $original.Dispose(); $pixels.Dispose(); $small.Dispose() }
    }
}
$rows | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'audit\image_fingerprints.json') -Encoding utf8
Write-Output "DECODED_IMAGE_FINGERPRINTS_OK: $($rows.Count)"
