# Download and Convert PaddleOCR 2.x Models to ONNX
# Based on official PaddleOCR documentation for version 2.x
# Downloads all 3 models: Detection, Recognition, and Classification

Write-Host "=" * 70
Write-Host "PaddleOCR 2.x ONNX Model Setup (All 3 Models)"
Write-Host "=" * 70

$projectRoot = (Resolve-Path "..\..\" ).ProviderPath
$inferenceDir = Join-Path $projectRoot "inference"

# Create directories
Write-Host "`nCreating directories..."
New-Item -ItemType Directory -Path $inferenceDir -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $inferenceDir "det_onnx") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $inferenceDir "rec_onnx") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $inferenceDir "cls_onnx") -Force | Out-Null

Set-Location $projectRoot

# URLs for model downloads
$detUrl = "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar"
$recUrl = "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar"
$clsUrl = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar"

Write-Host "`n" + "-" * 70
Write-Host "Step 1: Downloading Detection Model (PP-OCRv3)"
Write-Host "-" * 70

$detFile = Join-Path $inferenceDir "en_PP-OCRv3_det_infer.tar"
if (-not (Test-Path $detFile)) {
    Write-Host "Downloading: $detUrl"
    try {
        Invoke-WebRequest -Uri $detUrl -OutFile $detFile -UseBasicParsing
        Write-Host "✓ Downloaded successfully ($([math]::Round((Get-Item $detFile).Length / 1MB, 2)) MB)"
    } catch {
        Write-Host "✗ Download failed: $_"
        exit 1
    }
} else {
    Write-Host "✓ Detection model already downloaded"
}

Write-Host "Extracting detection model..."
Set-Location $inferenceDir
if (Test-Path "en_PP-OCRv3_det_infer") {
    Write-Host "✓ Already extracted"
} else {
    tar -xf en_PP-OCRv3_det_infer.tar
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Extracted successfully"
    } else {
        Write-Host "✗ Extraction failed"
        exit 1
    }
}
Set-Location $projectRoot

Write-Host "`n" + "-" * 70
Write-Host "Step 2: Downloading Recognition Model (PP-OCRv4)"
Write-Host "-" * 70

$recFile = Join-Path $inferenceDir "en_PP-OCRv4_rec_infer.tar"
if (-not (Test-Path $recFile)) {
    Write-Host "Downloading: $recUrl"
    try {
        Invoke-WebRequest -Uri $recUrl -OutFile $recFile -UseBasicParsing
        Write-Host "✓ Downloaded successfully ($([math]::Round((Get-Item $recFile).Length / 1MB, 2)) MB)"
    } catch {
        Write-Host "✗ Download failed: $_"
        exit 1
    }
} else {
    Write-Host "✓ Recognition model already downloaded"
}

Write-Host "Extracting recognition model..."
Set-Location $inferenceDir
if (Test-Path "en_PP-OCRv4_rec_infer") {
    Write-Host "✓ Already extracted"
} else {
    tar -xf en_PP-OCRv4_rec_infer.tar
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Extracted successfully"
    } else {
        Write-Host "✗ Extraction failed"
        exit 1
    }
}
Set-Location $projectRoot

Write-Host "`n" + "-" * 70
Write-Host "Step 3: Downloading Classification Model"
Write-Host "-" * 70

$clsFile = Join-Path $inferenceDir "ch_ppocr_mobile_v2.0_cls_infer.tar"
if (-not (Test-Path $clsFile)) {
    Write-Host "Downloading: $clsUrl"
    try {
        Invoke-WebRequest -Uri $clsUrl -OutFile $clsFile -UseBasicParsing
        Write-Host "✓ Downloaded successfully ($([math]::Round((Get-Item $clsFile).Length / 1MB, 2)) MB)"
    } catch {
        Write-Host "✗ Download failed: $_"
        exit 1
    }
} else {
    Write-Host "✓ Classification model already downloaded"
}

Write-Host "Extracting classification model..."
Set-Location $inferenceDir
if (Test-Path "ch_ppocr_mobile_v2.0_cls_infer") {
    Write-Host "✓ Already extracted"
} else {
    tar -xf ch_ppocr_mobile_v2.0_cls_infer.tar
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Extracted successfully"
    } else {
        Write-Host "✗ Extraction failed"
        exit 1
    }
}
Set-Location $projectRoot

Write-Host "`n" + "-" * 70
Write-Host "Step 4: Converting Detection Model to ONNX"
Write-Host "-" * 70

& .\.venv\Scripts\paddle2onnx.exe `
    --model_dir (Join-Path $inferenceDir "en_PP-OCRv3_det_infer") `
    --model_filename inference.pdmodel `
    --params_filename inference.pdiparams `
    --save_file (Join-Path $inferenceDir "det_onnx\model.onnx") `
    --opset_version 11 `
    --enable_onnx_checker True

if ($LASTEXITCODE -eq 0) {
    $sizeMB = [math]::Round((Get-Item (Join-Path $inferenceDir "det_onnx\model.onnx")).Length / 1MB, 2)
    Write-Host "✓ Detection model converted ($sizeMB MB)"
} else {
    Write-Host "✗ Detection conversion failed"
    exit 1
}

Write-Host "`n" + "-" * 70
Write-Host "Step 5: Converting Recognition Model to ONNX"
Write-Host "-" * 70

& .\.venv\Scripts\paddle2onnx.exe `
    --model_dir (Join-Path $inferenceDir "en_PP-OCRv4_rec_infer") `
    --model_filename inference.pdmodel `
    --params_filename inference.pdiparams `
    --save_file (Join-Path $inferenceDir "rec_onnx\model.onnx") `
    --opset_version 11 `
    --enable_onnx_checker True

if ($LASTEXITCODE -eq 0) {
    $sizeMB = [math]::Round((Get-Item (Join-Path $inferenceDir "rec_onnx\model.onnx")).Length / 1MB, 2)
    Write-Host "✓ Recognition model converted ($sizeMB MB)"
} else {
    Write-Host "✗ Recognition conversion failed"
    exit 1
}

Write-Host "`n" + "-" * 70
Write-Host "Step 6: Converting Classification Model to ONNX"
Write-Host "-" * 70

& .\.venv\Scripts\paddle2onnx.exe `
    --model_dir (Join-Path $inferenceDir "ch_ppocr_mobile_v2.0_cls_infer") `
    --model_filename inference.pdmodel `
    --params_filename inference.pdiparams `
    --save_file (Join-Path $inferenceDir "cls_onnx\model.onnx") `
    --opset_version 11 `
    --enable_onnx_checker True

if ($LASTEXITCODE -eq 0) {
    $sizeMB = [math]::Round((Get-Item (Join-Path $inferenceDir "cls_onnx\model.onnx")).Length / 1MB, 2)
    Write-Host "✓ Classification model converted ($sizeMB MB)"
} else {
    Write-Host "✗ Classification conversion failed"
    exit 1
}

Write-Host "`n" + "=" * 70
Write-Host "Setup Complete!"
Write-Host "=" * 70

Write-Host "`nONNX Model locations:"
Write-Host "  Detection:      " (Join-Path $inferenceDir "det_onnx\model.onnx")
Write-Host "  Recognition:    " (Join-Path $inferenceDir "rec_onnx\model.onnx")
Write-Host "  Classification: " (Join-Path $inferenceDir "cls_onnx\model.onnx")

Write-Host "`nNext steps:"
Write-Host "1. Run test_paddle_onnx.py to benchmark ONNX vs Paddle"
Write-Host "2. Expected: 2-3x speedup with ONNX Runtime"
Write-Host "3. Integration: Update minimal_perception.py to use ONNX models"
