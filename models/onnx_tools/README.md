# ONNX Tools

This directory contains helper scripts for downloading and converting PaddleOCR models to ONNX and small benchmarks.

Files:

- `setup_paddleocr2_onnx.ps1` - PowerShell script to download PaddleOCR 2.x models and convert to ONNX
- `test_paddle_onnx.py` - Benchmark script comparing PaddleOCR vs ONNX
- `test_multithreading_onnx.py` - Multi-threading benchmark demonstrating parallel OCR speedups

Usage:

1. Ensure you have the project virtualenv activated.
2. Run `.
un_setup.ps1` or `.
un_setup.bat` from the project root to prepare models.
