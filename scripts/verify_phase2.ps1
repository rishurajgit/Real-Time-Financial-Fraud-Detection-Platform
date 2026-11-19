<#
.SYNOPSIS
    SentinelRisk Phase 2 End-to-End Verification Script (Windows PowerShell)

.DESCRIPTION
    Checks FastAPI availability, XGBoost model artifact readiness, Express availability,
    Express-to-FastAPI proxy connectivity, and prediction endpoint behavior.
    Never pretends inference succeeded when trained model artifacts are absent.
#>

param(
    [string]$FastApiUrl = "http://127.0.0.1:8000",
    [string]$ExpressUrl = "http://localhost:3000"
)

$ErrorActionPreference = "Stop"

function Write-Check {
    param([string]$Label, [string]$Status, [string]$Detail)
    Write-Host ("[{0}] {1}: {2}" -f $Status, $Label, $Detail)
}

Write-Host "=== SentinelRisk Phase 2 Verification ==="
Write-Host "FastAPI URL : $FastApiUrl"
Write-Host "Express URL : $ExpressUrl"
Write-Host ""

# 1. Check FastAPI /health
try {
    $fastHealth = Invoke-RestMethod -Uri "$FastApiUrl/health" -Method Get -TimeoutSec 5
    Write-Check "FastAPI Health" "PASS" ("status={0}, model_ready={1}, version={2}" -f $fastHealth.status, $fastHealth.model_ready, $fastHealth.model_version)
}
catch {
    Write-Check "FastAPI Health" "FAIL" "FastAPI is unreachable at $FastApiUrl. Start it with: uvicorn src.api.app:app --host 127.0.0.1 --port 8000"
    exit 1
}

# 2. Check FastAPI /model/info
$modelInfo = Invoke-RestMethod -Uri "$FastApiUrl/model/info" -Method Get -TimeoutSec 5
if ($modelInfo.ready -eq $true) {
    Write-Check "Model Readiness" "PASS" ("Loaded version={0}, features={1}" -f $modelInfo.model_version, $modelInfo.feature_count)
}
else {
    $missing = if ($modelInfo.missing_artifacts) { $modelInfo.missing_artifacts -join ", " } else { "model_bundle.joblib" }
    Write-Check "Model Readiness" "BLOCKED" "Model artifacts unavailable. Missing: $missing"
}

# 3. Check Express /api/health
try {
    $expressHealth = Invoke-RestMethod -Uri "$ExpressUrl/api/health" -Method Get -TimeoutSec 5
    Write-Check "Express Health" "PASS" ("ok={0}, service={1}" -f $expressHealth.ok, $expressHealth.service)
}
catch {
    Write-Check "Express Health" "FAIL" "Express is unreachable at $ExpressUrl. Start it with: npm run dev"
    exit 1
}

# 4. Check Express -> FastAPI proxy /api/v1/ml/health
try {
    $proxyHealth = Invoke-RestMethod -Uri "$ExpressUrl/api/v1/ml/health" -Method Get -TimeoutSec 5
    Write-Check "Proxy Connectivity" "PASS" ("proxied status={0}, model_ready={1}" -f $proxyHealth.status, $proxyHealth.model_ready)
}
catch {
    Write-Check "Proxy Connectivity" "FAIL" "Express failed to reach FastAPI via /api/v1/ml/health. Check ML_SERVICE_URL."
    exit 1
}

# 5. Send prediction request through Express proxy POST /api/v1/ml/predict
$payload = @{
    transaction_id  = "VERIFY-P2-$(Get-Date -Format 'HHmmss')"
    transaction_dt  = 86400
    transaction_amt = 645.50
    product_cd      = "C"
    card1           = 13926
    card4           = "visa"
    card6           = "credit"
    addr1           = 315
    addr2           = 87
    p_emaildomain   = "gmail.com"
    device_type     = "mobile"
    device_info     = "SM-G991B"
} | ConvertTo-Json

try {
    $pred = Invoke-RestMethod -Uri "$ExpressUrl/api/v1/ml/predict" -Method Post -ContentType "application/json" -Body $payload -TimeoutSec 10
    Write-Check "Prediction Proxy" "PASS" ("tx={0}, prob={1}, decision={2}, version={3}, latency={4}ms" -f $pred.transaction_id, $pred.xgboost_probability, $pred.decision, $pred.model_version, $pred.inference_latency_ms)

    # Verify /features/{transaction_id} lookup
    $feat = Invoke-RestMethod -Uri "$ExpressUrl/api/v1/ml/features/$($pred.transaction_id)" -Method Get -TimeoutSec 5
    Write-Check "Feature Lookup" "PASS" ("Retrieved cached features for {0}" -f $feat.transaction_id)

    Write-Host ""
    Write-Host "RESULT: END-TO-END VERIFIED (Real XGBoost inference succeeded through Express -> FastAPI)."
    exit 0
}
catch {
    $statusCode = $_.Exception.Response.StatusCode.value__
    if ($statusCode -eq 503 -and $modelInfo.ready -eq $false) {
        Write-Check "Prediction Proxy (503 Guard)" "PASS" "Correctly returned HTTP 503 model_not_ready while artifacts are missing (no fabricated fallback)."
        Write-Host ""
        Write-Host "RESULT: BLOCKED ON MODEL ARTIFACT — Train Phase 1 (`python -m src.models.train_xgboost`) and call POST /api/v1/ml/model/reload to complete live inference verification."
        exit 2
    }
    Write-Check "Prediction Proxy" "FAIL" ("Unexpected failure (HTTP {0}): {1}" -f $statusCode, $_.Exception.Message)
    exit 1
}
