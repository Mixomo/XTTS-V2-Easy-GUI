$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# Keep every runtime/cache inside the application folder.  The installer is
# deliberately idempotent: rerunning it repairs missing packages without
# deleting a healthy venv or any user datasets/models.
$UvVersion='0.11.33'
$PythonVersion='3.11.15'
$TorchVersion='2.8.0'
$Runtime=Join-Path $Root '.runtime'
$UvDir=Join-Path $Runtime 'uv'
$UvExe=Join-Path $UvDir 'uv.exe'
$Venv=Join-Path $Root '.venv'
$PythonExe=Join-Path $Venv 'Scripts\python.exe'
$Downloads=Join-Path $Runtime 'downloads'
$UvCache=Join-Path $Runtime 'uv-cache'
$Temp=Join-Path $Runtime 'temp'
$UvPython=Join-Path $Runtime 'python'
$HfHome=Join-Path $Runtime 'cache\huggingface'
foreach($d in @($Runtime,$UvDir,$Downloads,$UvCache,$Temp,$UvPython,$HfHome,(Join-Path $HfHome 'xet'),(Join-Path $Root 'models'),(Join-Path $Root 'outputs'),(Join-Path $Root 'samples'),(Join-Path $Root 'datasets'),(Join-Path $Root 'projects'),(Join-Path $Root 'training'),(Join-Path $Root 'base_models'))){New-Item -ItemType Directory -Force -Path $d|Out-Null}
$env:UV_CACHE_DIR=$UvCache;$env:UV_PYTHON_INSTALL_DIR=$UvPython;$env:UV_PYTHON_PREFERENCE='managed';$env:UV_NO_CACHE='1';$env:PIP_NO_CACHE_DIR='1';$env:UV_LINK_MODE='copy';$env:PYTHONUTF8='1';$env:PYTHONDONTWRITEBYTECODE='1';$env:TEMP=$Temp;$env:TMP=$Temp;$env:HF_HOME=$HfHome;$env:HF_XET_CACHE=Join-Path $HfHome 'xet'
function Section($t){Write-Host "`n=== $t ===" -ForegroundColor Cyan}
function DL($url,$dst){
  if(Test-Path $dst){return}
  $curlOk=$false
  try {
    & curl.exe --fail --location --retry 4 --retry-delay 2 --output $dst $url
    $curlOk=($LASTEXITCODE -eq 0 -and (Test-Path $dst) -and ((Get-Item $dst).Length -gt 0))
  } catch { $curlOk=$false }
  if(-not $curlOk){
    Remove-Item -LiteralPath $dst -Force -ErrorAction SilentlyContinue
    $webOk=$false
    try {
      Invoke-WebRequest -Uri $url -OutFile $dst -UseBasicParsing -ErrorAction Stop
      $webOk=(Test-Path $dst) -and ((Get-Item $dst).Length -gt 0)
    } catch { $webOk=$false }
    if(-not $webOk){
      Remove-Item -LiteralPath $dst -Force -ErrorAction SilentlyContinue
      $python=Get-Command python.exe -ErrorAction SilentlyContinue
      if($python){
        try {
          & $python.Source -c "import sys,urllib.request; urllib.request.urlretrieve(sys.argv[1],sys.argv[2])" $url $dst
          $webOk=($LASTEXITCODE -eq 0 -and (Test-Path $dst) -and ((Get-Item $dst).Length -gt 0))
        } catch { $webOk=$false }
      }
    }
    if(-not $webOk){
      Remove-Item -LiteralPath $dst -Force -ErrorAction SilentlyContinue
      throw "Download failed with curl, Invoke-WebRequest and Python urllib: $url"
    }
  }
  if(!(Test-Path $dst) -or (Get-Item $dst).Length -le 0){throw "Download produced an empty file: $url"}
}
function Run($file,[string[]]$Arguments){& $file @Arguments;if($LASTEXITCODE -ne 0){throw "Command failed ($LASTEXITCODE): $file $($Arguments -join ' ')"}}

Section '1/6 - Local uv'
if(!(Test-Path $UvExe)){
  $z=Join-Path $Downloads "uv-$UvVersion.zip"
  DL "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip" $z
  tar.exe -xf $z -C $UvDir
  if($LASTEXITCODE -ne 0){throw 'Could not extract uv.'}
  Remove-Item -LiteralPath $z -Force -ErrorAction SilentlyContinue
}
if(!(Test-Path $UvExe)){throw "uv executable was not found at $UvExe"}
Run $UvExe @('self','version')

Section '2/6 - Local Python + venv'
if(!(Test-Path $PythonExe)){
  Run $UvExe @('python','install',$PythonVersion,'--no-cache','--no-bin','--no-registry')
  Run $UvExe @('venv','--python',$PythonVersion,'--no-cache',$Venv)
}
if(!(Test-Path $PythonExe)){throw "Python venv was not created at $PythonExe"}
$pythonVersionText = (& $PythonExe -c "import sys; print('.'.join(map(str,sys.version_info[:3])))").Trim()
if($LASTEXITCODE -ne 0 -or $pythonVersionText -notmatch '^3\.11\.') {throw "Expected Python 3.11 in the project venv, got '$pythonVersionText'."}

Section '3/6 - PyTorch CUDA 12.8'
Run $UvExe @('pip','install','--python',$PythonExe,'--no-cache',"torch==$TorchVersion","torchaudio==$TorchVersion",'torchvision==0.23.0','--index-url','https://download.pytorch.org/whl/cu128')

Section '4/6 - XTTS Easy GUI stack'
Run $UvExe @('pip','install','--python',$PythonExe,'--no-cache',
  'coqui-tts==0.27.2','coqui-tts-trainer==0.3.1',
  'prodigyopt==1.1.2',
  'gradio==5.49.1','faster-whisper==1.2.0','huggingface-hub[hf-xet]>=0.34,<1',
  'tokenizers>=0.21,<1','soundfile>=0.13','numpy==1.26.4','pandas>=2.2,<3',
  'tensorboard>=2.19,<3','psutil>=7,<8','librosa>=0.11,<1','pydub>=0.25,<1',
  'fugashi[unidic-lite]>=1.4,<2','transformers>=4.52.1,<4.56')

Section '5/6 - Runtime and GUI smoke test'
Run $PythonExe @('-c',@"
import sys, torch, gradio, TTS, prodigyopt
from tokenizers import Tokenizer
from TTS.tts.models.xtts import Xtts
import xtts_backend
print('Python', sys.version.split()[0])
print('Torch', torch.__version__, 'CUDA', torch.version.cuda, 'GPU', torch.cuda.is_available())
print('Gradio', gradio.__version__, 'TTS', getattr(TTS, '__version__', '?'))
print('Prodigy import OK')
print('XTTS import OK; backend root:', xtts_backend.ROOT)
"@)

Section '6/6 - Optional tools'
if(Get-Command ffmpeg.exe -ErrorAction SilentlyContinue){Write-Host 'ffmpeg detected: extended audio formats enabled.' -ForegroundColor DarkGray}else{Write-Host 'ffmpeg not found: WAV/FLAC remain supported; install ffmpeg for MP3/M4A dataset inputs.' -ForegroundColor Yellow}
Write-Host "`n[OK] Installation complete. Review the XTTS/CPML model terms, then run start.bat." -ForegroundColor Green
