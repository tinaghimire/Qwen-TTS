# Import Path Fix

## Problem

When running `uv run qwen_tts_server.py`, the following error occurred:

```
ModuleNotFoundError: No module named 'Qwen3_TTS'
```

## Root Cause

The import statement was:
```python
from Qwen3_TTS.qwen_tts import Qwen3TTSModel
```

However, the directory name is `Qwen3-TTS` (with hyphens), not `Qwen3_TTS` (with underscores).

**Python cannot import modules with hyphens in their names** because hyphens are not valid in Python identifiers.

## Solution

Added the Qwen3-TTS directory to the Python path and used the correct import:

### Before
```python
from Qwen3_TTS.qwen_tts import Qwen3TTSModel
```

### After
```python
import sys
from pathlib import Path

# Add Qwen3-TTS to Python path
qwen_tts_path = Path(__file__).parent / "Qwen3-TTS"
if qwen_tts_path.exists():
    sys.path.insert(0, str(qwen_tts_path))

from qwen_tts import Qwen3TTSModel
```

## Files Updated

1. **`qwen_tts_server.py`** - Fixed import statement
2. **`test_setup.py`** - Updated Qwen3-TTS import test

## Why This Works

### The Issue
- Directory name: `Qwen3-TTS` (hyphens)
- Python import: `from Qwen3_TTS` (underscores)
- Result: Module not found

### The Fix
1. Add the directory to `sys.path`
2. Import using the actual module name: `from qwen_tts import Qwen3TTSModel`
3. The `qwen_tts` module is inside the `Qwen3-TTS` directory

### Directory Structure
```
Qwen3-TTS-finetuning/
├── Qwen3-TTS/              # Directory with hyphens
│   ├── qwen_tts/           # Actual Python module
│   │   ├── __init__.py
│   │   └── inference/
│   └── ...
├── qwen_tts_server.py      # Imports from qwen_tts
└── ...
```

## Alternative Solutions

### Option 1: Rename Directory (Not Recommended)
```bash
mv Qwen3-TTS Qwen3_TTS
```

**Trade-off**: Breaks git history and may affect other tools.

### Option 2: Use PYTHONPATH Environment Variable
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/Qwen3-TTS-finetuning/Qwen3-TTS"
```

**Trade-off**: Requires setting environment variable every time.

### Option 3: Create a Symlink (Not Recommended)
```bash
ln -s Qwen3-TTS Qwen3_TTS
```

**Trade-off**: Confusing, may cause issues on Windows.

### Option 4: Use sys.path (Chosen Solution) ✅
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "Qwen3-TTS"))
```

**Benefits**:
- Works across all platforms
- No external configuration needed
- Clear and explicit
- Doesn't modify directory structure

## Verification

After the fix, the server should start successfully:

```bash
$ uv run qwen_tts_server.py
Loading Qwen3-TTS Base model...
Starting Qwen3-TTS server on http://0.0.0.0:8000
Model size: 1.7B
Voices directory: /home/ml/workspaces/kristina/Qwen3-TTS-finetuning/voices

WebSocket endpoints:
  - ws://0.0.0.0:8000/ws/voice-clone/{voice_id}
  - ws://0.0.0.0:8000/v1/text-to-speech/{voice_id}/stream-input

HTTP endpoint:
  - POST http://0.0.0.0:8000/voice-clone/{voice_id}
```

## Testing

Run the setup verification to ensure everything works:

```bash
uv run test_setup.py
```

Expected output:
```
Testing Python version...
  Python version: 3.11.x
  ✓ Python version is compatible (3.9-3.12)

Testing imports...
  ✓ FastAPI
  ✓ Uvicorn
  ✓ WebSockets
  ✓ Librosa
  ✓ SciPy
  ✓ NumPy
  ✓ PyTorch
  ✓ SoundFile
  ✓ HuggingFace Hub
  ✓ Transformers

Testing CUDA availability...
  ✓ CUDA is available

Testing Qwen3-TTS import...
  ✓ Qwen3-TTS can be imported

Summary
✓ All dependencies are installed correctly!
✓ CUDA is available for GPU acceleration
```

## Best Practices

### When Importing from Local Modules

1. **Use absolute paths** when possible
2. **Add to sys.path** at the top of the file
3. **Check if directory exists** before adding to path
4. **Document the import** with comments

### Example Pattern
```python
import sys
from pathlib import Path

# Add local module to path
local_module_path = Path(__file__).parent / "local-module"
if local_module_path.exists():
    sys.path.insert(0, str(local_module_path))

from local_module import something
```

## Summary

- **Problem**: Python cannot import modules with hyphens in directory names
- **Solution**: Add directory to `sys.path` and import using actual module name
- **Files updated**: `qwen_tts_server.py`, `test_setup.py`
- **Result**: Server can now import Qwen3-TTS successfully

The fix ensures that the server can import the Qwen3-TTS module regardless of the directory naming convention.
