import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.metadata as md
import inspect
import platform
import subprocess
import torch
from src.utils import ROOT, write_json


def check():
    packages = {}
    for name in ['torch','torchvision','triton','helion','numpy','pandas','matplotlib','pytest','PyYAML']:
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            packages[name] = None
    result = dict(python=platform.python_version(), platform=platform.platform(),
        packages=packages, cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available(),
        muon_signature=str(inspect.signature(torch.optim.Muon)),
        torch_git_version=torch.version.git_version,
        muon_source_sha256=__import__('hashlib').sha256(
            Path(inspect.getfile(torch.optim.Muon)).read_bytes()).hexdigest())
    if torch.cuda.is_available():
        result['gpu'] = torch.cuda.get_device_name(0)
        result['bf16_supported'] = torch.cuda.is_bf16_supported()
        result['nvidia_smi'] = subprocess.check_output(['nvidia-smi'], text=True)
    write_json(ROOT / 'results/environment.json', result)
    print(__import__('json').dumps(result, indent=2))
    return result

if __name__ == '__main__':
    check()
