"""Opt-in private testing credentials; never discover competition credentials."""
import json
import os
from pathlib import Path
import stat


def load_testing_credentials(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Credential symlinks are not allowed')
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('Credentials must be an owned regular file with mode 0600')
    data = json.loads(path.read_text())
    if data.get('account_scope') != 'general_portfolio_testing':
        raise ValueError('Only explicit general-portfolio testing credentials accepted')
    if not all(isinstance(data.get(k), str) and data[k].isalnum() for k in ('api_key', 'api_secret')):
        raise ValueError('Invalid testing credentials')
    return data['api_key'], data['api_secret']
