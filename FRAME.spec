# -*- mode: python ; coding: utf-8 -*-

import os

block_cipher = None

# Auto-load hidden imports from requirements.txt
def get_requirements_hidden_imports(path='requirements.txt'):
    hidden = []
    with open(path, 'r') as f:
        for line in f:
            pkg = line.strip()
            if not pkg or pkg.startswith('#'):
                continue
            for symbol in ['==', '>=', '<=', '>', '<']:
                if symbol in pkg:
                    pkg = pkg.split(symbol)[0]
            hidden.append(pkg)
    return hidden

hidden_imports = get_requirements_hidden_imports()

a = Analysis(
    ['run/FRAME.py'],
    pathex=['src'],
    binaries=[],
    datas=[],
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='FRAME',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True  # Set to False if you want no terminal popup
)
