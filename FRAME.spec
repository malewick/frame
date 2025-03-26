# -*- mode: python ; coding: utf-8 -*-

import os

block_cipher = None

# Read requirements.txt to extract hidden imports
def get_requirements_hidden_imports(path='requirements.txt'):
    hidden = []
    with open(path, 'r') as f:
        for line in f:
            pkg = line.strip()
            if not pkg or pkg.startswith('#'):
                continue
            if '==' in pkg:
                pkg = pkg.split('==')[0]
            elif '>=' in pkg:
                pkg = pkg.split('>=')[0]
            elif '<=' in pkg:
                pkg = pkg.split('<=')[0]
            elif '>' in pkg:
                pkg = pkg.split('>')[0]
            elif '<' in pkg:
                pkg = pkg.split('<')[0]
            hidden.append(pkg)
    return hidden

# Collect hidden imports from requirements.txt
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
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FRAME',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True  # Change to False if GUI-only
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FRAME'
)
