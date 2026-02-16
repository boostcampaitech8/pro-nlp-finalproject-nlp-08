# -*- mode: python ; coding: utf-8 -*-

# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

added_files = [('./config/*', './config')]

a = Analysis(
    ['app/gui.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=(
        collect_submodules("transformers")
        + collect_submodules("torch")
        + collect_submodules("tokenizers")
        + collect_submodules("safetensors")
        + collect_submodules("huggingface_hub")
    ),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data,
            cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='gui',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    uac_admin=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='gui',
)
