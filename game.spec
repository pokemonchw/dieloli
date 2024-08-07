# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['game.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'scipy._lib.messagestream',
        'scipy.ndimage._ni_support',
        'scipy.ndimage.filters',
        'scipy.ndimage._filters',
        'scipy.ndimage._measurements',
        'scipy.ndimage._morphology',
        'scipy.special.cython_special',
        'sklearn.utils._cython_blas',
        'sklearn.neighbors.typedefs',
        'sklearn.neighbors.quad_tree',
        'sklearn.tree._utils',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='game',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='game',
)

