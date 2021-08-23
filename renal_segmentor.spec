# -*- mode: python ; coding: utf-8 -*-
import gooey
gooey_root = os.path.dirname(gooey.__file__)
gooey_languages = Tree(os.path.join(gooey_root, 'languages'), prefix='gooey/languages')
gooey_images = Tree(os.path.join(gooey_root, 'images'), prefix='gooey/images')
custom_images = Tree('.\\images', prefix='images')
block_cipher = None


a = Analysis(['renal_segmentor.py'],
             pathex=['renal_segmentor.py'],
             binaries=[],
             datas=[('models\*.model', 'models')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
          cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          gooey_languages,  # Add them in to collected files
          gooey_images,  # Same here.
          custom_images,
          [],
          name='renal_segmentor',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          windowed=True,
          icon=os.path.join('images', 'program_icon.ico'))
