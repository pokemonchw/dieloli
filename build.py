import os
import shutil
import zipfile

#安装cx_freeze
os.system('pip install cx_freeze')

build_bat_str = \
    r'rd /s/q dist'+'\n' \
    r'rd /s/q pyera_dist'+'\n' \
    r''+'\n' \
    r'md pyera_dist'+'\n' \
    r'cd pyera_engine'+'\n' \
    r'python setup.py build'+'\n' \
    r'xcopy build ..\pyera_dist /e'+'\n' \
    r''+'\n' \
    r'rd /s/q build'+'\n' \
    r'cd ..\pyera_dist'+'\n' \
    r'if exist .\exe.win32-3.6 ren .\exe.win32-3.6 pyera_engine'+'\n' \
    r'if exist .\exe.win32-3.5 ren .\exe.win32-3.5 pyera_engine'+'\n' \
    r''+'\n' \
    r'cd ..'+'\n' \
    r'md .\pyera_dist\data'+'\n' \
    r'xcopy data .\pyera_dist\data /e'+'\n' \
    r'md .\pyera_dist\script'+'\n' \
    r'xcopy script .\pyera_dist\script /e'+'\n' \
    r'copy pyeraDebug.bat .\pyera_dist\pyeraDebug.bat'+'\n' \
    r'copy pyera.bat .\pyera_dist\pyera.bat'+'\n'

pyeraRelease_bat_str=\
    r'if "%1"=="h" goto begin'+'\n' \
    r'    start mshta vbscript:createobject("wscript.shell").run("""%~nx0"" h",0)(window.close)&&exit'+'\n' \
    r':begin'+'\n' \
    r''+'\n' \
    r'@echo off'+'\n' \
    r'pyera_engine\pyera.exe'+'\n' \
    r'exit'+'\n'

pyeraDebug_bat_str=\
    r'pyera_engine\pyera.exe'+'\n' \
    r'exit'+'\n'

with open('pyeraDebug.bat', 'wt') as f:
    f.write(pyeraDebug_bat_str)

with open('pyera.bat', 'wt') as f:
    f.write(pyeraRelease_bat_str)

with open('build.bat', 'wt') as f:
    f.write(build_bat_str)

os.system('build.bat')
os.remove('build.bat')
os.remove('pyeraDebug.bat')
os.remove('pyera.bat')

#打包目录为zip文件（未压缩）
def make_zip(source_dir, output_filename):
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    for parent, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)     #相对路径
            zipf.write(pathfile, arcname)
    zipf.close()

make_zip('.\pyera_dist','.\pyera_dist.zip')