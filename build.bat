@echo off
cmake  --build .\build  --config Release --target CNVR -j 18 --
copy  .\build\Release\CNVR.exe  .\CNVR.exe
echo=
pause