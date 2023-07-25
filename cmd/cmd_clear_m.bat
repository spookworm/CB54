@echo off

set "folder=.\instances_output\"

for %%F in ("%folder%\*_m*.npy") do (
    echo Deleting "%%~nxF"
    del "%%F"
)
