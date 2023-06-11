eval "$('C:/ProgramData/anaconda3/Scripts/conda.exe' 'shell.bash' 'hook')"
conda activate base
# rmdir env /q /s
# mkdir env
rm ./env/solveremf2.yml
conda env export -n solveremf2 > ./env/solveremf2.yml
read -p "Press enter to continue"