# Step 1: Compile model.c, complex.c, with warnings (-Wall), position independent 
# code (-fPIC), and optimization level three (-O3)
gcc -c -Wall -fPIC -O3 complex.c 
gcc -c -Wall -fPIC -O3 sfie.c
gcc -c -Wall -fPIC -O3 adaptation.c
gcc -c -Wall -fPIC -O3 model.c

# Step 2: Compile resulting objects together into a shared library 
gcc -shared -o libgfc2023.so complex.o sfie.o adaptation.o model.o