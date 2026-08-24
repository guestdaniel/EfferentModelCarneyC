# Step 1: Compile model.c, complex.c, with warnings (-Wall), position independent 
# code (-fPIC), and optimization level three (-O3)
gcc -c -Wall -fPIC -O3 complex.c 
gcc -c -Wall -fPIC -O3 sfie.c
gcc -c -Wall -fPIC -O3 adaptation.c
gcc -c -Wall -fPIC -O3 model.c

# Step 2: Compile resulting objects together into a shared library
# Detect platform and set the shared library extension
case "$(uname -s)" in
    Linux*)
        gcc -shared -o libgfc2026.so complex.o sfie.o adaptation.o model.o
        ;;
    Darwin*)
        gcc -shared -o libgfc2026.dylib complex.o sfie.o adaptation.o model.o
        ;;
    MINGW*|MSYS*|CYGWIN*)
        gcc -shared -o libgfc2026.so complex.o sfie.o adaptation.o model.o
        ;;
    *)          echo "Unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac
