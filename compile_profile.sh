gcc -pg -g complex.c sfie.c adaptation.c model.c test.c -o test
gcc -fprofile-arcs -ftest-coverage -g complex.c sfie.c adaptation.c model.c test.c -o test
gcc -O3 complex.c sfie.c adaptation.c model.c test.c -o test

# Useful snippets:
# For second compilation option, gcov test.c after ./test
# For first option, gprof test.exe gmon.out > analysis_function.txt
