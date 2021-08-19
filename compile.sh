mkdir build
cd build
cmake ..
make -j4
export LD_LIBRARY_PATH=/home/ligirk/Workplace/facial_exp_cpp/opencv/lib/:$LD_LIBRARY_PATH
./FacialExp $1