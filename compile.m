% copy data to device memory and initialize it
system('nvcc -c -gencode arch=compute_61,code=sm_61 -use_fast_math -maxrregcount=50 -O3 -DMLAB_PLUGIN init.cu -I/usr/local/cuda/samples/common/inc/ -Xcompiler -fPIC,-g,-fopenmp');

mex -g -DMLAB_PLUGIN init.cpp init.o  -L/usr/local/cuda/lib64 -lcudart -lgomp

system('rm init.o');

% calculate the sigma
system('nvcc -c -gencode arch=compute_61,code=sm_61 -use_fast_math -maxrregcount=50 -O3 -DMLAB_PLUGIN update_sig.cu -I/usr/local/cuda/samples/common/inc/ -Xcompiler -fPIC,-g,-fopenmp');

mex -g -DMLAB_PLUGIN update_sig.cpp  update_sig.o -L/usr/local/cuda/lib64 -lcudart -lgomp

system('rm update_sig.o');


% compute the coefficient for the energy function
system('nvcc -c -gencode arch=compute_61,code=sm_61 -use_fast_math -maxrregcount=50 -O3 -DMLAB_PLUGIN compi.cu -I/usr/local/cuda/samples/common/inc/ -Xcompiler -fPIC,-g,-fopenmp');

mex -g -DMLAB_PLUGIN compi.cpp  compi.o -L/usr/local/cuda/lib64 -lcudart -lgomp

system('rm compi.o');



% free the device memory
system('nvcc -c -gencode arch=compute_61,code=sm_61 -use_fast_math -maxrregcount=50 -O3 -DMLAB_PLUGIN fr_m.cu -I/usr/local/cuda/samples/common/inc/ -Xcompiler -fPIC,-g,-fopenmp');

mex -g -DMLAB_PLUGIN fr_m.cpp fr_m.o -L/usr/local/cuda/lib64 -lcudart -lgomp

system('rm fr_m.o');
