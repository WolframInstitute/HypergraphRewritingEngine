# The GPUMC image the harnesses run on: the CAV 2025 artifact with gpumc-fixes.patch applied.
# Built by verification/gpumc/checker/build.sh as gpumc_docker:hg-fixes.
#
# /opt/hg-stdcxx and /opt/hg-stdcxx-target are copies of libstdc++'s header directories. run.sh
# puts them on the include path ahead of GPUMC's own headers, whose <atomic> is an older libc++
# that collides with libstdc++. A copy, because clang drops an -I directory that resolves to a
# system include directory, which a symlink does.
FROM gpumc_docker:latest
COPY gpumc-fixes.patch /tmp/gpumc-fixes.patch
RUN cd /artifact/GPUMC && patch -p1 < /tmp/gpumc-fixes.patch && make -j4 && make install
RUN cp -r /usr/include/c++/11 /opt/hg-stdcxx && cp -r /usr/include/x86_64-linux-gnu/c++/11 /opt/hg-stdcxx-target
