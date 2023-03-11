Sample run on my laptop with an Intel(R) Core(TM) i7-10875H CPU @ 2.30GHz, GeForce RTX 2070 SUPER Mobile / Max-Q (rev a1), NVMe disks.

```
$ CXX=g++-12 CXX_FLAGS="-O3 -fwhole-program -march=native" CUDART_PKG=cudart-12.1 make
g++-12 -O3 -fwhole-program -march=native -I/usr/local/cuda-12.1/targets/x86_64-linux/include -I/home/stadler_h/JungfrauMX/clone/fast-feedback-indexer/install/include -I/usr/include/eigen3 simple-data-bulk-indexer.cpp -o simple-data-bulk-indexer  -L/usr/local/cuda-12.1/targets/x86_64-linux/lib -lcudart -L/home/stadler_h/JungfrauMX/clone/fast-feedback-indexer/install/lib -lfast_indexer
$ for i in $(seq 1 500); do ./simple-data-bulk-indexer --method=ifss --maxspots=200 --cells=32 --cands=32 --samples=$((32*1024)) --triml=.01 --trimh=.3 --delta=.1 --minpts=6 --contr=.8 --iter=8 -ths=8 --rblks=4 --ipg=2 ../../data/simple/files/image*_local.txt; done > /tmp/out.txt
$ awk 'BEGIN{sum=0.0; count=0}/clock time/{sum+=$3;count++}END{print "runs="count" avg_time="(sum/count)}' /tmp/out.txt 
runs=500 avg_time=0.000933981
```
