Sample run on my laptop with an AMD Ryzen 7 7840HS CPU, GeForce RTX 4060 Max-Q / Mobile (rev a1), NVMe disks.

```
$ CXX=g++-13 CXXFLAGS="-O3 -fwhole-program -march=native" CUDART_PKG=cudart-12.3 make
g++-13 -Ofast -fwhole-program -march=native -I/usr/local/cuda-12.3/targets/x86_64-linux/include -I/home/stadler_h/JungfrauMX/clone/fast-feedback-indexer/install/include -I/usr/include/eigen3 simple-data-bulk-indexer.cpp -o simple-data-bulk-indexer  -L/usr/local/cuda-12.3/targets/x86_64-linux/lib -lcudart -L/home/stadler_h/JungfrauMX/clone/fast-feedback-indexer/install/lib -lfast_indexer
$ for i in $(seq 1 500); do ./simple-data-bulk-indexer --method=ifss --maxspots=200 --cells=32 --cands=32 --hsp=$((32*1024)) --triml=.01 --trimh=.3 --delta=.1 --minpts=6 --contr=.8 --iter=8 -ths=7 --rblks=4 --ipg=4 ../../data/simple/files/image*_local.txt; done > /tmp/out.txt
$ awk 'BEGIN{sum=0.0; count=0}/clock time/{sum+=$3;count++}END{print "runs="count" avg_time="(sum/count)}' /tmp/out.txt
runs=500 avg_time=0.000658362
$ ./simple-data-bulk-indexer --method=ifss --maxspots=200 --cells=32 --cands=32 --hsp=$((32*1024)) --triml=.01 --trimh=.3 --delta=.1 --minpts=6 --contr=.8 --iter=8 -ths=7 --rblks=4 --ipg=4 --quiet --rep=1000 ../../data/simple/files/image*_local.txt
per file average timings:
    clock time: 0.000238432s
  reading time: 5.7922e-07s
    index time: 0.000947375s
   refine time: 1.72151e-06s
```
