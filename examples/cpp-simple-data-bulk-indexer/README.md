## Performance test program

Commandline options are described through the *--help* option.

### Input

This program takes simple data input files and indexes them using a pipeline approach. Every file gets a work item that is progressed through stages of the pipeline: ***read data*** --> **start indexing** --> **end indexing** --> **refinement**. The first stage (***read data***) is only done once, even if files are indexed several times using the *--rep* option. If the output contains several cells (*--cells*) option, the last stage (**refinement**) can be split into separate independent substages using the *--rblks* option. The number of parallel indexing operations per GPU (*--ipg* option) and the number of CPU worker threads (*--ths* option) can be adjusted separately.

### Output

The program spits out the cells found per file and timing information. The *--quiet* option disables the cell output. See below for timing information.

### Example

Sample run on my laptop with an AMD Ryzen 7 7840HS CPU, GeForce RTX 4060 Max-Q / Mobile (rev a1), NVMe disks.

```
$ CXX=g++-13 CXXFLAGS="-Ofast -fwhole-program -march=native" CUDART_PKG=cudart-12.3 make
g++-13 -Ofast -fwhole-program -march=native -I/usr/local/cuda-12.3/targets/x86_64-linux/include -I/home/stadler_h/JungfrauMX/clone/fast-feedback-indexer/install/include -I/usr/include/eigen3 simple-data-bulk-indexer.cpp -o simple-data-bulk-indexer  -L/usr/local/cuda-12.3/targets/x86_64-linux/lib -lcudart -L/home/stadler_h/JungfrauMX/clone/fast-feedback-indexer/install/lib -lfast_indexer
$ for i in $(seq 1 500); do ./simple-data-bulk-indexer --method=ifss --maxspots=200 --cells=32 --cands=32 --hsp=$((32*1024)) --triml=.01 --trimh=.3 --delta=.1 --minpts=6 --contr=.8 --iter=8 -ths=7 --rblks=4 --ipg=4 ../../data/simple/files/image*_local.txt; done > /tmp/out.txt
$ awk 'BEGIN{sum=0.0; count=0}/clock time/{sum+=$3;count++}END{print "runs="count" avg_time="(sum/count)}' /tmp/out.txt
runs=500 avg_time=0.000632549
$ ./simple-data-bulk-indexer --method=ifss --maxspots=200 --cells=32 --cands=32 --hsp=$((32*1024)) --triml=.01 --trimh=.3 --delta=.1 --minpts=6 --contr=.8 --iter=8 -ths=7 --rblks=4 --ipg=4 --rep=10 --quiet ../../data/simple/files/image*_local.txt
per file average timings:
    clock time: 0.000278318s
  reading time: 5.74394e-05s
    index time: 0.00103789s
   refine time: 0.000150532s
```

Timing information is per indexing operation. The timing information consists of the *clock time* giving the average wall clock time in the pipeline. Other timings are aggregated numbers. The *reading time* is the average time spent in the ***read data*** stage (multiply this number by the *--rep* count to get a per file average), *index time* is the average time spent in **start indexing** and **end indexing** stages, *refine time* is the average time spent in the **refinement** stage.

The *clock time* corresponds to pipeline throughput, or pipeline clock cycle, the sum of the other timing numbers to per operation time in the pipeline, or pipeline latency.
