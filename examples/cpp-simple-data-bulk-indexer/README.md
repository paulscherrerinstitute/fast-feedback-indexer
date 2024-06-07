## Performance test program

Commandline options are described through the *--help* option.

### Input

This program takes simple data input files and indexes them using a pipeline approach. Every file gets a work item that is progressed through stages of the pipeline: ***read data*** --> **start indexing** --> **end indexing** --> **refinement**. The first stage (***read data***) is only done once, even if files are indexed several times using the *--rep* option. If the output contains several cells (*--cells*) option, the last stage (**refinement**) can be split into separate independent substages using the *--rblks* option. The number of parallel indexing operations per GPU (*--ipg* option) and the number of CPU worker threads (*--ths* option) can be adjusted separately.

### Output

The program spits out the cells found per file and timing information. The *--quiet* option disables the cell output. See below for timing information.

### Example

Sample run on my laptop with an AMD Ryzen 7 7840HS CPU, GeForce RTX 4060 Max-Q / Mobile (rev a1), NVMe disks.

```
$ CXX=g++-13 CXXFLAGS="-Ofast -fwhole-program -march=native" CUDART_PKG=cudart-12.5 make
g++-13 -Ofast -fwhole-program -march=native -I/usr/local/cuda-12.5/targets/x86_64-linux/include -I/home/stadler_h/JungfrauMX/clone/fast-feedback-indexer/install/include -I/usr/include/eigen3 simple-data-bulk-indexer.cpp -o simple-data-bulk-indexer  -L/usr/local/cuda-12.5/targets/x86_64-linux/lib -lcudart -L/home/stadler_h/JungfrauMX/clone/fast-feedback-indexer/install/lib -lfast_indexer
$ ./simple-data-bulk-indexer --rep=30 --quiet --method=ifssr --maxspots=150 --cells=16 --cands=25 --hsp=$((25*1024)) --triml=.01 --trimh=.3 --delta=.1 --minpts=8 --contr=.8 --iter=8 -ths=7 --rblks=3 --ipg=5 --reducalc ../../data/simple/files/image*_local.txt
(version) 482d553e32cccf85de76b819bec864b92e55951a: Fri, 7 Jun 2024 10:58:07 +0200
per file average timings:
    clock time: 0.000489481s
  reading time: 1.73294e-05s
    index time: 0.00237557s
   refine time: 0.000294174s
```

Timing information is per indexing operation. The timing information consists of the *clock time* giving the average wall clock time between result delivery of the pipeline. Other timings are aggregated numbers. The *reading time* is the average time spent in the ***read data*** stage (multiply this number by the *--rep* count to get a per file average), *index time* is the average time spent in **start indexing** and **end indexing** stages, *refine time* is the average time spent in the **refinement** stage.

The *clock time* corresponds to pipeline throughput, or pipeline result delivery clock cycle, the sum of the other timing numbers to per operation time in the pipeline, or pipeline latency.
