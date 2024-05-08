## Prerequisites

Make sure you have the following installed on your system before getting started:

- Git
- CMake
- Intel(R) oneAPI DPC++/C++ Compiler 2024.0.0 (2024.0.0.20231017)
- MKL (Math Kernel Library)
- OpenMP

## Installation

1. Clone the repository: `git clone [REPO_URL]`
2. Navigate to the project directory: `cd projet-ppn`
3. If you can't clone it just download a copy and extract it.

## Data Preparation

1. Run the command line : 
```bash
chmod +x script/get-mnist.sh
cd script
./get-mnist.sh
``````
 to grant execution rights to the MNIST dataset download script and to download by execution of the script the dataset. /!\ Do not rerun to avoid deleting already downloaded data & lose time/space

## CMake

1. Create a build directory and navigate into it:
```bash
mkdir build && cd build/
```

2. Configure the project with CMake:
```bash
cmake ..
```

3. Compile the project:
```bash
make
```

## Execute the network
1. Return to the source directory to run the executable:
```bash
cd ..
```

2. The mlp need 5 to 6 arguments in total, here is a general execution : 
```bash
./mlp [routine:false OR true] [topology:784,size_of_all_hidden_layer_separated_by_coma,10]
 [activation:relu OR sigmoid OR fast_sigmoid OR leaky OR swish OR tanh] [numEpoch:int] 
 [batchSize:int (only if routine:true)]
```

If you do it wrong, the executable will tell you how to do it and end the code execution as follow :
```bash
./mlp
Usage : ./mlp [routine] [Topology] [Activation] [TrainingSample] [numEpoch] [batchSize]
                ^          ^         ^              ^                         ^         
                |          |         |              |                         |         
batch:true   ___|          |         |              |__=< 60000               |___only when routine:true
classic:false              |         |                                                  
                           |         |                                                  
Separate layers by ','_____|         |___ relu || sigmoid || fast_sigmoid || leaky || tanh || swish  
```

3. Here is a sample of how to do it with or whitout batch usage : 
```bash
/mlp false 784,300,200,10 relu 60000 3
```

```bash
/mlp true 784,128,10 leaky 60000 3 16
```

## Important Notes
1. This project is configured to compile and run using Intel's icx compiler due to dependencies on specific features like dgemm_batch from MKL, which are not supported by GCC.
2. For optimal performance, it is recommended to run this on the fob1 cluster, specifically on a hsw0X node.
3. If you are not on the cluster, a Docker container setup is provided in the docker/ directory for a compatible environment (Not tested yet !).

Fun stat : 
```bash
git ls-files | while read f; do git blame -w -M -C -C --line-porcelain "$f" | grep -I '^author '; done | sort -f | uniq -ic | sort -n --reverse
```

Project made by : 

MSILINI Yassine (RKG = RoKATG),

ARHAB Sofiane (dxhardys),

CHABANE Khaled (CHA-Khaled).

