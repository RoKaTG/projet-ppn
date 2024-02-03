<<<<<<< HEAD
## Tree & branches explanation
=======
## Trees & branches explanation
>>>>>>> 5c7177adef552a4826d8f7f729e6adf0490900b2

  You can note that we have three different branches. In the main branch, we have a neural network unfinished mathematically but compilable and executable (and so it can produce outputs that are either incorrect or correct, with some debug prints displaying the dimensions of every vector and matrix for each operation requiring specific dimensions, such as dot product (multiplication of matrix and vector)).

We have the branch named 'Updated Network'. This branch contains our 'final' network, validated mathematically by our supervisor. It is also the one that was presented during the oral presentation but is not executable because it has a segmentation fault in it, and we are currently working on it before beginning the tuning, benchmarking, and optimization of the network for the second semester.

For any other possible branches available, please ignore them. They are either old implementations not yet deleted or branches used to implement new functionalities for the second semester without affecting our main branch.

## Prerequisites

Make sure you have the following installed on your system before getting started:

- Git
- Make
- A C compiler (like GCC)

## Installation

1. Clone the repository: `git clone [REPO_URL]`
2. Navigate to the project directory: `cd path_to_project`

## Data Preparation

1. Run `make mnist` to grant execution rights to the MNIST dataset download script and to download by execution of the script the dataset. /!\ Do not rerun to avoid deleting already downloaded data & lose time/perf

## Testing

To ensure everything is set up correctly, run: `make test_matrix` & `make test_operand`.

## Training and Testing the Network

To start training and testing the network, run: `make run`.

- Training occurs on 100 images with 10 epochs.
- Testing is conducted on 5 images (adjustable in main.c by changing specific variables).

## Additional Information

Several features planned for semester 2, such as parallelization, benchmarking, and advanced debugging tools, are not yet implemented, but also some features have been implemented but are not called in the current version. These features have been added for debugging purposes or will be used during Semester 2 for benchmarking and performance improvement, including functions like "save" and "load network" to save and load the network's state, etc. (see the report for more details).
The project is continuously evolving, and these features will be added in future updates.

Project made by : 
MSILINI Yassine,
BASLIMANE Riad Mohamed (not anymore, he decided to the stop the Master's degree),
ARHAB Sofiane,
CHABANE Khaled.
<<<<<<< HEAD

=======
>>>>>>> 5c7177adef552a4826d8f7f729e6adf0490900b2
