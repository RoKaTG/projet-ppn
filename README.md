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
MSILINI Yassine
BASLIMANE Riad Mohamed
ARHAB Sofiane
CHABANE Khaled
