# Ubunty mirror
FROM ubuntu:latest

# Packages & dependancies 
RUN apt-get update && apt-get -y upgrade \
    && apt-get install -y build-essential \
    libc6-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    wget \
    gnupg \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Intel key for icx & mkl
RUN wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | apt-key add - \
    && echo "deb [arch=amd64] https://repositories.intel.com/graphics/ubuntu focal main" > /etc/apt/sources.list.d/intel-graphics.list

# Intel oneAPI DPC++/C++ Compiler installation
RUN apt-get update && apt-get install -y intel-oneapi-dpcpp-cpp-compiler-pro

# Work space directoy
WORKDIR /app

# Copy project file in the directory
COPY . /app