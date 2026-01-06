#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <random>

class COO{

public:

	COO( ){ }



private:

	int* rowIdx{ nullptr };
	int* colIdx{ nullptr };
	int* value{ nullptr };
};

class CSR{

public:


private:


};

class ELL{

public:


private:


};