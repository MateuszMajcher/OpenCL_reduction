#include <CL/cl.h>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <fstream>
#include <ctime>

using namespace std;
/*deklaracje*/
const char *getCLErrorString(cl_int err);
bool checkErrors(cl_int err, const char *message);
bool checkErrors(cl_int err, const string &message);
void getPlatformIDs(cl_uint *numPlatforms, cl_platform_id **platforms);
void deletePlatformIDs(cl_platform_id *platforms);
void getGPUDeviceIDs(cl_uint numPlatforms, cl_platform_id *platforms, cl_uint **numDevices, cl_device_id ***devices);
void getGPUDeviceIDsForPlatform(cl_platform_id platform, cl_uint *numDevices, cl_device_id **devices);
void deleteGPUDeviceIDs(cl_uint numPlatforms, cl_uint *numDevices, cl_device_id **devices);
void printDeviceInfo(cl_device_id device);
bool createContext(const cl_device_id *device, cl_context *context);
bool createCommandQueue(cl_context context, cl_device_id device, cl_command_queue *commandQueue);
bool createAndBuildProgram(const char *filename, cl_context context, cl_program *program);
bool createKernel(cl_program program, const char *kernelName, cl_kernel *kernel);
size_t computeGlobalWorkSize(cl_uint dataSize, size_t localWorkSize);




void delete2dArray(int **ar, int rows, int cols) {
	for (int row = 0; row<rows; row++) {
		delete[] ar[row];
	}
	delete[] ar;
}

void loadValue(int *ar, int rows, int cols) {
	float a = 0;


	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			ar[col + row* cols] = rand() % 256;
			;
		}
	}
}



void print(int *ar, int size) {
	cout << endl;
	int sum = 0;
	for (int i = 0; i<size; i++) {
		cout << i << " wystapilo " << " - " << ar[i] << " razy" << endl;
		sum += ar[i];
	}
	cout << sum << endl;
}


void print2d(int *ar, int rows, int cols) {
	cout << endl;
	for (int row = rows - 50; row<rows; row++) {
		for (int col = cols - 50; col<cols; col++) {
			cout << " | " << ar[col + row* cols];
		}
		cout << " | " << endl;
	}
}





int main(int argc, char **argv)
{

	cl_int err;
	cl_uint num_platforms = 0;
	cl_platform_id *platforms = NULL;
	cl_uint *num_devices = NULL;
	cl_device_id **devices = NULL;

	getPlatformIDs(&num_platforms, &platforms);
	getGPUDeviceIDs(num_platforms, platforms, &num_devices, &devices);

	int use_platform = 0;
	int use_device = 0;
	cl_platform_id platform = platforms[use_platform];
	cl_device_id device = devices[use_platform][use_device];
	cl_context context = 0;
	cl_command_queue command_queue = 0;

	if (createContext(&device, &context))
	{
		deleteGPUDeviceIDs(num_platforms, num_devices, devices);
		deletePlatformIDs(platforms);
		exit(EXIT_FAILURE);
	}

	if (createCommandQueue(context, device, &command_queue))
	{
		clReleaseContext(context);
		deleteGPUDeviceIDs(num_platforms, num_devices, devices);
		deletePlatformIDs(platforms);
		exit(EXIT_FAILURE);
	}

	cl_program program = 0;
	cl_kernel kernel = 0;
	cl_program program_s = 0;
	cl_kernel kernel_s = 0;

	err = createAndBuildProgram("kernel.cl", context, &program_s);
	createKernel(program_s, "vecAdd", &kernel_s);
	checkErrors(err, "blad tworzenia programu");

	err = createAndBuildProgram("reduce.cl", context, &program);
	createKernel(program, "vecMinReduce", &kernel);
	checkErrors(err, "blad tworzenia programu");

	cl_command_queue queue = clCreateCommandQueue(context, **devices, 0, &err);
	

	float a = 1;
	float b = 2;

	//cout << INT_MAX << endl;
	const  unsigned int sizes = 1000;

	const size_t byte = sizes*sizeof(int);



	/*alokacja pamiecji na urzadzeniu*/
	const int mem_size = sizeof(float) * sizes;
	const size_t global_work_sizes = sizes;
	const size_t local_work_sizes = 1;



	float *h_a = new float[sizes];



	cl_mem d_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		mem_size, NULL, &err);


	err = clSetKernelArg(kernel_s, 0, sizeof(cl_mem), &d_a);
	err |= clSetKernelArg(kernel_s, 1, sizeof(float), &a);
	err |= clSetKernelArg(kernel_s, 2, sizeof(float), &b);
	err |= clSetKernelArg(kernel_s, 3, sizeof(unsigned int), &sizes);

	checkErrors(err, "blad ustawiania argumentow");
	clEnqueueNDRangeKernel(command_queue, kernel_s, 1, NULL,
		&global_work_sizes, &local_work_sizes, 0, NULL, NULL);

	checkErrors(err, "blad wykonannia kernela");
	err = clFinish(command_queue);

	float* check = new float[sizes];

	err = clEnqueueReadBuffer(command_queue, d_a, CL_TRUE,
		0, mem_size, check, 0, NULL, NULL);
	checkErrors(err, "blad odczytu wyniku");

	/*wyswietlenie histogramu*/
	/*float sum = 0;
	for (int i = 0; i < 1000; i++) {
		cout << check[i] << endl;
		sum += check[i];
	}
	cout << "suma " << sum << endl;
	*/
	/*koniec calki*/

	//cout << "dokladnosc " << sum - 13.5 << endl;


	/*
	
	*/


	cl_uint size = sizes;
	size_t bytes = size*sizeof(float);
	size_t local_work_size = 256;
	size_t global_work_size = computeGlobalWorkSize(size, local_work_size);
	size_t num_work_groups = global_work_size / local_work_size;

	/*float *h_vec = new float[size];
	
	for (int i = 0; i < 1000; i++)
		h_vec[i] = check[i];*/

	cl_mem d_vec = clCreateBuffer(context, CL_MEM_READ_WRITE |
		CL_MEM_COPY_HOST_PTR, bytes, check, &err);
	cl_mem d_results = clCreateBuffer(context, CL_MEM_READ_WRITE,
		sizeof(float)*num_work_groups,
		NULL, &err);
	cl_mem d_size = clCreateBuffer(context, CL_MEM_READ_ONLY,
		sizeof(cl_uint), NULL, &err);
	
	checkErrors(err, "Blad ustawiania buforow");

	err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_vec);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_results);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_size);
	err |= clSetKernelArg(kernel, 3, sizeof(float)*local_work_size, NULL);
	checkErrors(err, "Blad ustawiania argumentow");

	cl_uint step = 0;
	while (num_work_groups > 0)
	{
		cout << "Step " << ++step << endl;
		cout << "Size of data to be reduced: " << size << endl;
		cout << "Local work size: " << local_work_size << endl;
		cout << "Global work size: " << global_work_size << endl;
		cout << "Num of work-groups: " << num_work_groups << endl << endl;
		err = clEnqueueWriteBuffer(command_queue, d_size, CL_TRUE, 0,
			sizeof(cl_uint), &size, 0, NULL, NULL);
		checkErrors(err, "Blad zapisu rozmiaru");
		clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
			&global_work_size, &local_work_size,
			0, NULL, NULL);
		err = clFinish(command_queue);
		checkErrors(err, "Blad wykonania kernela");
		if (num_work_groups > 1)
		{
			size = num_work_groups;
			global_work_size = computeGlobalWorkSize(size, local_work_size);
			num_work_groups = global_work_size / local_work_size;
			cl_mem tmp = d_vec;
			d_vec = d_results;
			d_results = tmp;
			err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_vec);
			err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_results);
			checkErrors(err, "Blad ustawiania argumentow");
		}

		else
			num_work_groups = 0;
	}

	float result = 0;
	err = clEnqueueReadBuffer(command_queue, d_results, CL_TRUE, 0,
		sizeof(float), &result, 0, NULL, NULL);
	checkErrors(err, "Blad odczytu wyniku");
	cout << "\n\n\n";
	cout << "Min value: " << result << endl;
	cout << "dokladnosc " << result - 13.5 << endl;
	



	clReleaseKernel(kernel);
	clReleaseKernel(kernel_s);

	clReleaseProgram(program);
	clReleaseProgram(program_s);

	delete[] h_a;
	delete[] check;

	clReleaseMemObject(d_vec);
	clReleaseMemObject(d_size);
	clReleaseMemObject(d_results);
	clReleaseMemObject(d_a);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	deleteGPUDeviceIDs(num_platforms, num_devices, devices);
	deletePlatformIDs(platforms);
	system("pause");
	return 0;
}

const char *getCLErrorString(cl_int err)
{
	switch (err)
	{
	case CL_SUCCESS: return "CL_SUCCESS";
	case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
	case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
	case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
	case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
	case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
	case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
	case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
	case CL_MISALIGNED_SUB_BUFFER_OFFSET: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
	case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
	case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
	case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
	case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
	case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
	case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
	case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
	case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
	case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
	case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
	case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
	case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
	case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
	case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
	case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
	case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
	case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
	case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
	case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
	case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
	case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
	case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
	case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
	case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
	case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
	case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
	case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
	case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
	case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
	case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
	case CL_INVALID_PROPERTY: return "CL_INVALID_PROPERTY";
	default: return "Unknown OpenCL error!";
	}
}

bool checkErrors(cl_int err, const char *message)
{
	if (CL_SUCCESS == err)
	{
		return false;
	}

	cerr << message << endl;
	cerr << "Error code: " << getCLErrorString(err) << endl;
	system("pause > nul");
	return true;
}

bool checkErrors(cl_int err, const string &message)
{
	return checkErrors(err, message.c_str());
}

void getPlatformIDs(cl_uint *numPlatforms, cl_platform_id **platforms)
{
	cl_int err;

	err = clGetPlatformIDs(0, NULL, numPlatforms);
	if (checkErrors(err, "Error while obtaining the number of OpenCL platforms!"))
	{
		exit(EXIT_FAILURE);
	}

	cout << "The number of available OpenCL platforms: " << *numPlatforms << endl;

	*platforms = new cl_platform_id[*numPlatforms];
	err = clGetPlatformIDs(*numPlatforms, *platforms, NULL);
	if (checkErrors(err, "Error while obtaining the IDs of OpenCL platforms!"))
	{
		deletePlatformIDs(*platforms);
		exit(EXIT_FAILURE);
	}

	cout << "The IDs of OpenCL platforms obtained." << endl << endl;
}

void deletePlatformIDs(cl_platform_id *platforms)
{
	delete[] platforms;
}

void getGPUDeviceIDs(cl_uint numPlatforms, cl_platform_id *platforms, cl_uint **numDevices, cl_device_id ***devices)
{
	cl_uint sum_num_devices = 0;

	*numDevices = new cl_uint[numPlatforms];
	*devices = new cl_device_id*[numPlatforms];
	for (cl_uint i = 0; i < numPlatforms; ++i)
	{
		cout << ">> OpenCL platform #" << i << endl;
		(*numDevices)[i] = 0;
		(*devices)[i] = NULL;
		getGPUDeviceIDsForPlatform(platforms[i], *numDevices + i, *devices + i);
		sum_num_devices += (*numDevices)[i];
		for (cl_uint j = 0; j < (*numDevices)[i]; ++j)
		{
			cout << "> OpenCL GPU device #" << j << ":" << endl;
			printDeviceInfo((*devices)[i][j]);
		}
		cout << endl;
	}
	if (0 == sum_num_devices)
	{
		cerr << "None of the OpenCL GPU devices were found!" << endl;
		deleteGPUDeviceIDs(numPlatforms, *numDevices, *devices);
		deletePlatformIDs(platforms);
		system("pause > nul");
		exit(EXIT_FAILURE);
	}
}

void getGPUDeviceIDsForPlatform(cl_platform_id platform, cl_uint *numDevices, cl_device_id **devices)
{
	cl_int err;

	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, numDevices);
	if (checkErrors(err, "Error while obtaining the number of available OpenCL GPU devices!"))
	{
		return;
	}

	cout << "The number of available OpenCL GPU devices: " << *numDevices << endl;
	if (0 == *numDevices)
	{
		return;
	}

	*devices = new cl_device_id[*numDevices];
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, *numDevices, *devices, NULL);
	if (checkErrors(err, "Error while obtaining the IDs of OpenCL GPU devices!"))
	{
		delete[] * devices;
		*devices = NULL;
		*numDevices = 0;
		return;
	}

	cout << "The IDs of OpenCL GPU devices obtained." << endl;
}

void deleteGPUDeviceIDs(cl_uint numPlatforms, cl_uint *numDevices, cl_device_id **devices)
{
	for (cl_uint i = 0; i < numPlatforms; ++i)
	{
		delete[] devices[i];
	}
	delete[] devices;
	delete[] numDevices;
}

void printDeviceInfo(cl_device_id device)
{
	cl_int err;
	size_t param_value_size = 0;
	char *buffer = NULL;
	cl_ulong size;

	// CL_DEVICE_NAME
	err = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &param_value_size);
	if (!checkErrors(err, "Error while obtaining the size of device name string!"))
	{
		buffer = new char[param_value_size];
		clGetDeviceInfo(device, CL_DEVICE_NAME, param_value_size, static_cast<void *>(buffer), NULL);
		cout << "CL_DEVICE_NAME: " << buffer << endl;
		delete[] buffer;
		buffer = NULL;
	}

	// CL_DEVICE_VENDOR
	err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, 0, NULL, &param_value_size);
	if (!checkErrors(err, "Error while obtaining the size of vendor name string!"))
	{
		buffer = new char[param_value_size];
		clGetDeviceInfo(device, CL_DEVICE_VENDOR, param_value_size, static_cast<void *>(buffer), NULL);
		cout << "CL_DEVICE_VENDOR: " << buffer << endl;
		delete[] buffer;
		buffer = NULL;
	}

	// CL_DEVICE_VERSION
	err = clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &param_value_size);
	if (!checkErrors(err, "Error while obtaining the size of OpenCL version string!"))
	{
		buffer = new char[param_value_size];
		clGetDeviceInfo(device, CL_DEVICE_VERSION, param_value_size, static_cast<void *>(buffer), NULL);
		cout << "CL_DEVICE_VERSION: " << buffer << endl;
		delete[] buffer;
		buffer = NULL;
	}

	// CL_DEVICE_PROFILE
	err = clGetDeviceInfo(device, CL_DEVICE_PROFILE, 0, NULL, &param_value_size);
	if (!checkErrors(err, "Error while obtaining the size of OpenCL profile string!"))
	{
		buffer = new char[param_value_size];
		clGetDeviceInfo(device, CL_DEVICE_PROFILE, param_value_size, static_cast<void *>(buffer), NULL);
		cout << "CL_DEVICE_PROFILE: " << buffer << endl;
		delete[] buffer;
		buffer = NULL;
	}

	// CL_DEVICE_ADDRESS_BITS
	err = clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(cl_ulong), &size, 0);
	if (!checkErrors(err, "Error while obtaining the size of OpenCL profile string!"))
	{
		buffer = new char[param_value_size];
		clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(cl_ulong), &size, 0);
		cout << "CL_DEVICE_ADDRESS_BITS: " << size << endl;
		delete[] buffer;
		buffer = NULL;
	}

	// CL_DRIVER_VERSION
	err = clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, NULL, &param_value_size);
	if (!checkErrors(err, "Error while obtaining the size of OpenCL software driver version string!"))
	{
		buffer = new char[param_value_size];
		clGetDeviceInfo(device, CL_DRIVER_VERSION, param_value_size, static_cast<void *>(buffer), NULL);
		cout << "CL_DRIVER_VERSION: " << buffer << endl;
		delete[] buffer;
		buffer = NULL;
	}
}

bool createContext(const cl_device_id *device, cl_context *context)
{
	cl_int err;

	*context = clCreateContext(NULL, 1, device, NULL, NULL, &err);
	if (checkErrors(err, "Error while creating an OpenCL context!"))
	{
		*context = 0;
		return true;
	}

	cout << "An OpenCL context created." << endl << endl;
	return false;
}

bool createCommandQueue(cl_context context, cl_device_id device, cl_command_queue *commandQueue)
{
	cl_int err;

	*commandQueue = clCreateCommandQueue(context, device, 0, &err);
	if (checkErrors(err, "Error while creating a command-queue!"))
	{
		*commandQueue = 0;
		return true;
	}

	cout << "A command-queue created." << endl << endl;
	return false;
}

bool createAndBuildProgram(const char *filename, cl_context context, cl_program *program)
{
	cl_int err;
	ifstream file_cl(filename);
	string source_code;
	string line;
	char *buffer = NULL;

	if (!file_cl.is_open())
	{
		cerr << "Error while opening the file '" << filename << "'!" << endl;
		return true;
	}

	while (file_cl.good())
	{
		getline(file_cl, line);
		source_code += line;
	}
	file_cl.close();
	buffer = new char[source_code.length() + 1];
	sprintf(buffer, "%s", source_code.c_str());
	*program = clCreateProgramWithSource(context, 1, const_cast<const char **>(&buffer), NULL, &err);
	delete[] buffer;
	buffer = NULL;
	if (checkErrors(err, string("Error while creating a program object for the source code '") + filename + "'!"))
	{
		*program = 0;
		return true;
	}

	err = clBuildProgram(*program, 0, NULL, NULL, NULL, NULL);
	if (checkErrors(err, string("Error while building a program executable from the source code '") + filename + "'!"))
	{
		clReleaseProgram(*program);
		*program = 0;
		return true;
	}

	cout << "An OpenCL program created and built from the source code '" << filename << "'." << endl << endl;
	return false;
}

bool createKernel(cl_program program, const char *kernelName, cl_kernel *kernel)
{
	cl_int err;

	*kernel = clCreateKernel(program, kernelName, &err);
	if (checkErrors(err, string("Error while creating a kernel object for a function '") + kernelName + "'!"))
	{
		*kernel = 0;
		return true;
	}

	cout << "A kernel object for a function '" << kernelName << "' created." << endl << endl;
	return false;
}

size_t computeGlobalWorkSize(cl_uint dataSize, size_t localWorkSize)
{
	return (dataSize%localWorkSize) ? dataSize - dataSize%localWorkSize +
		localWorkSize : dataSize;
}