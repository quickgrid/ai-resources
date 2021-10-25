/**
 * Task: Read a binary file saved in python with numpy tofile binary format and print the saved float contents.
 */


// To remove warning error preventing from code running in visual studio for sprintf, fopen
//#define _CRT_SECURE_NO_WARNINGS



//#include <stdio.h>
//#include <math.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <iomanip>


using namespace std;



int main()
{	
	const char* fileName = "chosen.dat";
	const int rc_size = 6;	// 3 rows * 2 columns



	//std::cout << std::setprecision(32);
	//std::cout << std::setprecision(7) << std::fixed;


	/**********************************************/
	cout << "1st Method:" << "\n";
	/**********************************************/


	// Uncomment above "#define _CRT_SECURE_NO_WARNINGS" code and run this OR use this in visual studio preprocessor definition

	/*
	float randn[rc_size];
	char buff[256];
	FILE* npyfile;
	
	sprintf(buff, "%s", fileName);
	npyfile = fopen(buff, "r");
	fread(&(randn[0]), sizeof(int), rc_size, npyfile);
	fclose(npyfile);
	for (int i = 0; i < rc_size; i++) {
		cout << randn[i] << "     ";
	}
	cout << "\n";
	//printf("\n%f     %f     %f     %f     %f     %f", randn[0], randn[1], randn[2], randn[3], randn[4], randn[5]);
	*/



	//=============================================================================================================



	/**********************************************/
	cout << "\n\n";
	cout << "2nd Method:" << "\n";
	cout << "\n";
	/**********************************************/



	FILE* filepoint;
	errno_t err;
	char err2[256];
	float randn2[rc_size];


	if ((err = fopen_s(&filepoint, fileName, "r")) != 0) {
		// file could not be opened. filepoint was set to null
		// error code is returned in err.
		// error message can be retrieved with strerror(err);
		
		fprintf(stderr, "cannot open file '%s': %s\n", fileName, strerror_s(err2, 256 * sizeof(char), 1));
	}
	else {
		// file was opened, filepoint can be used to read the stream.

		fread_s(randn2, sizeof(float) * rc_size, sizeof(float), rc_size, filepoint);
		fclose(filepoint);

		for (int i = 0; i < rc_size; i++) {
			cout << randn2[i] << "     ";
		}
		cout << "\n";
		//printf("\n%f     %f     %f     %f     %f     %f", randn2[0], randn2[1], randn2[2], randn2[3], randn2[4], randn2[5]);
	}



	//=============================================================================================================


	
	/**********************************************/
	cout << "\n\n";
	cout << "3rd Method:" << "\n";
	cout << "\n";
	/**********************************************/
	

	ifstream rf(fileName, ios::out | ios::binary);
	if (!rf) {
		cout << "Cannot open file!" << endl;
		return 1;
	}
	
	float f[rc_size];
	for (int i = 0; i < rc_size; i++)
		rf.read((char*)&f[i], sizeof(float));
	
	
	rf.close();
	
	if (!rf.good()) {
		cout << "Error occurred at reading time!" << endl;
		return 1;
	}



	for (int i = 0; i < rc_size; i++) {
		cout << f[i] << "     ";
	}
	cout << "\n";

	
}



