#include <stdio.h>

int main(int argc, char** argv)
{
	if(argc < 3)
	{
		printf("Pass the data file name and the output file names as parameters\n");
	}

	FILE *input;
	FILE *output;
	double d;

	input = fopen(argv[1],"r");
	if (!input)
	{
		printf("Unable to open data file!");
		return 1;
	}

	output = fopen(argv[2],"wb");
	if (!output)
	{
		printf("Unable to open output file!");
		return 1;
	}

	while(fscanf(input,"%lf",&d)!=EOF)
	{
		fwrite(&d,sizeof(double),1,output);
	}
	
	fclose(input);
	fclose(output);
	
	return 0;
}