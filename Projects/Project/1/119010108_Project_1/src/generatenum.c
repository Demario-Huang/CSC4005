#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define N 100
int main()
{
	srand(time(NULL));// random number 
	FILE *src = fopen("in.txt","w+"); // create in.txt
	
	setvbuf(src, NULL, _IONBF, 0);

	for(int i = 0; i < N; i++)
	{
		fprintf(src ,"%d\n", rand());// write the number into in.txt
	}
	fclose(src);
	return 0;
}