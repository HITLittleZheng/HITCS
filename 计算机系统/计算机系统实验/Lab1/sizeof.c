#include<stdio.h>
#include<stdlib.h>
int main()
{
	printf("Size of char:%d\n", sizeof(char));
	printf("Size of short:%d\n", sizeof(short));
	printf("Size of int:%d\n", sizeof(int));
	printf("Size of long:%d\n", sizeof(long));
	printf("Size of long long:%d\n", sizeof(long long));
	printf("Size of float:%d\n", sizeof(float));
	printf("Size of double:%d\n", sizeof(double));
	printf("Size of long double:%d\n", sizeof(long double));
	printf("Size of pointers:%d\n", sizeof(int *));

	return 0;
}
