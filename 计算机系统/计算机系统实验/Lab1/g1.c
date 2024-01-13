#include<stdio.h>
#include<stdlib.h>
long fib(int n)
{
	if (n == 1)
		return 1;
	if (n == 2)
		return 1;
	else
		return fib(n - 1) + fib(n - 2);
}
int main()
{
	int i,n;
	printf("Input n:\n");
	scanf_s("%d", &n);
	for (i = 1;i <= n + 1;i++)
	{
		printf("%d\n", fib(i));
	}
	double g;
	g = fib(n) / fib(n + 1);
	printf("g=%.8lf", g);
	return 0;
}