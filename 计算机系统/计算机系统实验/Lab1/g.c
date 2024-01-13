#include<stdio.h>
#include<stdlib.h>

double  fib(int n)
{
	int i;
	double  num1, num2;
	if (n == 1)
		return 1;
	if (n == 2)
		return 1;
	else
	{
		num1 = 1;
		num2 = 1;
		double  f;
		for (i = 3;i <= n;i++)
		{
			f = num1 + num2;

			num1 = num2;
			num2 = f;
		}
		return f;
	}
}
int main()
{
	int i, n;
	printf("Input n:\n");
	scanf_s("%d", &n);
	double g;
	for (i = 1;i <= n + 1;i++)
	{
		printf("%lf\n", fib(i));
	}
	g = fib(n) / fib(n + 1);
	printf("g=%.8lf", g);
	return 0;
}
