#include <stdio.h>
#include<time.h>
#include <math.h>

long img[1922][1082];
long out[1922][1082];

int min(int a, int b)
{
    return (a < b) ? a : b;
}

void test(long img[1922][1082])
{
    for (int k = 0;k < 1900;k += 500)
    {
        printf("%ld\t", img[k][1000]);
    }
    printf("\n");
}

void original(long img[1922][1082])
{
    for (int j = 1; j < 1081; j++)
    {
        for (int i = 1; i < 1921;i++)
        {
            out[i][j] = (img[i - 1][j] + img[i + 1][j] + img[i][j - 1] + img[i][j + 1]) / 4;
        }
    }
}

void common(long img[1922][1082])
{
    for (int j = 1; j < 1081; j++)
    {
        for (int i = 1; i < 1921;i++)
        {
            out[i][j] = (img[i - 1][j] + img[i + 1][j] + img[i][j - 1] + img[i][j + 1]) >> 2;
        }
    }
}

void cache(long img[1922][1082])
{
    for (int i = 1; i < 1921;i++)
    {
        for (int j = 1; j < 1081; j++)
        {
            out[i][j] = (img[i - 1][j] + img[i + 1][j] + img[i][j - 1] + img[i][j + 1]) / 4;
        }
    }
}

void cpu(long img[1922][1082])
{
    int i,j;
    for ( j = 1;j < 1081-3;j += 4)
    {
        for ( i = 1;i < 1921;i++)
        {
            out[i][j] = (img[i - 1][j] + img[i + 1][j] + img[i][j - 1] + img[i][j + 1]) / 4;
            out[i][j+1] = (img[i - 1][j + 1] + img[i + 1][j + 1] + img[i][j ] + img[i][j + 2]) / 4;
            out[i][j+2] = (img[i - 1][j + 2] + img[i + 1][j + 2] + img[i][j + 1] + img[i][j + 3]) / 4;
            out[i][j+3] = (img[i - 1][j + 3] + img[i + 1][j + 3] + img[i][j + 2] + img[i][j + 4]) / 4;
        }
    }
    for (; j < 1081; j++)
    {
        for (i = 1;i < 1921;i++)
        {
            out[i][j] = (img[i - 1][j] + img[i + 1][j] + img[i][j - 1] + img[i][j + 1]) / 4;
        }
    }
}
void cpu_add(long img[1922][1082])
{
    int i, j;
    for (i = 1;i < 1921 - 3;i += 4)
    {
        for (j = 1;j < 1081;j++)
        {
            out[i][j] = (img[i - 1][j] + img[i + 1][j] + img[i][j - 1] + img[i][j + 1]) / 4;
            out[i+1][j] = (img[i ][j] + img[i + 2][j] + img[i+1][j - 1] + img[i+1][j + 1]) / 4;
            out[i+2][j] = (img[i + 1][j] + img[i + 3][j] + img[i + 2][j - 1] + img[i + 2][j + 1]) / 4;
            out[i+3][j] = (img[i + 2][j] + img[i + 4][j] + img[i + 3][j - 1] + img[i + 3][j + 1]) / 4;
        }
    }
}



void Instruction(long img[1922][1082])
{
    for (int j = 1; j < 1081; j++)
    {
        for (int i = 1; i < 1921; i += 8)
        {
            out[i][j] = (img[i - 1][j] + img[i + 1][j] + img[i][j - 1] + img[i][j + 1]) / 4;
            out[i + 2][j] = (img[i +1][j ] + img[i + 3][j ] + img[i+2][j - 1] + img[i+2][j + 1]) / 4;
            out[i + 4][j] = (img[i +3][j ] + img[i + 5][j] + img[i+4][j - 1] + img[i+4][j + 1]) / 4;
            out[i + 6][j] = (img[i +5][j ] + img[i + 7][j ] + img[i+6][j - 1] + img[i+6][j + 1]) / 4;
        }
        for (int i = 2; i < 1921; i += 8)
        {
            out[i][j] = (img[i - 1][j] + img[i + 1][j] + img[i][j - 1] + img[i][j + 1]) / 4;
            out[i + 2][j] = (img[i + 1][j] + img[i + 3][j] + img[i + 2][j - 1] + img[i + 2][j + 1]) / 4;
            out[i + 4][j] = (img[i + 3][j] + img[i + 5][j] + img[i + 4][j - 1] + img[i + 4][j + 1]) / 4;
            out[i + 6][j] = (img[i + 5][j] + img[i + 7][j] + img[i + 6][j - 1] + img[i + 6][j + 1]) / 4;
        }
    }
}


void Instruction_add(long img[1922][1082])
{
    for (int i = 1; i < 1921; i++)
    {
        for (int j = 1; j < 1081; j += 8)
        {
            out[i][j] = (img[i - 1][j] + img[i + 1][j] + img[i][j - 1] + img[i][j + 1]) / 4;
            out[i][j + 2] = (img[i - 1][j + 2] + img[i + 1][j + 2] + img[i][j - 1 + 2] + img[i][j + 1 + 2]) / 4;
            out[i][j + 4] = (img[i - 1][j + 4] + img[i + 1][j + 4] + img[i][j - 1 + 4] + img[i][j + 1 + 4]) / 4;
            out[i][j + 6] = (img[i - 1][j + 6] + img[i + 1][j + 6] + img[i][j - 1 + 6] + img[i][j + 1 + 6]) / 4;
        }
        for (int j = 2; j < 1081; j += 8)
        {
            out[i][j] = (img[i - 1][j] + img[i + 1][j] + img[i][j - 1] + img[i][j + 1]) / 4;
            out[i][j + 2] = (img[i - 1][j + 2] + img[i + 1][j + 2] + img[i][j - 1 + 2] + img[i][j + 1 + 2]) / 4;
            out[i][j + 4] = (img[i - 1][j + 4] + img[i + 1][j + 4] + img[i][j - 1 + 4] + img[i][j + 1 + 4]) / 4;
            out[i][j + 6] = (img[i - 1][j + 6] + img[i + 1][j + 6] + img[i][j - 1 + 6] + img[i][j + 1 + 6]) / 4;
        }
    }
}


void block(long img[1922][1082])
{
    int x, y, i, j;
    for (x = 1; x < 1921; x += 8)
    {
        for (y = 1; y < 1081; y += 8)
        {
            int boundi = min(1921, x + 8);
            int boundj = min(1081, y + 8);
            for (i = x; i < boundi; i++)
            {
                for (j = y; j < boundj; j++)
                {
                    out[i][j] = (img[i - 1][j] + img[i + 1][j] + img[i][j - 1] + img[i][j + 1]) / 4;
                }
            }
        }
    }
}


void success(long img[1922][1082])
{
  for (int i = 1; i < 1921; i++)
    {
        for (int j = 1; j < 1081; j += 8)
        {
            out[i][j] = (img[i - 1][j] + img[i + 1][j] + img[i][j - 1] + img[i][j + 1]) >>2;
            out[i][j + 2] = (img[i - 1][j + 2] + img[i + 1][j + 2] + img[i][j + 1] + img[i][j + 3]) >> 2;
            out[i][j + 4] = (img[i - 1][j + 4] + img[i + 1][j + 4] + img[i][j + 3] + img[i][j + 5]) >> 2;
            out[i][j + 6] = (img[i - 1][j + 6] + img[i + 1][j + 6] + img[i][j + 5] + img[i][j + 7]) >> 2;
        }
        for (int j = 2; j < 1081; j += 8)
        {
            out[i][j] = (img[i - 1][j] + img[i + 1][j] + img[i][j - 1] + img[i][j + 1]) >> 2;
            out[i][j + 2] = (img[i - 1][j + 2] + img[i + 1][j + 2] + img[i][j + 1] + img[i][j + 3]) >> 2;
            out[i][j + 4] = (img[i - 1][j + 4] + img[i + 1][j + 4] + img[i][j + 3] + img[i][j + 5]) >> 2;
            out[i][j + 6] = (img[i - 1][j + 6] + img[i + 1][j + 6] + img[i][j + 5] + img[i][j + 7]) >> 2;
        }
    }
}

int main()
{
    int i = 0;
    for (int i = 0;i < 1922;i++)
    {
        for (int j = 0; j < 1082; j++)
        {
            img[i][j] = i + j;
        }
    }


    clock_t start_t = clock();
    for (i = 0;i < 50;i++)
    {
        original(img);
    }
    clock_t end_t = clock();
    test(img);
    double sum_time = ((double)(end_t - start_t)) / CLOCKS_PER_SEC;
    printf("original program cost time: %f(s)\n", sum_time);


    start_t = clock();
    for (i = 0;i < 50;i++)
    {
        common(img);
    }
    end_t = clock();
    test(img);
    sum_time = ((double)(end_t - start_t)) / CLOCKS_PER_SEC;
    printf("after shift cost time: %f(s)\n", sum_time);


    start_t = clock();
    for (i = 0;i < 50;i++)
    {
        cache(img);
    }
    end_t = clock();
    test(img);
    sum_time = ((double)(end_t - start_t)) / CLOCKS_PER_SEC;
    printf("after cache cost time: %f(s)\n", sum_time);


    start_t = clock();
    for (i = 0;i < 50;i++)
    {
        cpu(img);
    }
    end_t = clock();
    test(img);
    sum_time = ((double)(end_t - start_t)) / CLOCKS_PER_SEC;
    printf("after cpu cost time: %f(s)\n", sum_time);


    start_t = clock();
    for (i = 0;i < 50;i++)
    {
        cpu_add(img);
    }
    end_t = clock();
    test(img);
    sum_time = ((double)(end_t - start_t)) / CLOCKS_PER_SEC;
    printf("after cpu_add cost time: %f(s)\n", sum_time);


    start_t = clock();
    for (i = 0;i < 50;i++)
    {
        Instruction(img);
    }
    end_t = clock();
    test(img);
    sum_time = ((double)(end_t - start_t)) / CLOCKS_PER_SEC;
    printf("Instruction set concurrency cost time: %f(s)\n", sum_time);

    start_t = clock();
    for (i = 0;i < 50;i++)
    {
        Instruction_add(img);
    }
    end_t = clock();
    test(img);
    sum_time = ((double)(end_t - start_t)) / CLOCKS_PER_SEC;
    printf("Instruction_add cost time: %f(s)\n", sum_time);


    start_t = clock();
    for (i = 0;i < 50;i++)
    {
        block(img);
    }
    end_t = clock();
    test(img);
    sum_time = ((double)(end_t - start_t)) / CLOCKS_PER_SEC;
    printf("after block cost time: %f(s)\n", sum_time);


    start_t = clock();
    for (i = 0;i < 50;i++)
    {
        success(img);
    }
    end_t = clock();
    test(img);
    sum_time = ((double)(end_t - start_t)) / CLOCKS_PER_SEC;
    printf("after all except block cost time: %f(s)\n", sum_time);
}
