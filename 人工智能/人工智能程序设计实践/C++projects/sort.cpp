#include<iostream>
#include <cstdlib>
#include <ctime>
using namespace std;
int main()
{
    srand((int)time(NULL));  // 产生随机种子  把0换成NULL也行
    for (int i = 0; i < 10; i++)
    {
        cout << rand()%100<< " ";
    }
    return 0;
}