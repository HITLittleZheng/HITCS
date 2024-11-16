#include<iostream>

using namespace std;
int sum = 0;

void display(int row_pos[])
{
    cout<<row_pos[0]<<" "<<row_pos[1]<<" "
    <<row_pos[2]<<" "<<row_pos[3]<<" "<<row_pos[4]<<" "
    <<row_pos[5]<<" "<<row_pos[6]<<" "<<row_pos[7]<<" "
    <<endl;
    sum += 1;
}


void queen_try(int i,int row_pos[],int col[],int left_diag[],int right_diag[])
{
    if(i < 8)
    {
        for(int j = 0;j < 8; j++)
        {
            if(col[j] == 0 && left_diag[i + j] == 0 && right_diag[i - j +7] == 0)
            {
                row_pos[i] = j;
                col[j] = 1;
                left_diag[i + j] = 1;
                right_diag[i - j +7] = 1;
                queen_try(i+1, row_pos, col, left_diag, right_diag);
                col[j] = 0;
                left_diag[i + j] = 0;
                right_diag[i - j +7] = 0;
            }
        }
    }
    if(i == 8)
    {
        display(row_pos);
        return;
    }
}

int main()
{
    int n = 8;

    int row_pos[n] = {0};
    int col[n] = {0};
    int left_diag[2*n] = {0};
    int right_diag[2*n] = {0};
    queen_try(0, row_pos, col, left_diag, right_diag);
    cout<<"The Total Solution Number is: "<<sum<<endl;
}