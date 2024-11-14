#include <stdio.h>
#include <stdlib.h>
#define MAXSIZE 20

typedef struct list
{
    int data[MAXSIZE]; /* 数组，存储数据元素 */
    int length;             /* 线性表当前长度 */
} SeqList;

int GetElem(SeqList L, int i, int* e)/* 操作结果：用e返回L中第i个数据元素的值,注意i是指位置，第1个位置的数组是从0开始 */
{
    if (L.length == 0 || i<1 || i>L.length)
        return 0;
    *e = L.data[i - 1];
}

int LocateElem(SeqList L, int e)/* 操作结果：返回L中第1个与e满足关系的数据元素的位序。 */
{
    int i;
    if (L.length == 0)
        return 0;
    for (i = 0; i < L.length; i++)
    {
        if (L.data[i] == e)
            break;
    }
    if (i >= L.length)
        return 0;

    return i + 1;
}

int ListInsert(SeqList* L, int i, int e)/* 操作结果：在L中第i个位置之前插入新的数据元素e，L的长度加1 */
{
    int k;
    if (L->length == MAXSIZE)  /* 顺序线性表已经满 */
        return 0;
    if (i<1 || i>L->length + 1)/* 当i比第一位置小或者比最后一位置后一位置还要大时 */
        return 0;

    if (i <= L->length)        /* 若插入数据位置不在表尾 */
    {
        for (k = L->length - 1; k >= i - 1; k--) /* 将要插入位置之后的数据元素向后移动一位 */
            L->data[k + 1] = L->data[k];
    }
    L->data[i - 1] = e;          /* 将新元素插入 */
    L->length++;

    return 1;
}

void ListDelete(SeqList* L, int e)/*删除给定元素*/
{
    int k = 0;//记录L中等于x的数据元素个数
    for (int i = 0; i < L->length; i++)
    {
        if (L->data[i] == e)
            k++;
        else
            L->data[i - k] = L->data[i];
    }
    L->length -= k;
}

void DeleteList(SeqList* L)/*删除重复元素*/
{
    int i = 0;
    int k = 0;//k是重复的次数或是要删除的数的个数
    for (i = 0; i < L->length - 1; i++)
    {
        if (L->data[i] != L->data[i + 1])
            L->data[i - k + 1] = L->data[i + 1];
        else
            k++;
    }

    L->length = L->length - k;
}

void BubbleSort(SeqList* L)/*排序*/
{
    int i, j, temp;
    for (i = 0; i < L->length - 1; i++)
    {
        for (j = 1; j < L->length - i; j++)
        {
            if (L->data[j] < L->data[j - 1])
            {
                temp = L->data[j];
                L->data[j] = L->data[j - 1];
                L->data[j - 1] = temp;
            }
        }
    }
}
void Coverts(SeqList* L)     //将顺序表中的元素逆置
{
    int i, n;
    int temp;
    n = L->length;              //n 为线性表*A 的长度
    for (i = 0; i < n / 2; i++)  //实现逆置
    {
        temp = L->data[i];
        L->data[i] = L->data[n - i - 1];
        L->data[n - i - 1] = temp;
    }
}

void Cirmove(SeqList* L, int k)//将顺序表中的元素循环右移K位
{
    int temp;
    int i, j;
    for (i = 1; i < k + 1; i++)
    {

        temp = L->data[L->length - 1];
        for (j = L->length - 1; j > 0; j--)
        {
            L->data[j] = L->data[j - 1];
        }
        L->data[0] = temp;

    }
}
void Union(SeqList* La, SeqList Lb)//合并两个已排好序的线性表
{
    int La_len, Lb_len, i;
    int e;                        /*声明与La和Lb相同的数据元素e*/
    La_len = La->length;            /*求线性表的长度 */
    Lb_len = Lb.length;
    for (i = 1; i <= Lb_len; i++)
    {
        GetElem(Lb, i, &e);              /*取Lb中第i个数据元素赋给e*/
        if (!LocateElem(*La, e))        /*La中不存在和e相同数据元素*/
            ListInsert(La, ++La_len, e); /*插入*/
    }
}
void Print(SeqList* L)       //输出顺序表
{
    int i;
    for (i = 0; i < L->length; i++)
    {
        printf("%4d", L->data[i]);
    }
    printf("\n");
}

int main()
{
    SeqList L;
    SeqList Lb;
    int e;
    int i = 0;
    L.length = 0;
    printf("input values of L\n");
    while (1)
    {
        scanf("%d", &e);
        L.data[i] = e;
        L.length++;
        i++;
        if (getchar() == '\n') break;
    }
    int j = 0;
    Lb.length = 0;
    printf("input values of Lb\n");
    while (1)
    {
        scanf("%d", &e);
        Lb.data[j] = e;
        Lb.length++;
        j++;
        if (getchar() == '\n') break;
    }
    printf("which element do you want to delete in L\n");//删除特定元素
    scanf("%d", &e);
    ListDelete(&L, e);
    Print(&L);

    printf("Remove duplicate elements in L\n");//删除重复元素
    BubbleSort(&L);
    DeleteList(&L);
    Print(&L);

    printf("The inversion of L\n");//逆置
    Coverts(&L);
    Print(&L);

    printf("右循环几位\n");
    int k;
    scanf("%d", &k);
    Cirmove(&L, k);
    Print(&L);

    printf("The union of L and Lb\n");//合并
    BubbleSort(&L);
    BubbleSort(&Lb);
    Union(&L, Lb);
    Print(&L);

    return 0;
}
