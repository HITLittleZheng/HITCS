#include <stdio.h>
#include <stdlib.h>
#include<time.h>
#define MAXSIZE 1025
typedef struct celltype
{
    int data;
    struct celltype *lchild,*rchild;
} BSTNode;
typedef  BSTNode* BST;
void InsertBST(int n, BST* F)
{
    if((*F)==NULL)
    {
        (*F) = (BST)malloc(sizeof(BSTNode));
        (*F)->data = n;
        (*F)->lchild = NULL;
        (*F)->rchild = NULL;
    }
    else if(n<(*F)->data)
        InsertBST( n,  &(*F)->lchild);
    else if(n>(*F)->data)
        InsertBST( n,  &(*F)->rchild);
}
BST CreatBST()
{
    BST F =NULL;
    FILE *f=NULL;
    if((f=fopen("BST.txt","r"))==NULL)
    {
        printf("Fail to open BST.txt!\n");
        exit(0);
    }
    int n;
    fscanf(f,"%d",&n);
    while(n>0)
    {
        InsertBST( n,   &F);
        fscanf(f,"%d ",&n);
    }
    fclose(f);
    return F;
}
int deletemin(BST* F)
{
    int tmp;
    BST p;
    if((*F)->lchild == NULL)
    {
        p = *F;
        tmp = (*F)->data;
        *F=(*F)->lchild;
        free(p);
        return tmp;
    }
    else
        return deletemin(&(*F)->lchild);
}
void DeleteBST(int k,BST* F)
{
    if((*F)!=NULL)
    {
        if(k<(*F)->data)
            DeleteBST( k,&(*F)->lchild);
        else if(k>(*F)->data)
            DeleteBST(k,&(*F)->rchild);
        else
        {
            if((*F)->rchild ==NULL)
                *F =(*F)->lchild;
            else if((*F)->lchild ==NULL)
                *F=(*F)->rchild;
            else
                (*F)->data=deletemin(&(*F)->rchild);
        }
    }
}
int count ;
int SearchBST(int k,BST F)
{
    BST p= F;
    if(p==NULL)
        return count*(-1);
    else if(k==p->data)
        return count+1;
    else if(k<p->data)
    {
        count=count+1;
        return (SearchBST( k,p->lchild));
    }
    else
    {
        count=count+1;
        return (SearchBST( k,p->rchild));
    }
}
int BinSearch(int k,int a[],int low,int up)
{
    int mid;
    if(low>up)
        return count*(-1);
    else
    {
        mid = (low+up)/2;
        if(k<a[mid])
        {
            count=count+1;
            return BinSearch(k,a, low,mid-1);
        }
        else if(k>a[mid])
        {
            count=count+1;
            return BinSearch(k,a, mid+1,up);
        }
        else
            return count;
    }
}
void InOrder(BST F,int a[])
{
    if(F==NULL)
        return;
    InOrder(F->lchild,a);
    a[count++]=F->data;
    InOrder(F->rchild,a);
}
void Sort(BST F)
{
    if(F==NULL)
        return;
    Sort(F->lchild);
    printf("%d ",F->data);
    Sort(F->rchild);
}
void countnum(BST F)
{
    int i;
    int success =0,failure=0;
    int num;
    for(i=1; i<=2048; i++)
    {
        count=0;
        num=SearchBST(i,F);
        if(num<0)
        {
            failure+=num*(-1);
        }
        else
        {
            success+=num;
        }
    }
    printf("平均成功查找次数%d\n",success/1024);
    printf("平均失败查找次数%d\n",failure/1024);
}
void countnum2(int a[])
{
    int i;
    int success =0,failure=0;
    int num;
    for(i=1; i<=2048; i++)
    {
        count=0;
        num=BinSearch(i,a,0,1023);
        if(num<0)
        {
            failure+=num*(-1);
        }
        else
        {
            success+=num;
        }
    }
    printf("平均成功查找次数%d\n",success/1024);
    printf("平均失败查找次数%d\n",failure/1024);
}
int main()
{
    BST F=NULL;
    F =CreatBST(&F);
    printf("The creation result is as follows.\n");
    Sort(F);
    printf("\nPlease enter the element you want to delete.\n");
    int n1;
    scanf("%d",&n1);
    DeleteBST(n1,&F);
    printf("The result of the delete operation.\n");
    Sort(F);
    int p;
    printf("\nEnter the number you want to find.\n");
    int m;
    scanf("%d",&m);
    p=SearchBST(m,F);
    printf("%d\n",p);

    int a[MAXSIZE];
    int i,j;
    j=0;
    BST F1=NULL;
    for(i=1; i<2048; i=i+2)
    {
        InsertBST(i,   &F1);
        a[j]=i;
        j++;
    }
    printf("有序插入结果\n");
    countnum(F1);

    int T=1000,n=1024,tmp;
    srand((unsigned)time(0));
    while(T--)
    {
        i=rand()%n;
        j=rand()%n;
        tmp=a[i];
        a[i]=a[j];
        a[j]=tmp;
    }
    BST F2=NULL;
    for(i=0; i<1024; i++)
    {
        InsertBST(a[i],   &F2);
    }
    printf("乱序插入结果\n");
    countnum(F2);

    count=0;
    InOrder( F1,a);
    printf("第一个折半查找结果\n");
    countnum2(a);

    count=0;
    InOrder( F2,a);
    printf("第二个折半查找结果\n");
    countnum2(a);
    return 0;
}
