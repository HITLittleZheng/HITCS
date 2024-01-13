#include <stdio.h>
#include <string.h>
#define MAXSIZE 20 /* 存储空间初始分配量 */

typedef struct linklist
{
    int data;
    struct linklist *next;
} LinkList;

/* 初始条件：链式线性表L已存在。操作结果：返回L中数据元素个数 */

/* 初始条件：链式线性表L已存在，1≤i≤ListLength(L) */
/* 操作结果：用e返回L中第i个数据元素的值 */
int GetElem(LinkList *L,int i,int *e)
{
    int j;
    LinkList *p;		/* 声明一结点p */
    p = L;		/* 让p指向链表L的第一个结点 */
    j = 1;		/*  j为计数器 */
    while (p && j<i)  /* p不为空或者计数器j还没有等于i时，循环继续 */
    {
        p = p->next;  /* 让p指向下一个结点 */
        ++j;
    }
    if ( !p || j>i )
        return 0;  /*  第i个元素不存在 */
    *e = p->data;   /*  取第i个元素的数据 */
    return 1;
}
/* 初始条件：链式线性表L已存在 */
/* 操作结果：返回L中第1个与e满足关系的数据元素的位序。 */
/* 若这样的数据元素不存在，则返回值为0 */
int LocateElem(LinkList *L,int e)
{
    int i=0;
    LinkList *p=L->next;
    while(p)
    {
        i++;
        if(p->data==e) /* 找到这样的数据元素 */
            return i;
        p=p->next;
    }

    return 0;
}
/* 初始条件：链式线性表L已存在,1≤i≤ListLength(L)， */
/* 操作结果：在L中第i个位置之前插入新的数据元素e，L的长度加1 */
void ListInsert(LinkList *L,int e)
{
    LinkList *p,*s;
    p = L;

    s = (LinkList*)malloc(sizeof(LinkList));  /*  生成新结点(C语言标准函数) */
    s->data = e;
    s->next = p->next;      /* 将p的后继结点赋值给s的后继  */
    p->next = s;          /* 将s赋值给p的后继 */
}

/* 初始条件：链式线性表L已存在，1≤i≤ListLength(L) */
/* 操作结果：删除L的第i个数据元素，并用e返回其值，L的长度减1 */
void ListDelete(LinkList *L,int e)//删除元素e
{
    LinkList *p,*q;
    p = L;

    while(p->data==e)
    {
        L=p->next;
        free(p);
        p=L;
    }
    q=p->next;
    while (q->next!=NULL )	/* 遍历寻找第i个元素 */
    {
         if(q->data==e)
        {
            p->next = q->next;
            free(q);
            q=p->next;
        }
        else
        {
           p = p->next;
           q=p->next;
        }
    }
    if((q->next==NULL)&&(q->data==e))
    {
        p->next=NULL;
    }
    else
    {
        q->next=NULL;
    }
}
//冒泡排序的主要思想两两相性比较，每比较一次会把一轮最大或最小的数放在最后
void Sort(LinkList *L){
	LinkList *cur,*tail;
	cur=L;
	tail=NULL;
	if(cur==NULL||cur->next==NULL){
		return;
	}
	while(cur!=tail){
		while(cur->next!=tail){
			if(cur->data>cur->next->data){
				int temp=cur->data;
				cur->data=cur->next->data;
				cur->next->data=temp;

			}
			cur=cur->next;
		}
		tail=cur;
		cur=L;
	}

}
void DeleteList(LinkList *L)//先排序，和前一个元素比较，相同就下一个，不同就直接删除
{

    Sort(L);
    LinkList *p,*q;
    p = L->next;
    while(p->next!=NULL)
    {
        if(p->data==p->next->data)
        {
            p -> next = p->next->next;

        }
        else
        {
            p = p->next;
        }
    }
}
LinkList *reverse( LinkList *head )
{
    LinkList *L=(LinkList*)malloc(sizeof(LinkList)),*p,*q;//L就是新的头节点
    L->next=NULL;
    p=head;//遍历的是p，相当于中间变量
    while(p)
    {
        q=(LinkList*)malloc(sizeof(LinkList));
        q->data=p->data;
        q->next=L->next;//头插法
        L->next=q;//头插法  q,新节点
        p=p->next;
    }
    return L->next;
}
LinkList *rotateRight(LinkList *head, int k)
{
    if (head == NULL || head->next == NULL || k == 0) return head;
    int len = 1;
    LinkList *tail = head;

    /* find the end of list */
    while (tail->next != NULL)
    {
        tail = tail->next;
        len++;
    }

    /* form a circle */
    tail->next = head;
    k = k % len;
    for (int i = 0; i < len - k; i++)
    {
        tail = tail->next;
    }
    head = tail->next;
    tail->next = NULL;
    return head;
}
void Union(LinkList *La, LinkList *Lb)//合并两个已排好序的线性表
{
    LinkList *p=La;
    int  i=1;
    int e;                        /*声明与La和Lb相同的数据元素e*/
    for (p=La; p!=NULL; p=p->next)
    {
        GetElem(Lb,i,&e);
        i++;             /*取Lb中第i个数据元素赋给e*/
        if (!LocateElem(La, e))        /*La中不存在和e相同数据元素*/
            ListInsert(La, e); /*插入*/
    }
}
void Print(LinkList *L)
{
    LinkList *p=L;
    while(p!=NULL)
    {
        printf("%d   ",p->data);
        p=p->next;
    }
    printf("\n");
}


int main()
{
    LinkList *L=NULL;
    LinkList *p=NULL;
    p=(LinkList*) malloc(sizeof(LinkList));
    p -> next == NULL;
    L=p;
    int e;
    printf("input values of L\n");
    while (1)
    {
        scanf("%d",&e);
        p->data = e;
        if (getchar() == '\n') break;
        p->next = (LinkList*) malloc(sizeof(LinkList));
        p->next->next = NULL;
        p = p->next;
    }
    p=NULL;

    LinkList *Lb=NULL;
    LinkList *q=NULL;
    q=(LinkList*) malloc(sizeof(LinkList));
    q -> next == NULL;
    Lb=q;
    printf("input values of L\n");
    while (1)
    {
        scanf("%d",&e);
        q->data = e;
        if (getchar() == '\n') break;
        q->next = (LinkList*) malloc(sizeof(LinkList));
        q->next->next = NULL;
        q = q->next;
    }
    q=NULL;

    printf("which element do you want to delete in L\n");//删除特定元素
    scanf("%d", &e);
    ListDelete(L, e);
    Print(L);

    printf("Remove duplicate elements in L\n");//删除重复元素
    Sort(L);
    DeleteList(L);
    Print(L);

    printf("The inversion of L\n");//逆置
    L=reverse( L);
    Print(L);

    printf("右循环几位\n");
    int k;
    scanf("%d", &k);
    L=rotateRight(L, k);
    Print(L);

    printf("The union of L and Lb\n");//合并
    Sort(L);
    Sort(Lb);
    Union(L, Lb);
    Print(L);

    return 0;
}
