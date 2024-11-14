#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define MAXSIZE 100 /* 存储空间初始分配量 */
#define MAX_LEN 50 /* 栈初始分配量 */
//树结构
typedef struct BTNode  /* 结点结构 */
{
    char data;		/* 结点数据 */
    struct BTNode *lchild,*rchild; /* 左右孩子指针 */
} BTREE;

int treeIndex=0;
char str[MAXSIZE];
/* 循环队列的顺序存储结构 */
typedef struct
{
    BTREE *data[MAXSIZE];
    int front;    	/* 头指针 */
    int rear;		/* 尾指针，若队列不空，指向队列尾元素的下一个位置 */
} SqQueue;
/*为队列分配空间*/
SqQueue *Createqueue()
{
    SqQueue *p;
    p = (SqQueue *)malloc(sizeof(SqQueue));
    p->front = 0;
    p->rear = 0;
    return p;
}

//栈结构
typedef struct
{
    BTREE *data[MAX_LEN];
    int top;
} Stack;
/*为栈分配空间*/
Stack *Createstack()
{
    Stack *p;
    p = (Stack *)malloc(sizeof(Stack));
    p->top = -1;
    return p;
}

//压栈
int Push(Stack *p,BTREE *x)
{
    if (p->top == MAX_LEN - 1)
    {
        return -1;
    }
    p->top++;
    p->data[p->top] = x;
    return 0;
}

/*出栈*/
BTREE *Pop(Stack *s)
{
    BTREE *p;
    if (s->top == -1)
    {
        printf("栈空！\n");
        return NULL;
    }
    p = s->data[s->top];
    s->top--;
    return p;
}

/*栈顶*/
BTREE *Top(Stack *s)
{
    BTREE *p;
    if (s->top == -1)
    {
        printf("栈空！\n");
        return NULL;
    }
    p = s->data[s->top];
    return p;
}

//判断栈是否为空
int Empty(Stack *L)
{
    return (L->top == -1);
}
//判断栈是否为空
void MakeNull(Stack *L)
{
    L->top = -1;
}

//构造空二叉树T
int InitBT(BTREE *T)
{
    T=NULL;
    return 1;
}

//按前序输入二叉树中结点的值（一个字符）
//#表示空树，构造二叉链表表示二叉树T。
void CreateBT(BTREE **T)
{
    char ch;
    scanf(" %c",&ch);
    if(ch=='#')
        *T = NULL;
    else
    {
        *T=(BTREE*)malloc(sizeof(BTREE));
        if(!*T)
            exit(OVERFLOW);
        (*T)->data=ch; /* 生成根结点 */
        CreateBT(&((*T)->lchild)); /* 构造左子树 */
        CreateBT(&((*T)->rchild)); /* 构造右子树 */
    }
}

/* 初始条件: 二叉树T存在 */
/* 操作结果: 若T为空二叉树,则返回TRUE,否则FALSE */
int BTEmpty(BTREE *T)
{
    if(T)
        return 0;
    else
        return 1;
}



/* 初始条件: 二叉树T存在。操作结果: 返回T的根 */
char Root(BTREE *T)
{
    char Nil =' ';
    if(BTEmpty(T))
        return Nil;
    else
        return T->data;
}

/* 初始条件: 二叉树T存在，p指向T中某个结点 */
/* 操作结果: 返回p所指结点的值 */
char Value(BTREE *p)
{
    return p->data;
}

/* 给p所指结点赋值为value */
void Assign(BTREE *p,char value)
{
    p->data=value;
}

/* 初始条件: 二叉树T存在 */
/* 操作结果: 前序递归遍历T */
void PreOrder(BTREE *T)
{
    if(T==NULL)
        return;
    printf("%c",T->data);/* 显示结点数据，可以更改为其它对结点操作 */
    PreOrder(T->lchild); /* 再先序遍历左子树 */
    PreOrder(T->rchild); /* 最后先序遍历右子树 */
}
void Pre_Order(BTREE *T)
{
    Stack *S;
    S=Createstack(); //递归工作栈
    BTREE* p = T;
    while ( p !=NULL )
    {
        printf("%c",p->data);
        if ( p->rchild != NULL )
            Push (S, p->rchild );
        if ( p->lchild != NULL )
            p = p->lchild; //进左子树
        else
        {
            if(S->top!=-1)
            {
                p=Top(S);
                Pop(S);
            }
            else
            {
                p=NULL;
            }
        }
    }
}
void InOrder(BTREE *T)
{
    if(T==NULL)
        return;
    InOrder(T->lchild); /* 中序遍历左子树 */
    printf("%c",T->data);/* 显示结点数据，可以更改为其它对结点操作 */
    InOrder(T->rchild); /* 最后中序遍历右子树 */
}
void In_Order(BTREE *root)
{
    Stack *s;
    s=Createstack();
    int top= -1; //采用顺序栈，并假定不会发生上溢
    while (root!=NULL ||top!= -1)
    {
        while (root!= NULL)
        {
            s->data[++top]=root;
            root=root->lchild;
        }
        if (top!= -1)
        {
            root=s->data[top--];
            printf("%c",root->data);
            root=root->rchild;
        }
    }
}
/* 初始条件: 二叉树T存在 */
/* 操作结果: 后序递归遍历T */
void PostOrder(BTREE *T)
{
    if(T==NULL)
        return;
    PostOrder(T->lchild); /* 先后序遍历左子树  */
    PostOrder(T->rchild); /* 再后序遍历右子树  */
    printf("%c",T->data);/* 显示结点数据，可以更改为其它对结点操作 */
}
void Post_Order (BTREE *t)
{
    BTREE *p, *pr;
    Stack *s;
    s=Createstack();
    p=t;
    while(p!=NULL||!Empty(s))
    {
        while (p!=NULL)
        {
            Push(s,p);
            pr=p->rchild;
            p=p->lchild;
            if(p==NULL)
                p=pr;
        }
        p=Pop(s);
        printf("%c",p->data);
        if(!Empty(s)&&Top(s)->lchild==p)
            p=Top(s)->rchild;
        else
            p=NULL;
    }
}

void LeverOrder (BTREE *root)
{
    BTREE *q;
    SqQueue *Q;
    Q->front=Q->rear=0; //采用顺序队列，并假定不会发生上溢
    if (root==NULL)
        return;
    Q->data[++Q->rear]=root;
    while (Q->front!=Q->rear)
    {
        q=Q->data[++Q->front];
        printf("%c",q->data);
        if (q->lchild!=NULL)
            Q->data[++Q->rear]=q->lchild;
        if (q->rchild!=NULL)
            Q->data[++Q->rear]=q->rchild;
    }
}
/*判断二叉树是否为完全二叉树，采用层序遍历的方式*/
int CompleteTree(BTREE *T)
{
    SqQueue *p;
    p = Createqueue(); //创建队列
    if(T==NULL)
        return 0;
    p->rear++; //二叉树非空，根指针入队
    p->data[p->rear]=T;
    while(p->front !=p->rear)   //循环直到队列不空
    {
        if(T->lchild!=NULL&&T->rchild!=NULL) //若结点左右孩子都不为空，结点出队，左右孩子进队
        {
            p->front++;
            T=p->data[p->front];
            if(T->lchild!=NULL)
            {
                p->rear++;
                p->data[p->rear]=T->lchild;
            }
            if(T->rchild!=NULL)
            {
                p->rear++;
                p->data[p->rear]=T->rchild;
            }
        }
        if(T->lchild==NULL&&T->rchild!=NULL) //如果只有右子树没有左子树，不是
            return 0;
        if((T->lchild!=NULL&&T->rchild==NULL)||(T->lchild==NULL&&T->rchild==NULL))
            //左孩子不空右孩子空，或者左右都空,则该结点之后的所有结点必定是叶子结点
        {
            p->front++;
            while(p->front !=p->rear)
            {
                T=p->data[p->front];
                if(T->lchild==NULL&&T->rchild==NULL)
                    p->front++;
                else
                    return 0;
            }
            return 1;
        }
    }
    return 1;
}
int BTwidth (BTREE *T)
{
    SqQueue *q;
    q = Createqueue(); //创建队列
    int count=0,max=0,right;
    q->front=q->rear=-1;
    if (T!=NULL)
        q->data[++q->rear]=T;
    max=1;
    right=q->rear;
    while(q->front!=q->rear)
    {
        T=q->data[++q->front];
        if(T->lchild!=NULL)
        {
            q->data[++q->rear]=T->lchild;
            count++;
        }
        if(T->rchild!=NULL)
        {
            q->data[++q->rear]=T->rchild;
            count++;
        }
        if(q->front==right)
        {
            if(max<count)

                max=count;
            count=0;
            right=q->rear;

        }
    }
    return max;
}
int main()
{
    int i=0;
    BTREE *T;
    char e;
    InitBT(T);
    printf("输入前缀表达式\n");
    CreateBT(&T);
    printf("\n递归前序遍历二叉树:");
    PreOrder(T);
    printf("\n非递归前序遍历二叉树:");
    Pre_Order(T);
    printf("\n递归中序遍历二叉树:");
    InOrder(T);
    printf("\n非递归中序遍历二叉树:");
    In_Order(T);
    printf("\n递归后序遍历二叉树:");
    PostOrder(T);
    printf("\n非递归后序遍历二叉树:");
    Post_Order(T);
    int flag = CompleteTree(T);
    if(flag)
        printf("\n该二叉树是完全二叉树！\n");
    else
        printf("\n该二叉树不是完全二叉树！\n");
    int num;
    num =BTwidth (T);
    printf("宽度为%d",num);
    return 0;
}
