#include <stdio.h>
#include <stdlib.h>
#define MAXVEX  50
#define MAXSIZE 50
#define NumVertices  50
#define INFINITY   65535
typedef  struct
{
    char verlist[NumVertices];
    int edge[NumVertices][NumVertices];
    int n,e;
} MTGraph;//邻接矩阵
void CreatMGraph(MTGraph *G)//构造邻接矩阵
{
    int i,j,k,w;
    /*printf("please input vertex.\n");
    i=0;
    while(1)
    {
        if(getchar()!='\n'&&i<G->n)
        {
            G->verlist[i]=getchar();
             i++;
        }
        if(i==G->n)
            break;
    }*/
    for(i=0; i<G->n; i++)
    {
        for(j=0; j<G->n; j++)
        {
            G->edge[i][j]=INFINITY;
        }
    }
    for(k=0; k<G->e; k++)
    {
        printf("please input Vertices of edge and its weight.\n");
        scanf("%d,%d,%d",&i,&j,&w);
        G->edge[i][j]=w;
        G->edge[j][i]=w;
    }
}
void DisplayMGraph(MTGraph *G)
{
    int i,j;
    for(i=0; i<G->n; i++)
    {
        printf("顶点%d ",i);
        for(j=0; j<G->n; j++)
        {
            printf(" %d",G->edge[i][j]);
        }
        printf("\n");
    }
}
typedef struct node//边表节点
{
    int adjvex;
    int cost;
    struct node*next;
} EdgeNode;
typedef struct//顶点表节点
{
    char vertex;
    EdgeNode*firstedge;
} VertexNode;
typedef struct
{
    VertexNode vexlist[NumVertices];
    int n,e;
} AdjGraph;
void CreatAGraph(AdjGraph *G)//构造邻接表
{
    int i;
    int tail,head,weight;
    /*printf("please input vertex.\n");
    i=0;
    while(1)
    {
        if(getchar()!='\n'&&i<G->n)
        {
            G->vexlist[i].vertex=getchar();
            G->vexlist[i].firstedge=NULL;
             i++;
        }
        if(i==G->n)
            break;
    }*/
    for(i=0; i<G->e; i++)
    {
        EdgeNode *p=NULL;
        printf("please input Vertices of edge and its weight.\n");
        scanf("%d,%d,%d",&tail,&head,&weight);
        p=(EdgeNode *)malloc(sizeof(EdgeNode));
        p->adjvex=head;
        p->cost = weight;
        p->next=G->vexlist[tail].firstedge;
        G->vexlist[tail].firstedge=p;
        p=(EdgeNode *)malloc(sizeof(EdgeNode));
        p->adjvex=tail;
        p->cost = weight;
        p->next=G->vexlist[head].firstedge;
        G->vexlist[head].firstedge=p;
    }
}
void DisplayAGraph(AdjGraph *G)
{
    int k;
    for(int k=0; k<G->n; k++)
    {
        printf("节点%d ",k);
        EdgeNode *p=G->vexlist[k].firstedge;
        while(p!=NULL)
        {
            printf("%d,%d  ",p->adjvex,p->cost);
            p=p->next;
        }
        printf("\n");
    }
}
void matrixtochart(MTGraph G1,AdjGraph *G2)
{
    int i,j;
    G2->n=G1.n;
    G2->e=G1.e;
    for( i=0; i<G2->n; i++)
    {
        G2->vexlist[i].vertex=G1.verlist[i];
        G2->vexlist[i].firstedge=NULL;
    }
    for(i=0; i<G2->n; i++)
    {
        for(j=0; j<G2->n; j++)
        {
            if(G1.edge[i][j]<INFINITY)
            {
                EdgeNode *p=NULL;
                p=(EdgeNode *)malloc(sizeof(EdgeNode));
                p->adjvex=i;
                p->cost = G1.edge[i][j];
                p->next=G2->vexlist[j].firstedge;
                G2->vexlist[j].firstedge=p;
            }
        }
    }
}
void charttomatrix(AdjGraph G1,MTGraph *G2)
{
    int i,j;
    G2->n=G1.n;
    G2->e=G1.e;
    for( i=0; i<G2->n; i++)
    {
        G2->verlist[i]=G1.vexlist[i].vertex;
    }
    for(i=0; i<G2->n; i++)
    {
        for( j=0; j<G2->n; j++)
        {
            G2->edge[i][j]=INFINITY;
        }
    }
    for(int k=0; k<G2->n; k++)
    {
        EdgeNode *p=G1.vexlist[k].firstedge;
        i=k;
        while(p!=NULL)
        {
            j=p->adjvex;
            G2->edge[i][j]=p->cost;
            p=p->next;
        }
    }
}
#define TRUE 1
#define FALSE 0
typedef int Boolean;
Boolean visited[MAXVEX];
typedef struct
{
    int data[MAXSIZE];
    int front;
    int rear;
} Queue;
void MakeNull(Queue*Q)
{
    Q->front=0;
    Q->rear=0;
}
int Empty(Queue*Q)
{
    if(Q->front=Q->rear)
        return 1;
    else
        return 0;
}
int EnQueue(int k,Queue*Q)
{
    if ((Q->rear+1)%MAXSIZE == Q->front)	/* 队列满的判断 */
        return 0;
    Q->data[Q->rear]=k;			/* 将元素e赋值给队尾 */
    Q->rear=(Q->rear+1)%MAXSIZE;/* rear指针向后移一位置， */								/* 若到最后则转到数组头部 */
    return  1;
}
int DeQueue(Queue *Q)
{
    int e;
    if (Q->front == Q->rear)			/* 队列空的判断 */
        return 0;
    e=Q->data[Q->front];				/* 将队头元素赋值给e */
    Q->front=(Q->front+1)%MAXSIZE;	/* front指针向后移一位置， */
    return  e;
}

#define    MAXSIZE 50
void DFS_1(MTGraph *G)//非递归深度邻接矩阵
{
    int stacks[MAXSIZE];
    int i;
    for(i = 0; i < G->n; i++)
        visited[i] = FALSE;
    int top = -1;
    i=0;
    visited[i] = TRUE;
    printf("%d ",i);
    stacks[++top] =i;
    while(top !=-1)
    {
        i = stacks[top];
        int tmp = 0;
        for(int j = 0 ; j <G->n; j++)
        {
            if( G->edge[i][j] <INFINITY && !visited[j])
            {
                visited[j] = TRUE;
                printf("%d ",j);
                stacks[++top] =j;
                break;
            }
            tmp = j;
        }
        if( tmp == G->n-1)
            top--;
    }
}
void DFS1(MTGraph *G, int i)//递归深度邻接矩阵
{
    int j;
    visited[i] = TRUE;
    printf("%d ",i);
    for(j=0; j<G->n; j++)
    {
        if((G->edge[i][j]<INFINITY)&&(!visited[j]))
            DFS1(G, j);
    }
}
void DFSTraverse1(MTGraph *G)//递归深度邻接矩阵
{
    int i;
    for(i = 0; i < G->n; i++)
        visited[i] = FALSE;
    for(i = 0; i < G->n; i++)
        if(!visited[i])
            DFS1(G, i);
}
void BFS1(MTGraph *G,int k)//非递归广度邻接矩阵
{
    int i;
    Queue Q;
    MakeNull(&Q);
    visited[k]=TRUE;
    printf("%d ",k);
    EnQueue(k,&Q);
    while(!Empty(&Q))
    {
        i=DeQueue(&Q);
        for(int j=0; j<G->n; j++)
        {
            if((G->edge[i][j]<INFINITY)&&(!visited[j]))
            {
                visited[j]=TRUE;
                printf("%d ",j);
                EnQueue(j,&Q);
            }
        }
    }
}
void BFSTraverse1(MTGraph *G)//非递归广度邻接矩阵
{
    int i;
    for(i = 0; i < G->n; i++)
        visited[i] = FALSE;
    for(i = 0; i < G->n; i++)
        if(!visited[i])
            BFS1(G, i);
}
int count =0;
void MBFS1(MTGraph *G,int k)//非递归广度邻接矩阵编码
{
    int i;

    Queue Q;
    MakeNull(&Q);
    visited[k]=TRUE;
    printf("顶点%d编码为%d ",k,count++);
    EnQueue(k,&Q);
    while(!Empty(&Q))
    {
        i=DeQueue(&Q);
        for(int j=0; j<G->n; j++)
        {
            if((G->edge[i][j]==1)&&(!visited[j]))
            {
                visited[j]=TRUE;
                printf("顶点%d编码为%d ",j,count++);
                EnQueue(j,&Q);
            }
        }
    }
}
void MBFSTraverse1(MTGraph *G)//非递归广度邻接矩阵编码
{
    int i;
    for(i = 0; i < G->n; i++)
        visited[i] = FALSE;
    for(i = 0; i < G->n; i++)
        if(!visited[i])
            MBFS1(G, i);
}

void DFS2(AdjGraph *G, int i)//递归深度邻接表
{
    EdgeNode *p;
    visited[i] = TRUE;
    printf("%d ",i);
    p = G->vexlist[i].firstedge;
    while(p)
    {
        if(!visited[p->adjvex])
            DFS2(G, p->adjvex);
        p = p->next;
    }
}
void DFSTraverse2(AdjGraph *G)//递归深度邻接表
{
    int i;
    for(i = 0; i < G->n; i++)
        visited[i] = FALSE;
    for(i = 0; i < G->n; i++)
        if(!visited[i])
            DFS2(G, i);
}
void DFS_2(AdjGraph *G)//非递归深度邻接表
{
    int stacks[MAXSIZE];
    int i;
    for(i = 0; i < G->n; i++)
        visited[i] = FALSE;
    int top = -1;
    i=0;
    visited[i] = TRUE;
    printf("%d ",i);
    stacks[++top] =i;
    while(top !=-1)
    {
        i = stacks[top];
        int tmp = 0;
        EdgeNode *p;
        p = G->vexlist[i].firstedge;
        while(p)
        {
            if(!visited[p->adjvex])
            {
                visited[p->adjvex]=TRUE;
                printf("%d ",p->adjvex);
                stacks[++top] =p->adjvex;
                break;
            }
            p = p->next;
        }
        if( p==NULL)
            top--;
    }
}

void BFS2(AdjGraph *G,int k)//非递归广度邻接表
{
    int i;
    EdgeNode *p;
    Queue Q;
    MakeNull(&Q);
    printf("%d ",k);
    visited[k]=TRUE;
    EnQueue(k,&Q);
    while(!Empty(&Q))
    {
        i=DeQueue(&Q);
        p=G->vexlist[i].firstedge;
        while(p)
        {
            if(!visited[p->adjvex])
            {
                printf("%d ",p->adjvex);
                visited[p->adjvex]=TRUE;
                EnQueue(p->adjvex,&Q);
            }
            p=p->next;
        }
    }
}
void BFSTraverse2(AdjGraph *G)//非递归广度邻接表
{
    int i;
    for(i = 0; i < G->n; i++)
        visited[i] = FALSE;
    for(i = 0; i < G->n; i++)
        if(!visited[i])
            BFS2(G, i);
}
int count2 =0;
void ABFS2(AdjGraph *G,int k)//非递归广度邻接表编码
{
    int i;

    EdgeNode *p;
    Queue Q;
    MakeNull(&Q);
    printf("顶点%d编码为%d ",k,count2++);
    visited[k]=TRUE;
    EnQueue(k,&Q);
    while(!Empty(&Q))
    {
        i=DeQueue(&Q);
        p=G->vexlist[i].firstedge;
        while(p)
        {
            if(!visited[p->adjvex])
            {
                printf("顶点%d编码为%d ",p->adjvex,count2++);
                visited[p->adjvex]=TRUE;
                EnQueue(p->adjvex,&Q);
            }
            p=p->next;
        }
    }
}
void ABFSTraverse2(AdjGraph *G)//非递归广度邻接表编码
{
    int i;
    for(i = 0; i < G->n; i++)
        visited[i] = FALSE;
    for(i = 0; i < G->n; i++)
        if(!visited[i])
            ABFS2(G, i);
}
typedef struct Tnode
{
    int data;
    struct Tnode *firstchild;
    struct Tnode *nextsibling;
} Node;
void DFSTree(MTGraph *G, int i, Node* T)
{
    visited[i] = TRUE; // 标记一下该点 已经被访问
    int first = 1;// 标记第一棵子树
    Node *q;
    for(int j= 0; j < G->n; j++)
    {
        if((G->edge[i][j]==1)&&(!visited[j]))
        {
            Node *p = (Node *)malloc(sizeof(Node));//申请空间 ， 创建节点
            p->data = j;
            p->firstchild =NULL;
            p->nextsibling = NULL;
            if(first)// 如果是第一个
            {
                T = p;
                first = 0;
            }
            else
            {
                q->nextsibling = p;
            }
            q  = p;
            DFSTree(G, j, q->firstchild);//构建 左子树
        }
    }
}
Node* DFSForest(MTGraph *G, Node *T)
{
    T = NULL;// 初始化 该森林为空
    int i;
    for( i = 0; i <G->n; i++) // 初始化标记数组
        visited[i] = FALSE;

    Node *q = NULL;// 指向 上一个兄弟
    for( i = 0; i < G->n; i++)
    {
        if(!visited[i])
        {
            Node *p = (Node *)malloc(sizeof(Node)); // 申请 空间
            //赋值 初始化节点
            p->data = i;
            p->firstchild = NULL;
            p->nextsibling = NULL;

            if(!T) // 如果该森林为空， 就让第一颗树的根， 作为森林的根
            {
                T = p;
            }
            else // 否则就是上一个兄弟 的右子树
            {
                q->nextsibling = p;
            }
            q = p; // 更新一下q

            DFSTree(G, i, p->firstchild); //  将 子树组成的森林  构建成 左子树
        }
    }
    return T;
}
void DFS_Tree(AdjGraph *G, int i, Node* T)
{
    visited[i] = TRUE; // 标记一下该点 已经被访问
    int first = 1;// 标记第一棵子树
    Node *q;
    EdgeNode *pe;
    pe = G->vexlist[i].firstedge;
    while(pe)
    {
        if(!visited[pe->adjvex])
        {
            Node *p = (Node *)malloc(sizeof(Node));//申请空间 ， 创建节点
            p->data =pe->adjvex;
            p->firstchild =NULL;
            p->nextsibling = NULL;
            if(first)// 如果是第一个
            {
                T->firstchild = p;
                first = 0;
            }
            else
            {
                q->nextsibling = p;
            }
            q  = p;
            DFS_Tree(G, pe->adjvex, q);//构建 左子树
        }
        pe = pe->next;
    }
}
Node* DFS_Forest(AdjGraph *G, Node *T)
{
    T = NULL;// 初始化 该森林为空
    int i;
    for( i = 0; i <G->n; i++) // 初始化标记数组
        visited[i] = FALSE;
    Node *q = NULL;// 指向 上一个兄弟
    for( i = 0; i < G->n; i++)
    {
        if(!visited[i])
        {
            Node *p = (Node *)malloc(sizeof(Node)); // 申请 空间
            //赋值 初始化节点
            p->data = i;
            p->firstchild = NULL;
            p->nextsibling = NULL;
            if(!T) // 如果该森林为空， 就让第一颗树的根， 作为森林的根
            {
                T = p;
            }
            else // 否则就是上一个兄弟 的右子树
            {
                q->nextsibling = p;
            }
            q = p; // 更新一下q
            DFS_Tree(G, i, p); //  将 子树组成的森林  构建成 左子树
        }
    }
    return T;
}
void PreOrder(Node *T)
{
    if(T==NULL)
        return;
    printf("%d  ",T->data);/* 显示结点数据，可以更改为其它对结点操作 */
    PreOrder(T->firstchild); /* 再先序遍历左子树 */
    PreOrder(T->nextsibling); /* 最后先序遍历右子树 */
}
void degree(MTGraph *G,int degrees[])
{
    int i,j;
    for(i=0; i<G->n; i++)
    {
        degrees[i]=0;
        for(j=0; j<G->n; j++)
        {
            if(G->edge[i][j]<INFINITY)
                degrees[i]++;
        }
        printf("顶点%d的度数为%d\n",i,degrees[i]);
    }
}
void degree2(AdjGraph *G,int degrees[])
{
    int i,j;
    for(i=0; i<G->n; i++)
    {
        degrees[i]=0;
        EdgeNode *p;
        p= G->vexlist[i].firstedge;
        while(p)
        {
            degrees[i]++;
            p=p->next;
        }
        printf("顶点%d的度数为%d\n",i,degrees[i]);
    }
}
int main()
{
   MTGraph G1;
    printf("please input the number of vertex and edge.\n");
    scanf("%d,%d",&G1.n,&G1.e);
    CreatMGraph(&G1);
    printf("邻接矩阵\n");
    DisplayMGraph(&G1);

    AdjGraph G2;
    printf("please input the number of vertex and edge.\n");
    scanf("%d,%d",&G2.n,&G2.e);
    CreatAGraph(&G2);
    printf("邻接表\n");
    DisplayAGraph(&G2);

    matrixtochart( G1,&G2);
    printf("\n邻接矩阵转为邻接表\n");
    DisplayAGraph(&G2);

    charttomatrix( G2,&G1);
    printf("\n邻接表转为邻接矩阵\n");
    DisplayMGraph(&G1);

    printf("\n非递归深度优先搜索邻接矩阵\n");
    DFS_1(&G1);
    printf("\n递归深度优先搜索邻接矩阵\n");
    DFSTraverse1(&G1);
    printf("\n广度优先搜索邻接矩阵\n");
    BFSTraverse1(&G1);
    printf("\n广度优先搜索邻接矩阵编码\n");
    MBFSTraverse1(&G1);

    printf("\n非递归深度优先搜索邻接表\n");
   DFS_2(&G2);
    printf("\n递归深度优先搜索邻接表\n");
    DFSTraverse2(&G2);
    printf("\n广度优先搜索邻接表\n");
    BFSTraverse2(&G2);
    printf("\n广度优先搜索邻接表编码\n");
    ABFSTraverse2(&G2);

    Node *T;
    T = DFSForest(&G1, T);
    printf("\n先序遍历邻接矩阵生成树\n");
    PreOrder(T);
    Node *T1;
    T1 = DFS_Forest(&G2, T1);
    printf("\n先序遍历邻接表生成树\n");
    PreOrder(T1);
   int degrees[MAXVEX];
    printf("\n邻接矩阵各顶点度数\n");
    degree(&G1,degrees);
    printf("\n邻接表各顶点度数\n");
    degree2(&G2,degrees);
    return 0;
}
