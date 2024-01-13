#include <stdio.h>
#include <stdlib.h>
#define MAXVEX  50
#define INFINITY   65535
typedef  struct
{
    int vertex[MAXVEX];
    int edge[MAXVEX][MAXVEX];
    int n,e;
} MTGraph;//邻接矩阵
void CreatMGraph(MTGraph *G)//构造邻接矩阵
{
    FILE *f=NULL;
    if((f=fopen("MTGraph.txt","r"))==NULL)
    {
        printf("Fail to open MTGraph.txt!\n");
        exit(0);
    }
    fscanf(f,"%d",&G->n);
    fscanf(f,"%d",&G->e);
    int i,j;
    for (i = 0; i < G->n; i++)/* 初始化图 */
    {
        G->vertex[i]=i;
    }
    for(i=0; i<G->n; i++)
    {
        for(j=0; j<G->n; j++)
        {
            if(i==j)
            {
                G->edge[i][j]=0;
            }
            else
            {
                G->edge[i][j]=INFINITY;
            }
        }
    }
    for(i=0; i<G->n; i++)
    {
        fscanf(f,"%d",&i);
        fscanf(f,"%d",&j);
        fscanf(f,"%d",&G->edge[i][j]);
        G->edge[j][i]=G->edge[i][j];
    }
    fclose(f);
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
void Floyd(MTGraph *G,int Path[][MAXVEX], int Dis[][MAXVEX])
{
    int i,j,k;
    for(i=0; i<G->n; ++i)
    {
        for(j=0; j<G->n; ++j)
        {
            Dis[i][j]=G->edge[i][j];
            Path[i][j]=j;
        }
    }
    for(k=0; k<G->n; ++k)
    {
        for(i=0; i<G->n; ++i)
        {
            for(j=0; j<G->n; ++j)
            {
                if (Dis[i][j]>Dis[i][k]+Dis[k][j])
                {
                    Dis[i][j]=Dis[i][k]+Dis[k][j];
                    Path[i][j]=Path[i][k];
                }
            }
        }
    }
}
int MinCost(int *D,int *Vis,MTGraph *G)
{
    int temp=INFINITY;
    int i,w;
    for(i=0; i<G->n; i++)
    {
        if(!Vis[i]&&D[i]<temp)
        {
            temp=D[i];
            w=i;
        }
    }
    return w;
}
void Dijkstra(MTGraph *G,int *D,int *P,int *Vis,int v)
{
    int i,j,w;
    for(i=0; i<G->n; i++)
    {
        Vis[i] = 0;
        D[i] = G->edge[v][i];
        P[i] = -1;
    }
    Vis[v] = 1;
    for(i=0; i<G->n; i++)
    {
        w=MinCost( D, Vis,G);
        Vis[w] = 1;
        for(j=0; j<G->n; j++)
        {
            if(!Vis[j] && (D[w]+G->edge[w][j]<D[j]))
            {
                D[j] = D[w]+G->edge[w][j];
                P[j]=w;
            }
        }
    }
}
void Dijkstra2(MTGraph *G,int *D,int *P,int *Vis,int v,int u)
{
    int i,j,w;
    for(i=0; i<G->n; i++)
    {
        Vis[i] = 0;
        D[i] = G->edge[v][i];
        P[i] = -1;
    }
    Vis[v] = 1;
    for(i=0; i<G->n; i++)
    {
        w=MinCost( D, Vis,G);
        Vis[w] = 1;
        if(w==u)
        {
            break;
        }
        for(j=0; j<G->n; j++)
        {
            if(!Vis[j] && (D[w]+G->edge[w][j]<D[j]))
            {
                D[j] = D[w]+G->edge[w][j];
                P[j]=w;
            }
        }
    }
}

int main()
{
    MTGraph G;
    CreatMGraph(&G);
    DisplayMGraph(&G);
    int Path[MAXVEX][MAXVEX];
    int Dis[MAXVEX][MAXVEX];
    Floyd(&G,Path,Dis);
    int v, u	;
    printf("请输入u和v\n");
    scanf("%d,%d",&u,&v);
    printf("v%d-v%d weight: %d ",u,v,Dis[u][v]);
    int k;
    k=Path[u][v];
    printf(" path: %d",u);
    while(k!=v)
    {
        printf(" -> %d",k);
        k=Path[k][v];
    }
    printf(" -> %d\n",v);
    int  D[MAXVEX];
    int  P[MAXVEX];
    int  Vis[MAXVEX];
    Dijkstra(&G,D, P, Vis,0);
    printf("最短路径倒序如下:\n");
    int i,j;
    int v0=0;
    for(i=1; i<G.n; ++i)
    {
        printf("v%d - v%d : ",v0,i);
        j=i;
        while(P[j]!=-1)
        {
            printf("v%d ",P[j]);
            j=P[j];
        }
        printf("v%d ",v0);
        printf("\n");
    }
    printf("\n源点到各顶点的最短路径长度为:\n");
    for(i=1; i<G.n; ++i)
    {
        printf("v%d - v%d : %d \n",G.vertex[0],G.vertex[i],D[i]);
    }

    printf("\n请输入v和u\n");
    scanf("%d,%d",&v,&u);
    Dijkstra2(&G,D, P, Vis, v, u);
    printf("v%d - v%d : ",v,u);
    j=u;
    while(P[j]!=-1)
    {
        printf("v%d ",P[j]);
        j=P[j];
    }
    printf("\n");
    printf("\n请输入u\n");
    scanf("%d",&u);
    for(i=0; i<G.n; i++)
    {
        Dijkstra2(&G,D, P, Vis, i, u);
        printf("v%d - v%d : ",i,u);
        j=u;
        while(P[j]!=-1)
        {
            printf("v%d ",P[j]);
            j=P[j];
        }
        printf("v%d ",i);
        printf("\n");
    }
    return 0;
}
