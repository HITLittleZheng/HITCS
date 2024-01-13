#include <stdio.h>
#include <stdlib.h>

int main()
{
    int i;
    char c;
    int flag =0;
    char str[16];
    int byte[16];
    FILE *fp;
    fp=fopen("hello.c","r+");
    if(fp==NULL)
    {
        printf("Fail!");
    }
    else
    {
        c=fgetc(fp);
        while(c!=EOF)
        {
            for(i=0;i<16;i++)
            {
                str[i]=c;
                byte[i]=(int) c;
                c=fgetc(fp);
            }
             for(i=0;i<16;i++)
            {
                if(str[i]==EOF)
                    break;
                if(str[i]=='\n')
                {
                    printf("  \\n");
                }

                else
                 printf("%4c",str[i]);
            }
            printf("\n");
            for(i=0;i<16;i++)
            {
                 if(byte[i]==EOF)
                    break;
                printf("%4d",byte[i]);
            }
             printf("\n");
        }
    }
    return 0;
}
