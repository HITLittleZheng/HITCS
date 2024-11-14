#include <string.h>
#include <errno.h>
#include <asm/segment.h>

char msg[24];

int sys_iam(const char * name)
{
    char tep[26];
    int i = 0;
    for(; i < 26; i++)
    {
        tep[i] = get_fs_byte(name+i);
        if(tep[i] == '\0')  break;
    }

    if (i > 23) return -(EINVAL);

    strcpy(msg, tep);
    return i;
}

int sys_whoami(char * name, unsigned int size)
{
    int len = 0;
    for (;msg[len] != '\0'; len++);
    
    if (len > size) 
    {
        return -(EINVAL);
    }
    
    int i = 0;
    for(i = 0; i < size; i++)
    {
        put_fs_byte(msg[i], name+i);
        if(msg[i] == '\0') break;
    }
    return i;
}
