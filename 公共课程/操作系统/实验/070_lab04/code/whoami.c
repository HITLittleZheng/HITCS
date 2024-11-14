#define __LIBRARY__
#include <unistd.h>
#include <errno.h>
#include <stdio.h>

_syscall2(int, whoami, char*, name, unsigned int, size);

int main(int argc, char ** argv)
{
    char t[30];
    whoami(t, 30);
    printf("%s\n", t);
    return 0;
}
