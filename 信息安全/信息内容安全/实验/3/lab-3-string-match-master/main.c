#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include "wm.h"  // 引入模式匹配算法的相关头文件
#include "acsmx.h"
#include "tools.h"
#define FileNum_Char 50  // 文件名的最大字符长度

int nline;
int nfound;

typedef struct {
    char **patterns;
    int count;
    char *text_file;
} PatternSet;

void wm_pattern_matching(PatternSet *set) {
    nfound = 0;
    nline = 1;
    FILE *ft;
    char text[4096];
    struct timeval starttv, endtv;
    WM_STRUCT *p = wmNew();

    gettimeofday(&starttv, NULL);
    // 添加长模式到WM结构体
    for (int i = 0; i < set->count; i++) {
        wmAddPattern(p, (unsigned char *)set->patterns[i], strlen(set->patterns[i]));
    }
    wmPrepPatterns(p);
    gettimeofday(&endtv, NULL);
    printf("[WM] Initialization_time=%ldms\n",
           endtv.tv_sec * 1000 + endtv.tv_usec / 1000 - starttv.tv_sec * 1000 - starttv.tv_usec / 1000);
	print_memory_usage();
    // 打开文本文件
    if ((ft = fopen(set->text_file, "r")) == NULL) {
        fprintf(stderr, "open %s failed\n", set->text_file);
        return;
    }

    gettimeofday(&starttv, NULL);
    while (fscanf(ft, "%s", text) == 1) {
        wmSearch(p, (unsigned char *)text, strlen(text));
        nline++;
    }
    gettimeofday(&endtv, NULL);

    printf("[WM] Search_time=%ldms Hit_num=%d\n",
           endtv.tv_sec * 1000 + endtv.tv_usec / 1000 - starttv.tv_sec * 1000 - starttv.tv_usec / 1000, nfound);
    fclose(ft);
    wmFree(p);
}

void ac_pattern_matching(PatternSet *set) {
    nfound = 0;
    nline = 1;
    FILE *ft;
    char text[4096];
    struct timeval starttv, endtv;
    ACSM_STRUCT *acsm = acsmNew();

    gettimeofday(&starttv, NULL);
    // 添加短模式到ACSM结构体
    for (int i = 0; i < set->count; i++) {
        acsmAddPattern(acsm, (unsigned char *)set->patterns[i], strlen(set->patterns[i]), 1);
    }
    acsmCompile(acsm);
    gettimeofday(&endtv, NULL);
    printf("[AC] Initialization_time=%ldms\n",
           endtv.tv_sec * 1000 + endtv.tv_usec / 1000 - starttv.tv_sec * 1000 - starttv.tv_usec / 1000);
	print_memory_usage();
    // 打开文本文件
    if ((ft = fopen(set->text_file, "r")) == NULL) {
        fprintf(stderr, "open %s failed\n", set->text_file);
        return;
    }

    gettimeofday(&starttv, NULL);
    while (fscanf(ft, "%s", text) == 1) {
        acsmSearch(acsm, (unsigned char *)text, strlen(text), PrintMatch);
        nline++;
    }
    gettimeofday(&endtv, NULL);

    printf("[AC] Search_time=%ldms Hit_num=%d\n",
           endtv.tv_sec * 1000 + endtv.tv_usec / 1000 - starttv.tv_sec * 1000 - starttv.tv_usec / 1000, nfound);
    fclose(ft);
    acsmFree(acsm);
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <pattern_file> <text_file> <method_choice> [length_threshold]\n", argv[0]);
        exit(1);
    }

    char *PatternFileName = argv[1];
    char *TextFileName = argv[2];
    int method_choice = atoi(argv[3]);
    int length_threshold = 0;

    if (method_choice == 3) {
        if (argc < 5) {
            fprintf(stderr, "Usage: %s <pattern_file> <text_file> <method_choice> [length_threshold]\n", argv[0]);
            exit(1);
        }
        length_threshold = atoi(argv[4]);
    }

    FILE *fp;
    char pattern[4096];
    char **short_patterns = NULL;
    char **long_patterns = NULL;
    int short_count = 0, long_count = 0;

    // 打开模式文件并读取模式串
    if ((fp = fopen(PatternFileName, "r")) == NULL) {
        fprintf(stderr, "open %s failed\n", PatternFileName);
        exit(0);
    }

    // 扫描模式文件以确定长模式和短模式的数量
    while (fscanf(fp, "%s", pattern) == 1) {
        if (method_choice == 3 && strlen(pattern) > length_threshold) {
            long_count++;
        } else if (method_choice == 3) {
            short_count++;
        } else if (method_choice == 2) {
            long_count++;
        } else if (method_choice == 1) {
            short_count++;
        }
    }
    rewind(fp);

    // 为模式分配内存
    if (method_choice == 3 || method_choice == 1) {
        short_patterns = (char **)malloc(short_count * sizeof(char *));
    }
    if (method_choice == 3 || method_choice == 2) {
        long_patterns = (char **)malloc(long_count * sizeof(char *));
    }
    if ((method_choice == 3 && (!short_patterns || !long_patterns)) ||
        (method_choice == 1 && !short_patterns) ||
        (method_choice == 2 && !long_patterns)) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // 读取模式并分类
    short_count = 0;
    long_count = 0;
    while (fscanf(fp, "%s", pattern) == 1) {
        if (method_choice == 3 && strlen(pattern) > length_threshold) {
            long_patterns[long_count++] = strdup(pattern);
        } else if (method_choice == 3) {
            short_patterns[short_count++] = strdup(pattern);
        } else if (method_choice == 2) {
            long_patterns[long_count++] = strdup(pattern);
        } else if (method_choice == 1) {
            short_patterns[short_count++] = strdup(pattern);
        }
    }
    fclose(fp);

    // 执行匹配
    if (method_choice == 1) {
        PatternSet short_set = {short_patterns, short_count, TextFileName};
        ac_pattern_matching(&short_set);
    } else if (method_choice == 2) {
        PatternSet long_set = {long_patterns, long_count, TextFileName};
        wm_pattern_matching(&long_set);
    } else if (method_choice == 3) {
        PatternSet short_set = {short_patterns, short_count, TextFileName};
        PatternSet long_set = {long_patterns, long_count, TextFileName};
        wm_pattern_matching(&long_set);
        ac_pattern_matching(&short_set);
    }

    // 释放模式串内存
    if (method_choice == 1 || method_choice == 3) {
        for (int i = 0; i < short_count; i++) {
            free(short_patterns[i]);
        }
        free(short_patterns);
    }
    if (method_choice == 2 || method_choice == 3) {
        for (int i = 0; i < long_count; i++) {
            free(long_patterns[i]);
        }
        free(long_patterns);
    }

    return 0;
}