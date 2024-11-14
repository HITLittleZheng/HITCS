import re
import csv

txt_path = 'data.txt'
re_path = 'data_re.txt'

def preprocess_text(text):
    # 将日期和时间信息替换成空格
    text = text.replace("年", "-").replace("月", "-").replace("日", "-").replace("时", "-").replace("分", " ").strip()
    text = re.sub("\s+", "", text)
    
    # 删除一些特殊字符
    regex_list = [r"(\d{4}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2})|(\d{4}-\d{1,2}-\d{1,2})|(\d{4}-\d{1,2})",
                    r"[_.!+-=——,$%^，：“”（）:。？、~@#￥%……&*《》<>「」{}【】()/]",
                    r"[a-zA-Z]"
                    ]
    
    for regex in regex_list:
        pattern = re.compile(regex)
        text = re.sub(pattern, '', text)
    
    return text

with open(txt_path,'r',encoding='utf-8') as file:
    reader = file.read().splitlines()
    #print(type(reader[0]))
    with open(re_path,'w',encoding='utf-8') as refile:
        for row in reader:
            row = preprocess_text(row)
            refile.write(row)
            refile.write('\n')
    