import re
import csv

csv_path = 'foodsafety_data.csv'
re_path = 'data.txt'

def preprocess_text(text):
    # 将日期和时间信息替换成空格
    text = text.replace("年", "-").replace("月", "-").replace("日", "-").replace("时", "-").replace("分", " ").strip()
    text = re.sub("\s+", "", text)
    
    # 删除一些特殊字符
    regex_list = [r"(\d{4}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2})|(\d{4}-\d{1,2}-\d{1,2})|(\d{4}-\d{1,2})",
                    r"[_.!+-=——,$%^，：“”（）:。？、~@#￥%……&*《》<>「」{}【】()/?]",
                    ]
    
    for regex in regex_list:
        pattern = re.compile(regex)
        text = re.sub(pattern, '', text)
    
    return text

with open(csv_path,'r',encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    with open(re_path,'w',encoding='utf-8') as refile:
        for row in csv_reader:
            for i in row:
                row = preprocess_text(i)
                refile.write(i)
            refile.write('\n')
