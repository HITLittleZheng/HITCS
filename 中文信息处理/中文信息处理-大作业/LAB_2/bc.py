def build_corpus(inPath):
    """
    读取txt文件，获取wordList和tagList
    :param inPath:
    :return:
    """
    word_lists = []
    tag_lists = []
    with open(inPath, 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

        return word_lists, tag_lists
