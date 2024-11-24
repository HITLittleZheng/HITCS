# main.py
from optimizer import optimize
from query_tree import QueryNode
from sql_parser import parser

def print_tree(node, indent="", last=True):
    """递归打印树结构，使用图形符号提高可读性"""
    if node is not None:
        connector = "└─ " if last else "├─ "
        print(indent + connector + repr(node))
        indent += "   " if last else "│  "
        child_count = len(node.children)
        for i, child in enumerate(node.children):
            print_tree(child, indent, i == (child_count - 1))

def visualize_tree(node, filename):
    """生成并展示查询执行树的图形表示"""
    node.visualize(filename)

def main():
    queries = [
        "SELECT [ ENAME = 'Mary' & DNAME = 'Research' ] ( EMPLOYEE JOIN DEPARTMENT )",
        "PROJECTION [ BDATE ] ( SELECT [ ENAME = 'John' & DNAME = 'Research' ] ( EMPLOYEE JOIN DEPARTMENT ) )",
        "SELECT [ ESSN = '01' ] ( PROJECTION [ ESSN, PNAME ] ( WORKS_ON JOIN PROJECT ) )"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n--- 查询 {i} ---")
        print("原始查询语句:")
        print(query)

        # 解析查询
        tree = parser.parse(query)
        if tree is None:
            print("\n原始查询执行树:")
            print("None")
        else:
            print("\n原始查询执行树:")
            print_tree(tree)

        if tree is not None:
            # 生成并展示图形化的查询执行树
            visualize_tree(tree, f"query_{i}_original")

        # 优化查询
        optimized_tree = optimize(tree)
        if optimized_tree is None:
            print("\n优化后的查询执行树:")
            print("None")
        else:
            print("\n优化后的查询执行树:")
            print_tree(optimized_tree)

            # 生成并展示图形化的优化后查询执行树
            visualize_tree(optimized_tree, f"query_{i}_optimized")

        print("\n" + "-"*50)

if __name__ == "__main__":
    main()