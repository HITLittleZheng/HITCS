# optimizer.py

from query_tree import QueryNode

def split_conditions(condition):
    """将复杂条件字符串按 '&' 拆分为单个条件的列表。"""
    return condition.split('&')

def optimize(node):
    if node is None:
        return None

    # 优化子节点
    optimized_children = [optimize(child) for child in node.children]
    node.children = optimized_children

    # 实现选择下推并分解复杂选择条件
    if node.operator == 'SELECT' and len(node.children) == 1:
        child = node.children[0]
        
        # 处理 SELECT -> PROJECTION 的情况
        if child.operator == 'PROJECTION':
            grandchild = child.children[0]
            if grandchild.operator == 'JOIN':
                # 下推 SELECT，尝试作用于 JOIN 的左右子节点
                select_condition = node.condition
                left_child, right_child = grandchild.children
                
                # 分配选择条件到 JOIN 的左、右子节点
                left_conditions = [cond for cond in split_conditions(select_condition) if "ESSN" in cond]
                right_conditions = [cond for cond in split_conditions(select_condition) if "PNAME" in cond]
                
                if left_conditions:
                    left_select_condition = " & ".join(left_conditions)
                    left_child = QueryNode('SELECT', children=[left_child], condition=left_select_condition)
                if right_conditions:
                    right_select_condition = " & ".join(right_conditions)
                    right_child = QueryNode('SELECT', children=[right_child], condition=right_select_condition)
                
                # 更新 JOIN 节点
                optimized_join = QueryNode('JOIN', children=[left_child, right_child])
                # 保留 PROJECTION 操作
                node = QueryNode('PROJECTION', children=[optimized_join], attributes=child.attributes)
            else:
                # 如果 PROJECTION 的子节点不是 JOIN，无法下推，保留原结构
                node.children = [optimize(child)]
        
        # 处理 SELECT -> JOIN 的情况
        elif child.operator == 'JOIN':
            conditions = split_conditions(node.condition)
            left_child, right_child = child.children

            # 分配选择条件到 JOIN 的左、右子节点
            left_conditions = [cond for cond in conditions if "ESSN" in cond or "ENAME" in cond]
            right_conditions = [cond for cond in conditions if "PNAME" in cond or "DNAME" in cond]

            if left_conditions:
                left_select_condition = " & ".join(left_conditions)
                left_child = QueryNode('SELECT', children=[left_child], condition=left_select_condition)
            if right_conditions:
                right_select_condition = " & ".join(right_conditions)
                right_child = QueryNode('SELECT', children=[right_child], condition=right_select_condition)

            # 更新 JOIN 节点的子节点为优化后的选择子节点
            node = QueryNode('JOIN', children=[left_child, right_child])
    
    return node