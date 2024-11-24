# parser.py

import ply.lex as lex
import ply.yacc as yacc
from query_tree import QueryNode

# 定义保留字
reserved = {
    'SELECT': 'SELECT',
    'PROJECTION': 'PROJECTION',
    'JOIN': 'JOIN',
    'AVG': 'AVG',
}

# 词法分析器
tokens = [
    'LPAREN',
    'RPAREN',
    'LBRACKET',
    'RBRACKET',
    'COMMA',
    'AND',
    'EQUALS',
    'LT',
    'IDENT',
    'STRING',
    'NUMBER'
] + list(reserved.values())

# 定义简单的符号
t_LPAREN    = r'\('
t_RPAREN    = r'\)'
t_LBRACKET  = r'\['
t_RBRACKET  = r'\]'
t_COMMA     = r','
t_AND       = r'&'
t_EQUALS    = r'='
t_LT        = r'<'

t_ignore = ' \t'

# 定义字符串
def t_STRING(t):
    r'\'[^\']*\''
    t.value = t.value.strip('\'')
    return t

# 定义数字
def t_NUMBER(t):
    r'\d+'
    t.value = t.value
    return t

# 定义标识符和保留字
def t_IDENT(t):
    r'[A-Za-z_][A-Za-z0-9_]*'
    t.type = reserved.get(t.value.upper(), 'IDENT')  # 保留字优先
    return t

# 跳过换行符
def t_newline(t):
    r'\n+'
    pass

# 处理非法字符
def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

lexer = lex.lex()

# 定义运算符优先级
precedence = (
    ('left', 'AND'),
    ('left', 'JOIN'),
    ('left', 'SELECT', 'PROJECTION', 'AVG'),
)

# 语法分析器

def p_query(p):
    '''query : operation'''
    p[0] = p[1]

def p_operation_select(p):
    '''operation : SELECT LBRACKET condition RBRACKET LPAREN operation RPAREN'''
    node = QueryNode('SELECT', children=[p[6]], condition=p[3])
    p[0] = node

def p_operation_projection(p):
    '''operation : PROJECTION LBRACKET attributes RBRACKET LPAREN operation RPAREN'''
    node = QueryNode('PROJECTION', children=[p[6]], attributes=p[3])
    p[0] = node

def p_operation_avg(p):
    '''operation : AVG LBRACKET IDENT RBRACKET LPAREN operation RPAREN'''
    node = QueryNode('AVG', children=[p[6]], attributes=[p[3]])
    p[0] = node

def p_operation_join(p):
    '''operation : operation JOIN operation'''
    left = p[1]
    right = p[3]
    node = QueryNode('JOIN', children=[left, right])
    p[0] = node

def p_operation_table(p):
    '''operation : IDENT'''
    node = QueryNode('TABLE', attributes=[p[1]])
    p[0] = node

def p_condition_and(p):
    '''condition : condition AND condition'''
    p[0] = f"{p[1]} & {p[3]}"

def p_condition_equals(p):
    '''condition : IDENT EQUALS value'''
    p[0] = f"{p[1]} = {p[3]}"

def p_condition_lt(p):
    '''condition : IDENT LT value'''
    p[0] = f"{p[1]} < {p[3]}"

def p_value(p):
    '''value : STRING
             | NUMBER'''
    p[0] = p[1]

def p_attributes_multiple(p):
    '''attributes : attributes COMMA IDENT'''
    p[0] = p[1] + [p[3]]

def p_attributes_single(p):
    '''attributes : IDENT'''
    p[0] = [p[1]]

def p_error(p):
    if p:
        print(f"Syntax error at '{p.value}'")
    else:
        print("Syntax error at EOF")

parser = yacc.yacc()