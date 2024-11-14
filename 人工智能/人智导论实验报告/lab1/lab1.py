# -*- coding: utf-8 -*-
import os

'''
monkey  -1:A    0:B     1:c
box	-1:A    0:B 	1:C
banana	-1:A	0:B	1:C
monbox	-1:monkey not on	1:monkey on the box && banana picked
'''

dict = {'-1':'A','0':'B','1':'C'}
class State:
	def __init__(state, monkey=-1-1,box=0,banana=1,monbox=-1):
		state.monkey = monkey
		state.box = box
		state.banana = banana
		state.monbox = monbox

def check(state):
    if state.monbox != -1 and state.monbox != 1 \
	or state.banana != -1 and state.banana != 1 and state.banana != 0  \
	or state.monkey != -1 and state.monkey != 1 and state.monkey != 0 \
	or state.box != -1 and state.box != 1 and state.box != 0 \
	or state.monbox == 1 and state.monkey != state.box :
        print("输入状态错误")
        os._exit(0)
    else:
	    return state

def MonkeyDown(state,step):
	print ("步数" + str(step) + ":猴子从箱子上下来")
	state.monbox = -1
	return state

def MonkeyUp(state,step):
	print ("步数" + str(step) + ":猴子爬到箱子上去")
	state.monbox = 1
	return state

def MoveBox(state,step):
	print ("步数" + str(step) + ":猴子将箱子从" + str(dict[str(state.box)]) + "移动到" + str(dict[str(state.banana)]))
	state.monkey = state.banana
	state.box = state.banana
	return state

def MoveMonkey(state,step):
	print ("步数" + str(step) + ":猴子从" + str(dict[str(state.monkey)]) + "移动到" + str(dict[str(state.box)]))
	state.monkey = state.box
	return state

def Action(state):
	step = 1
	if state.monkey == state.box and state.box == state.banana and state.monbox==1:
		print ("步数" + str(step) + ":猴子将香蕉摘下来")
		os._exit(0)
	if state.monbox==1:	#猴子在箱子上面站着
		if state.box!=state.banana:	#箱子和香蕉不在一起
			state = MonkeyUp(MoveBox(MonkeyDown(state,step),step+1),step+2)
			step = step + 3
	if state.monbox==-1:	#猴子不在箱子上面，此时猴子和箱子的位置不一定在一起
		if state.box!=state.banana:	#箱子和香蕉不在一起
			if state.monkey!=state.box:	#猴子和箱子不在一起
				MonkeyUp(MoveBox(MoveMonkey(state,step),step+1),step+2)
				step = step + 3
			else:				#猴子和箱子在一起
				MonkeyUp(MoveBox(state,step),step+1)
				step = step + 2
		else:				#箱子和香蕉在一起
			if state.monkey!=state.box:
				MonkeyUp(MoveMonkey(state,step),step + 1)
				step = step + 2
			else:
				MonkeyUp(state,step)
				step = step + 1
	print ("步数" + str(step) + ":猴子将香蕉摘下来")

if __name__ == "__main__":
	s = input("please 1 0input: monkey[-1,0,1] box[-1,0,1] banana[-1,0,1] monbox[-1,1]\n")
	states = s.split(" ")
	state = State(int(states[0]), int(states[1]),int(states[2]),int(states[3]))
	#print states
	Action(check(state))