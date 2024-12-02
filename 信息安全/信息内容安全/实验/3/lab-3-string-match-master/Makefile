AC = acsmx.c acsmx.h
WM = wm.c wm.h
TOOLS = tools.c tools.h

all: StringMatch

StringMatch: main.c $(AC) $(WM) $(TOOLS)
	gcc -o StringMatch main.c $(AC) $(WM) $(TOOLS) -O3

clean:
	rm -f StringMatch