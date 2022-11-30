all : genvrp

CCC = g++
# CCC = clang++-10
CCFLAGS = -O3 -Wall -std=c++2a
# CCFLAGS = -Wall -std=c++2a -g
TARGETDIR=.

OBJS2 = \
        $(TARGETDIR)/Genetic.o \
        $(TARGETDIR)/Individual.o \
        $(TARGETDIR)/LocalSearch.o \
        $(TARGETDIR)/main.o \
        $(TARGETDIR)/Params.o \
        $(TARGETDIR)/Population.o \
        $(TARGETDIR)/Split.o

$(TARGETDIR)/genvrp: $(OBJS2)
	$(CCC) $(CCFLAGS) -o $(TARGETDIR)/genvrp $(OBJS2)
	
$(TARGETDIR)/Genetic.o: Genetic.h Genetic.cpp
	$(CCC) $(CCFLAGS) -c Genetic.cpp -o $(TARGETDIR)/Genetic.o

$(TARGETDIR)/Individual.o: Individual.h Individual.cpp
	$(CCC) $(CCFLAGS) -c Individual.cpp -o $(TARGETDIR)/Individual.o

$(TARGETDIR)/LocalSearch.o: LocalSearch.h LocalSearch.cpp
	$(CCC) $(CCFLAGS) -c LocalSearch.cpp -o $(TARGETDIR)/LocalSearch.o
	
$(TARGETDIR)/main.o: main.cpp
	$(CCC) $(CCFLAGS) -c main.cpp -o $(TARGETDIR)/main.o

$(TARGETDIR)/Params.o: Params.h Params.cpp
	$(CCC) $(CCFLAGS) -c Params.cpp -o $(TARGETDIR)/Params.o

$(TARGETDIR)/Population.o: Population.h Population.cpp
	$(CCC) $(CCFLAGS) -c Population.cpp -o $(TARGETDIR)/Population.o

$(TARGETDIR)/Split.o: Split.h Split.cpp
	$(CCC) $(CCFLAGS) -c Split.cpp -o $(TARGETDIR)/Split.o

test: genvrp
	./genvrp ../../instances/ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35.txt test.sol -seed 1
test2: genvrp
	mkdir -p tmp
	cp ../../instances_2/For_Eval_50/ORTEC-VRPTW-ASYM-d98c5742-d1-n204-k15.txt tmp
	$(TARGETDIR)/genvrp ./tmp/ORTEC-VRPTW-ASYM-d98c5742-d1-n204-k15.txt 5 
test3: genvrp
	mkdir -p tmp
	cp ../../instances_2/For_Eval_50/ORTEC-VRPTW-ASYM-d98c5742-d1-n204-k15.txt tmp
	$(TARGETDIR)/genvrp ./tmp/ORTEC-VRPTW-ASYM-d98c5742-d1-n204-k15.txt 5 -verbose 1


clean:
	$(RM) \
    $(TARGETDIR)/main.o \
    $(TARGETDIR)/Genetic.o \
    $(TARGETDIR)/Individual.o \
    $(TARGETDIR)/LocalSearch.o \
    $(TARGETDIR)/Params.o \
    $(TARGETDIR)/Population.o \
    $(TARGETDIR)/Split.o

	$(RM) -rf ./tmp
