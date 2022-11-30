
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

all : $(TARGETDIR)/hgsvrptw

$(TARGETDIR)/hgsvrptw: $(OBJS2)
	$(CCC) $(CCFLAGS) -o $(TARGETDIR)/hgsvrptw $(OBJS2)
	
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

testd: 
	mkdir -p tmp
	cp ../../instances_2/For_Eval_249/ORTEC-VRPTW-ASYM-35fbecef-d1-n220-k15.txt tmp
	$(TARGETDIR)/hgsvrptw ./tmp/ORTEC-VRPTW-ASYM-35fbecef-d1-n220-k15.txt 30 -verbose 1

test1: hgsvrptw
	mkdir -p tmp
	cp ../../instances_2/For_Eval_249/ORTEC-VRPTW-ASYM-35fbecef-d1-n220-k15.txt tmp
	$(TARGETDIR)/hgsvrptw ./tmp/ORTEC-VRPTW-ASYM-35fbecef-d1-n220-k15.txt 300 -verbose 1 -it 5000

test14: hgsvrptw
	mkdir -p tmp
	cp ../../instances_2/For_Eval_249/ORTEC-VRPTW-ASYM-35fbecef-d1-n220-k15.txt tmp
	$(TARGETDIR)/hgsvrptw ./tmp/ORTEC-VRPTW-ASYM-35fbecef-d1-n220-k15.txt 300 -verbose 1 &
	$(TARGETDIR)/hgsvrptw ./tmp/ORTEC-VRPTW-ASYM-35fbecef-d1-n220-k15.txt 300 -verbose 1 -seed 2 &
	$(TARGETDIR)/hgsvrptw ./tmp/ORTEC-VRPTW-ASYM-35fbecef-d1-n220-k15.txt 300 -verbose 1 -seed 3 &
	$(TARGETDIR)/hgsvrptw ./tmp/ORTEC-VRPTW-ASYM-35fbecef-d1-n220-k15.txt 300 -verbose 1 -seed 4

test2: hgsvrptw
	mkdir -p tmp
	cp ../../instances_2/For_Eval_249/ORTEC-VRPTW-ASYM-3ff97275-d1-n426-k30.txt tmp
	$(TARGETDIR)/hgsvrptw ./tmp/ORTEC-VRPTW-ASYM-3ff97275-d1-n426-k30.txt 600 -verbose 1  -it 5000

test24: hgsvrptw
	mkdir -p tmp
	cp ../../instances_2/For_Eval_249/ORTEC-VRPTW-ASYM-3ff97275-d1-n426-k30.txt tmp
	$(TARGETDIR)/hgsvrptw ./tmp/ORTEC-VRPTW-ASYM-3ff97275-d1-n426-k30.txt 600 -verbose 1 &
	$(TARGETDIR)/hgsvrptw ./tmp/ORTEC-VRPTW-ASYM-3ff97275-d1-n426-k30.txt 600 -verbose 1 -seed 2 &
	$(TARGETDIR)/hgsvrptw ./tmp/ORTEC-VRPTW-ASYM-3ff97275-d1-n426-k30.txt 600 -verbose 1 -seed 3 &
	$(TARGETDIR)/hgsvrptw ./tmp/ORTEC-VRPTW-ASYM-3ff97275-d1-n426-k30.txt 600 -verbose 1 -seed 4

test3: hgsvrptw
	mkdir -p tmp
	cp ../../instances_2/For_Eval_249/ORTEC-VRPTW-ASYM-93ee144d-d1-n688-k38.txt tmp
	$(TARGETDIR)/hgsvrptw ./tmp/ORTEC-VRPTW-ASYM-93ee144d-d1-n688-k38.txt 900 -verbose 1

test34: hgsvrptw
	mkdir -p tmp
	cp ../../instances_2/For_Eval_249/ORTEC-VRPTW-ASYM-93ee144d-d1-n688-k38.txt tmp
	$(TARGETDIR)/hgsvrptw ./tmp/ORTEC-VRPTW-ASYM-93ee144d-d1-n688-k38.txt 900 -verbose 1 &
	$(TARGETDIR)/hgsvrptw ./tmp/ORTEC-VRPTW-ASYM-93ee144d-d1-n688-k38.txt 900 -verbose 1 -seed 2 &
	$(TARGETDIR)/hgsvrptw ./tmp/ORTEC-VRPTW-ASYM-93ee144d-d1-n688-k38.txt 900 -verbose 1 -seed 3 &
	$(TARGETDIR)/hgsvrptw ./tmp/ORTEC-VRPTW-ASYM-93ee144d-d1-n688-k38.txt 900 -verbose 1 -seed 4

test_: hgsvrptw
	mkdir -p tmp
	cp ../../instances_2/For_Eval_50/ORTEC-VRPTW-ASYM-d98c5742-d1-n204-k15.txt tmp
	$(TARGETDIR)/hgsvrptw ./tmp/ORTEC-VRPTW-ASYM-d98c5742-d1-n204-k15.txt 5 -verbose 1

	cp ../../instances_2/For_Eval_50/ORTEC-VRPTW-ASYM-e260d9dc-d1-n480-k45.txt tmp
	$(TARGETDIR)/hgsvrptw ./tmp/ORTEC-VRPTW-ASYM-e260d9dc-d1-n480-k45.txt 5 -verbose 1

	cp ../../instances_2/For_Eval_50/ORTEC-VRPTW-ASYM-dd43a785-d1-n880-k50.txt tmp
	$(TARGETDIR)/hgsvrptw ./tmp/ORTEC-VRPTW-ASYM-dd43a785-d1-n880-k50.txt 5 -verbose 1
	

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
