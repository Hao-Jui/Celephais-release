CXX = g++
CXXFLAGS = -O3 -std=c++17

SRC = 1dtov.cpp
SETUP_SRC = ../margherita/table/setup_cold_table.cc ../margherita/pwp/setup_polytrope.cc
EXE = tov

all:
	$(CXX) $(CXXFLAGS) $(SRC) $(SETUP_SRC) -o $(EXE)

clean:
	rm -f $(EXE)
