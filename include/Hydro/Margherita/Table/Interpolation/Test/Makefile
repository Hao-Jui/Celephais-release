CXX ?= c++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic

test_inverse_interp: test_inverse_interp.cpp ../inverse_interp.hh ../linear_interp.hh
	$(CXX) $(CXXFLAGS) test_inverse_interp.cpp -o $@
	./$@

.PHONY: test clean
test: test_inverse_interp

clean:
	rm -f test_inverse_interp
