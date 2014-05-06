SRCS = data_set.cpp index.cpp regression_tree.cpp serialize.cpp boosted.cpp gbdt.cpp
OBJ=$(SRCS:.cpp=.o)

%.o: %.cpp
	g++ -g $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
test:${OBJ}
	g++ -g main.cpp ${OBJ} -o train
	g++ -g classify.cpp ${OBJ} -o classify
clean:
	rm *.o
	rm train classify
