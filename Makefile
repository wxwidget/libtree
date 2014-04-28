SRCS = data_set.cpp index.cpp regression_tree.cpp 
OBJ=$(SRCS:.cpp=.o)

%.o: %.cpp
	g++ -g $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
test:${OBJ}
	g++ -g main.cpp ${OBJ} -o learn
clean:
	rm *.o
	rm learn
