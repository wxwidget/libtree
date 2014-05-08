SRCS = data_set.cpp index.cpp regression_tree.cpp serialize.cpp boosted.cpp gbdt.cpp
OBJ=$(SRCS:.cpp=.o)

%.o: %.cpp
	g++ -g $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

libtree.a:${OBJ}
	$(AR) rcs $@ $^ 

test:libtree.a
	g++ -g main.cpp -ltree -o train
	g++ -g classify.cpp -ltree -o classify
clean:
	rm *.o
	rm train classify
	rm libtree.a
