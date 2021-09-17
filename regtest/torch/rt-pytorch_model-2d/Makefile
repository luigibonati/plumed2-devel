#overload default file to create pythorch model

#include ../../scripts/test.make

test:
	python create-pytorch-model.py
	@../../scripts/run

reset:
	../../scripts/reset

clean:
	rm -fr tmp/ report.txt

valgrind:
	../../scripts/run --valgrind

testclean:
	$(MAKE) test
	rm -fr tmp/
