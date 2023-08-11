all: clean install

clean:
	$(RM) -r ntmg.egg-info/ dist/ build/

install:
	pip3 install .