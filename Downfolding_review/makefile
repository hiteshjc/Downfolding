SHELL := /bin/bash
PROJECT=main
TEX=pdflatex
BIBTEX=bibtex
BUILDTEX=$(PROJECT).tex

all:
	$(BIBTEX) $(BUILDTEX);                      \
	$(TEX) $(BUILDTEX);                      \
	for auxfile in $(wildcard *.aux); 	\
	do                                	\
		$(BIBTEX) `basename $$auxfile .aux`; \
	done;                                        \
	$(TEX) $(BUILDTEX);                         \
	$(TEX) $(BUILDTEX);                         
clean:
	rm -f *.log *.bak *.aux *.bbl *.blg *.idx *.toc *.out *~ *.synctex.gz* *Notes.bib main.pdf honeycomb.pdf Figures/*-converted-to.pdf

honeycomb:
	$(BIBTEX) $(BUILDTEX);                      \
	$(TEX) honeycomb.tex
	for auxfile in $(wildcard *.aux); 	\
	do                                	\
		$(BIBTEX) `basename $$auxfile .aux`; \
	done;                                        \
	$(TEX) honeycomb.tex
	$(TEX) honeycomb.tex
commit:
	cp honeycomb.tex honeycomb.tex.backup
	sed -e "s/\\\input/%\\\input/g" honeycomb.tex.backup > honeycomb.tex
	git commit -am "${message}"
	git push
	rm honeycomb.tex.backup
