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
	rm -f *.log *.bak *.aux *.bbl *.blg *.idx *.toc *.out *~ *.synctex.gz* *Notes.bib
