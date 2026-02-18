PAPER = goal_evolution_paper
PAPER2 = paper2
TEX = pdflatex
BIB = bibtex

.PHONY: all paper2 clean

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	$(TEX) $(PAPER)
	$(BIB) $(PAPER)
	$(TEX) $(PAPER)
	$(TEX) $(PAPER)

paper2: $(PAPER2).pdf

$(PAPER2).pdf: $(PAPER2).tex references.bib
	$(TEX) $(PAPER2)
	$(BIB) $(PAPER2)
	$(TEX) $(PAPER2)
	$(TEX) $(PAPER2)

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log \
	      $(PAPER).out $(PAPER).pdf $(PAPER).toc $(PAPER).fdb_latexmk \
	      $(PAPER).fls $(PAPER).synctex.gz
	rm -f $(PAPER2).aux $(PAPER2).bbl $(PAPER2).blg $(PAPER2).log \
	      $(PAPER2).out $(PAPER2).pdf $(PAPER2).toc $(PAPER2).fdb_latexmk \
	      $(PAPER2).fls $(PAPER2).synctex.gz
