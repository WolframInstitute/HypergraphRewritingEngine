# Makefile for "Rewriting the Universe" paper

MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f $(MAIN).pdf

view: $(MAIN).pdf
	xdg-open $(MAIN).pdf 2>/dev/null || open $(MAIN).pdf 2>/dev/null || evince $(MAIN).pdf

# For continuous compilation during writing
watch:
	latexmk -pdf -pvc $(MAIN).tex

# Quick single-pass compile (no bibliography)
quick:
	$(LATEX) $(MAIN)

# arxiv submission package
arxiv: $(MAIN).pdf
	mkdir -p arxiv-submission
	cp $(MAIN).tex arxiv-submission/
	cp references.bib arxiv-submission/
	cp -r figures arxiv-submission/ 2>/dev/null || true
	tar -czvf arxiv-submission.tar.gz arxiv-submission/
	rm -rf arxiv-submission/
	@echo "Created arxiv-submission.tar.gz"
