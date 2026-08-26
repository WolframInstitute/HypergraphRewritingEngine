# Makefile for "Rewriting the Universe" paper

MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean view

all: $(MAIN).pdf

# THE TABLES ARE INPUTS, so the pdf depends on them. Without this, regenerating every fragment
# and running `make` reports "Nothing to be done": main.tex has not changed, and the pdf that
# ships still carries the previous run's numbers. The fragments are the numbers.
TABLES := $(wildcard tables/*.tex)

$(MAIN).pdf: $(MAIN).tex references.bib $(TABLES)
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
	# tables/ is not optional: main.tex \input's every fragment in it, so a submission
	# without them does not compile at the other end. Copied unconditionally, and the
	# build below fails loudly rather than shipping a tarball that cannot be built.
	mkdir -p arxiv-submission/tables
	cp tables/*.tex arxiv-submission/tables/
	cd arxiv-submission && $(LATEX) -interaction=nonstopmode $(MAIN) >/dev/null 2>&1 \
		|| { echo "error: the submission tree does not compile on its own"; exit 1; }
	rm -f arxiv-submission/*.aux arxiv-submission/*.log arxiv-submission/*.out
	tar -czf arxiv-submission.tar.gz arxiv-submission/
	rm -rf arxiv-submission/
	@echo "Created arxiv-submission.tar.gz (compiled from its own copy before packing)"
