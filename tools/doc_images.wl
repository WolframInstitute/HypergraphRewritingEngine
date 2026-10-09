(* The image line that follows a documentation example whose result is graphical.

   A page docs/en/<Dir>/<Page>.md shows the output of its example block n as the line
     ![<alt>](<up>images/<Page>/<Page>-<n>.png)
   placed after the block's closing fence, where <up> leads from the page's directory back to
   docs/en (../ for Tutorials and Guides, ../../ for ReferencePages/Symbols). n counts the
   page's non-empty fenced blocks from 1, in the order reference/verify_doc_examples.wls
   evaluates them.

   reference/verify_doc_examples.wls writes these lines (--render-images) and checks them;
   tools/build_docs.wls removes them before conversion, because the evaluated notebook shows
   the output itself. *)

docImageLineQ[line_String] := StringMatchQ[StringTrim[line],
  "![" ~~ Except["]"] ... ~~ "](" ~~ ("../" ..) ~~ "images/" ~~ Except[")"] .. ~~ ".png)"];

(* The image of block n of a page, relative to docs/en. *)
docImageRel[page_String, n_Integer] := "images/" <> page <> "/" <> page <> "-" <> IntegerString[n] <> ".png";

(* The page's text without its image lines, and without the blank line that precedes each. *)
stripDocImageLines[text_String] := Module[{lines = StringSplit[text, "\n", All], keep},
  keep = Table[
    ! docImageLineQ[lines[[k]]] &&
      ! (StringTrim[lines[[k]]] === "" && k < Length[lines] && docImageLineQ[lines[[k + 1]]]),
    {k, Length[lines]}];
  StringRiffle[Pick[lines, keep], "\n"]];
