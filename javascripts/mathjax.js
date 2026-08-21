// NOTE: We intentionally omit the `options.ignoreHtmlClass` / `processHtmlClass`
// restriction that many MkDocs-Material examples include. The default config
// limits MathJax to elements with class "arithmatex", which works for .md files
// (pymdownx.arithmatex wraps math in that class), but breaks Jupyter notebooks
// rendered by mkdocs-jupyter — notebook HTML contains raw $/$$ delimiters inside
// deeply nested elements without the "arithmatex" class, so MathJax never
// processes them. Removing the class restriction lets MathJax scan the full page.
//
// We also include both \(...\) and $...$ delimiters because:
//   - pymdownx.arithmatex (for .md files) converts $...$ → \(...\) and $$...$$ → \[...\]
//   - mkdocs-jupyter (for .ipynb files) passes $...$ and $$...$$ through as-is
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  }
};

document$.subscribe(() => {
  if (!window.MathJax || !MathJax.startup || !MathJax.startup.promise) {
    return;
  }

  MathJax.startup.promise
    .then(() => {
      MathJax.startup?.output?.clearCache?.();
      MathJax.typesetClear?.();
      MathJax.texReset?.();
      MathJax.typesetPromise?.();
    })
    .catch(() => {
      // Silently ignore MathJax startup errors to avoid breaking navigation
    });
})
