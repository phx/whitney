window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document.addEventListener("DOMContentLoaded", function() {
  document.querySelectorAll(".md-content").forEach(function(el) {
    el.querySelectorAll('p, span, div').forEach(function(el) {
      if (el.innerHTML.includes('\\') || el.innerHTML.includes('$')) {
        MathJax.typesetPromise([el]);
      }
    });
  });
}); 