window.MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)']],
    displayMath: [['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true,
    packages: {'[+]': ['ams']}
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  loader: {
    load: ['[tex]/ams']
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