/* =============================================================
   GUIDE — Project Page interactions
   Vanilla JS, no dependencies. Respects prefers-reduced-motion.
   ============================================================= */
(function () {
  'use strict';
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ----------------------- Theme toggle ----------------------- */
  (function theme() {
    var root = document.documentElement;
    var btn = document.getElementById('themeToggle');
    var stored = null;
    try { stored = localStorage.getItem('guide-theme'); } catch (e) {}
    if (stored) {
      root.setAttribute('data-theme', stored);
    } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      root.setAttribute('data-theme', 'dark');
    }
    if (!btn) return;
    btn.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('guide-theme', next); } catch (e) {}
    });
  })();

  /* ----------------------- Mobile nav ------------------------- */
  (function burger() {
    var b = document.getElementById('navBurger');
    var links = document.getElementById('navLinks');
    if (!b || !links) return;
    b.addEventListener('click', function () {
      var open = links.classList.toggle('open');
      b.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    links.addEventListener('click', function (e) {
      if (e.target.tagName === 'A') {
        links.classList.remove('open');
        b.setAttribute('aria-expanded', 'false');
      }
    });
  })();

  /* ------------------- Active section in nav ------------------ */
  (function scrollSpy() {
    var links = Array.prototype.slice.call(document.querySelectorAll('.nav-links a'));
    var map = {};
    links.forEach(function (a) {
      var id = a.getAttribute('href');
      if (id && id.charAt(0) === '#') { var el = document.querySelector(id); if (el) map[id] = a; }
    });
    var ids = Object.keys(map);
    if (!ids.length || !('IntersectionObserver' in window)) return;
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          links.forEach(function (a) { a.classList.remove('active'); });
          var a = map['#' + en.target.id];
          if (a) a.classList.add('active');
        }
      });
    }, { rootMargin: '-45% 0px -50% 0px', threshold: 0 });
    ids.forEach(function (id) { io.observe(document.querySelector(id)); });
  })();

  /* --------------------- Scroll reveal ------------------------ */
  (function reveal() {
    var els = Array.prototype.slice.call(document.querySelectorAll('.reveal'));
    if (reduce || !('IntersectionObserver' in window)) {
      els.forEach(function (el) { el.classList.add('in'); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); }
      });
    }, { threshold: 0.12 });
    els.forEach(function (el) { io.observe(el); });
  })();

  /* ----------------------- Lightbox --------------------------- */
  (function lightbox() {
    var lb = document.getElementById('lightbox');
    if (!lb) return;
    var img = lb.querySelector('img');
    var closeBtn = lb.querySelector('.close');
    function open(src, alt) {
      img.setAttribute('src', src);
      img.setAttribute('alt', alt || '');
      lb.classList.add('open');
      lb.setAttribute('aria-hidden', 'false');
    }
    function close() {
      lb.classList.remove('open');
      lb.setAttribute('aria-hidden', 'true');
      img.setAttribute('src', '');
    }
    document.addEventListener('click', function (e) {
      var z = e.target.closest('.zoomable');
      if (z) {
        var im = z.tagName === 'IMG' ? z : z.querySelector('img');
        if (im) open(im.getAttribute('src'), im.getAttribute('alt'));
      }
    });
    lb.addEventListener('click', function (e) { if (e.target === lb || e.target === closeBtn) close(); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') close(); });
  })();

  /* --------------------- Copy BibTeX -------------------------- */
  (function copyBib() {
    var btn = document.getElementById('copyBib');
    var pre = document.getElementById('bibText');
    if (!btn || !pre) return;
    btn.addEventListener('click', function () {
      var text = pre.innerText.trim();
      var done = function () {
        var old = btn.textContent;
        btn.textContent = 'Copied ✓';
        btn.classList.add('ok');
        setTimeout(function () { btn.textContent = old; btn.classList.remove('ok'); }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, fallback);
      } else { fallback(); }
      function fallback() {
        var ta = document.createElement('textarea');
        ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta); ta.select();
        try { document.execCommand('copy'); done(); } catch (e) {}
        document.body.removeChild(ta);
      }
    });
  })();

  /* ------------------ Qualitative tabs ------------------------ */
  (function qualTabs() {
    var tabs = Array.prototype.slice.call(document.querySelectorAll('.tabs .tab'));
    if (!tabs.length) return;
    var panels = {
      plan: document.getElementById('qpanel-plan'),
      ground: document.getElementById('qpanel-ground')
    };
    tabs.forEach(function (tab) {
      tab.addEventListener('click', function () {
        var ch = tab.getAttribute('data-ch');
        tabs.forEach(function (t) { t.setAttribute('aria-selected', t === tab ? 'true' : 'false'); });
        Object.keys(panels).forEach(function (k) {
          if (panels[k]) panels[k].hidden = (k !== ch);
        });
      });
    });
  })();

  /* ------------------ GUIDE Loop stepper ---------------------- */
  (function guideLoop() {
    var root = document.querySelector('.guide-loop');
    if (!root) return;
    var dots = Array.prototype.slice.call(root.querySelectorAll('.dot'));
    var panels = Array.prototype.slice.call(root.querySelectorAll('.panel'));
    var ba = root.querySelector('.ba');
    var pp = document.getElementById('loopPP');
    var typedEl = root.querySelector('.typed');
    var caret = root.querySelector('.caret');
    var N = panels.length;
    var DWELL = [2800, 3000, 3400, 3000, 4400];

    var i = 0, timer = null, baTimer = null, typeTimer = null;
    var playing = !reduce;
    var FULL = typedEl ? typedEl.textContent : '';

    function clearAll() { clearTimeout(timer); clearTimeout(baTimer); clearTimeout(typeTimer); }

    function typeInstruction() {
      if (!typedEl) return;
      if (reduce) { typedEl.textContent = FULL; if (caret) caret.style.display = 'none'; return; }
      typedEl.textContent = '';
      if (caret) caret.style.display = '';
      var c = 0;
      (function tick() {
        if (c <= FULL.length) { typedEl.textContent = FULL.slice(0, c++); typeTimer = setTimeout(tick, 38); }
      })();
    }

    function show(n) {
      i = (n + N) % N;
      panels.forEach(function (p, k) { p.hidden = k !== i; p.classList.toggle('is-active', k === i); });
      dots.forEach(function (d, k) { d.setAttribute('aria-selected', k === i ? 'true' : 'false'); });
      clearTimeout(baTimer);
      if (i === 4 && ba) {
        ba.dataset.state = 'before';
        if (!reduce) baTimer = setTimeout(function () { ba.dataset.state = 'after'; }, 1700);
      }
      if (i === 0) typeInstruction();
    }

    function schedule() { clearTimeout(timer); timer = setTimeout(advance, DWELL[i]); }
    function advance() { if (playing) { show(i + 1); schedule(); } }

    function pause() { playing = false; clearAll(); if (pp) pp.innerHTML = '▶ Play'; }
    function play() { playing = true; if (pp) pp.innerHTML = '❚❚ Pause'; schedule(); }

    dots.forEach(function (d) {
      d.addEventListener('click', function () { pause(); show(parseInt(d.getAttribute('data-step'), 10)); });
      d.addEventListener('keydown', function (e) {
        var idx = dots.indexOf(d);
        if (e.key === 'ArrowRight') { e.preventDefault(); dots[(idx + 1) % dots.length].focus(); pause(); show(idx + 1); }
        if (e.key === 'ArrowLeft') { e.preventDefault(); dots[(idx - 1 + dots.length) % dots.length].focus(); pause(); show(idx - 1); }
      });
    });
    if (pp) pp.addEventListener('click', function () { playing ? pause() : play(); });

    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (es) {
        es.forEach(function (en) {
          if (en.isIntersecting) { if (playing) schedule(); }
          else { clearTimeout(timer); }
        });
      }, { threshold: 0.35 }).observe(root);
    }

    show(0);
    if (playing) schedule();
    else if (pp) pp.innerHTML = '▶ Play';
  })();

})();
