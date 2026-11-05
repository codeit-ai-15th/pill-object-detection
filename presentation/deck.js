/* ============================================================
   deck.js — 공유 네비게이션 엔진 (순수 JS, 외부 의존성 0개)
   각 슬라이드 페이지(NN-slug.html)에서 slides.js 다음에 로드.

   동작 모델 (페이지당 한 슬라이드 + 파일 간 이동):
   - 현재 슬라이드는 파일명을 manifest(window.DECK.slides)와 대조해 판별
   - 다음(→/Space/PageDown): 페이지 내 fragment가 남아 있으면 하나씩 공개,
                             다 공개되면 다음 파일로 이동
   - 이전(←/Shift+Space/PageUp): 이전 파일로 이동(#end 해시 → 도착 시 fragment 전부 공개)
   - Home/End: 처음/마지막, Esc: 목차(index.html)
   - 전체화면 기능은 의도적으로 없음(파일 간 이동 시 브라우저가 전체화면을 해제하므로)
   ============================================================ */
(function(){
  "use strict";
  var DECK = window.DECK || {slides:[]};
  var slides = DECK.slides || [];
  var total = slides.length;

  /* 현재 슬라이드 인덱스: 파일명으로 판별 */
  var file = (location.pathname.split('/').pop() || '').toLowerCase();
  var cur = 0;
  for(var i=0;i<slides.length;i++){
    if((slides[i].file||'').toLowerCase() === file){ cur = i; break; }
  }

  /* 크롬: 슬라이드 번호 카운터 + 진행바 */
  var curEl   = document.querySelector('.cur');
  var totalEl = document.querySelector('.total');
  var bar     = document.querySelector('.progress__bar');
  if(curEl)   curEl.textContent = cur + 1;
  if(totalEl) totalEl.textContent = total;
  if(bar){ bar.style.transform = 'scaleX(' + (total<=1 ? 1 : cur/(total-1)) + ')'; }

  /* 화자 노트: 이 슬라이드의 <aside class="notes"> 내용을 오버레이에 채움 */
  var notesOverlay = document.querySelector('.notes-overlay');
  var notesBody = document.querySelector('.notes-body');
  if(notesBody){
    var note = document.querySelector('.slide .notes');
    notesBody.innerHTML = note ? note.innerHTML : '<em>(노트 없음)</em>';
  }

  /* 페이지 내 fragment (순서: data-fragment-index → DOM 순) */
  var slideEl = document.querySelector('.slide');
  var frags = slideEl ? [].slice.call(slideEl.querySelectorAll('.fragment')) : [];
  frags = frags.map(function(el,idx){
    return {el:el, k: el.dataset.fragmentIndex!==undefined ? +el.dataset.fragmentIndex : 1000+idx};
  }).sort(function(a,b){return a.k-b.k;}).map(function(o){return o.el;});
  var shown = 0;

  function revealAll(){ frags.forEach(function(f){f.classList.add('visible');}); shown = frags.length; }

  /* 이전 슬라이드에서 돌아온 경우(#end) → fragment 전부 공개하고 해시 정리 */
  if(location.hash === '#end'){
    revealAll();
    history.replaceState(null,'',location.pathname + location.search);
  }

  /* ---------- 이동 ---------- */
  function goTo(i){
    i = Math.max(0, Math.min(total-1, i));
    if(i === cur) return;
    location.href = slides[i].file;
  }
  function goNext(){
    if(shown < frags.length){ frags[shown].classList.add('visible'); shown++; return; }
    if(cur < total-1) location.href = slides[cur+1].file;
  }
  function goPrev(){
    if(cur > 0) location.href = slides[cur-1].file + '#end';
  }

  /* ---------- 키보드 ---------- */
  document.addEventListener('keydown', function(e){
    if(e.isComposing || e.ctrlKey || e.metaKey || e.altKey) return;
    var t = e.target.tagName;
    if(t==='INPUT' || t==='TEXTAREA' || e.target.isContentEditable) return;
    var handled = true;
    switch(e.key){
      case 'ArrowRight': case 'PageDown': goNext(); break;
      case 'ArrowLeft':  case 'PageUp':   goPrev(); break;
      case ' ': e.shiftKey ? goPrev() : goNext(); break;
      case 'Home': goTo(0); break;
      case 'End':  goTo(total-1); break;
      case 'Escape': location.href = 'index.html'; break;
      case 'n': case 'N':
        if(notesOverlay) notesOverlay.classList.toggle('show'); break;
      default: handled = false;
    }
    if(handled) e.preventDefault();
  });

  /* ---------- 오토핏: 슬라이드 내용이 무대보다 높으면 축소해 16:9 안에 맞춤 ----------
     무대(.stage)는 높이가 폭×9/16(≈56cqi)로 고정이고 overflow:hidden이라,
     내용이 그보다 높으면(예: 제목 2줄 줄바꿈 + 프레임 다수) 아래가 잘린다.
     fragment는 visibility:hidden이라 레이아웃 공간을 미리 차지하므로 공개 상태와
     무관하게 측정이 안정적이고, transform-origin을 상단 중앙에 둬 제목 위치를
     유지한 채 세로 오버플로만 흡수한다. (표지 .cover는 세로 중앙정렬이라 제외) */
  function fitSlide(){
    if(!slideEl || slideEl.classList.contains('cover')) return;
    slideEl.style.transform = '';
    var avail = slideEl.clientHeight;        /* = 무대 높이(패딩 포함) */
    var need  = slideEl.scrollHeight;        /* 내용 전체 높이 */
    if(need > avail + 1){
      var k = (avail / need) * 0.985;        /* 살짝 여유 */
      slideEl.style.transformOrigin = 'top center';
      slideEl.style.transform = 'scale(' + k + ')';
    }
  }
  window.addEventListener('resize', fitSlide);
  window.addEventListener('load', fitSlide);
  requestAnimationFrame(fitSlide);
  if(document.fonts && document.fonts.ready){ document.fonts.ready.then(fitSlide); }
  setTimeout(fitSlide, 250);

  /* ---------- 터치 스와이프 (임계값 40px, 우세축 판별) ---------- */
  var SWIPE=40, sx=0, sy=0, track=false;
  document.addEventListener('touchstart', function(e){
    if(e.touches.length!==1){ track=false; return; }
    track=true; sx=e.touches[0].clientX; sy=e.touches[0].clientY;
  }, {passive:true});
  document.addEventListener('touchend', function(e){
    if(!track) return; track=false;
    var c=e.changedTouches[0], dx=c.clientX-sx, dy=c.clientY-sy;
    if(Math.abs(dx)<=Math.abs(dy) || Math.abs(dx)<SWIPE) return;
    dx<0 ? goNext() : goPrev();
  }, {passive:true});
})();
