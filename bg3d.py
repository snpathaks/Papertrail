"""3D knowledge-network background for PaperTrail.

Teal nodes = papers, ~10% amber nodes = research gaps, thin lines = links.
Injects ONE fixed Three.js canvas into the parent page (guarded against
Streamlit reruns). Respects prefers-reduced-motion.
"""
import streamlit.components.v1 as components
import streamlit as st

NODES = 70  # lower (e.g. 50) if the laptop heats up while the LLM runs

_JS = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const P = window.parent, D = P.document;
if (!P.__ptBg) {
  P.__ptBg = true;
  const c = D.createElement('canvas');
  c.id = 'pt-bg';
  c.style.cssText = 'position:fixed;inset:0;width:100vw;height:100vh;z-index:0;pointer-events:none';
  D.body.prepend(c);

  const renderer = new THREE.WebGLRenderer({canvas: c, alpha: true, antialias: true});
  renderer.setPixelRatio(Math.min(P.devicePixelRatio, 2));
  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
  cam.position.z = 14;

  const N = __NODES__, pos = [], col = [];
  const teal = new THREE.Color('#2A9D8F'), amber = new THREE.Color('#E9A23B');
  for (let i = 0; i < N; i++) {
    const r = 9 * Math.cbrt(Math.random());
    const t = Math.random() * Math.PI * 2, p = Math.acos(2 * Math.random() - 1);
    pos.push(r * Math.sin(p) * Math.cos(t), r * Math.sin(p) * Math.sin(t), r * Math.cos(p));
    const k = Math.random() < 0.1 ? amber : teal;
    col.push(k.r, k.g, k.b);
  }
  const pg = new THREE.BufferGeometry();
  pg.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  pg.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  const points = new THREE.Points(pg, new THREE.PointsMaterial({
    size: 0.2, vertexColors: true, transparent: true, opacity: 0.9}));

  const seg = [];
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    const dx = pos[i*3]-pos[j*3], dy = pos[i*3+1]-pos[j*3+1], dz = pos[i*3+2]-pos[j*3+2];
    if (dx*dx + dy*dy + dz*dz < 10)
      seg.push(pos[i*3],pos[i*3+1],pos[i*3+2],pos[j*3],pos[j*3+1],pos[j*3+2]);
  }
  const lg = new THREE.BufferGeometry();
  lg.setAttribute('position', new THREE.Float32BufferAttribute(seg, 3));
  const lines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({
    color: 0x2A9D8F, transparent: true, opacity: 0.22}));

  const group = new THREE.Group();
  group.add(points, lines);
  scene.add(group);

  let mx = 0, my = 0;
  P.addEventListener('mousemove', e => {
    mx = (e.clientX / P.innerWidth - 0.5) * 2;
    my = (e.clientY / P.innerHeight - 0.5) * 2;
  });
  function resize() {
    renderer.setSize(P.innerWidth, P.innerHeight, false);
    cam.aspect = P.innerWidth / P.innerHeight;
    cam.updateProjectionMatrix();
  }
  P.addEventListener('resize', resize); resize();

  const reduce = P.matchMedia('(prefers-reduced-motion: reduce)').matches;
  (function loop() {
    if (!reduce) group.rotation.y += 0.0012;
    group.rotation.x += (my * 0.25 - group.rotation.x) * 0.03;
    cam.position.x += (mx * 1.5 - cam.position.x) * 0.03;
    cam.lookAt(0, 0, 0);
    renderer.render(scene, cam);
    P.requestAnimationFrame(loop);
  })();
}
</script>
"""


def background_3d():
    """Call once per run, after st.set_page_config()."""
    components.html(_JS.replace("__NODES__", str(NODES)), height=0)
    st.iframe(_JS.replace("__NODES__", str(NODES)), height=0, scrolling=False)