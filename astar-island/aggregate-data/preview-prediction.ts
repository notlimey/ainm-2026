import "@std/dotenv/load";

const BIN_DIR = "../simulation/data";
const DATA_DIR = "data";
const NUM_CLASSES = 6;

function terrainToClass(code: number): number {
	switch (code) {
		case 1: return 1;
		case 2: return 2;
		case 3: return 3;
		case 4: return 4;
		case 5: return 5;
		default: return 0;
	}
}

function readPredictionBin(path: string) {
	const data = Deno.readFileSync(path);
	const view = new DataView(data.buffer);
	const round = view.getInt32(6, true);
	const seed = view.getInt32(10, true);
	const W = view.getInt32(14, true);
	const H = view.getInt32(18, true);

	const prediction: number[][][] = [];
	let offset = 22;
	for (let y = 0; y < H; y++) {
		const row: number[][] = [];
		for (let x = 0; x < W; x++) {
			const cell: number[] = [];
			for (let c = 0; c < NUM_CLASSES; c++) {
				cell.push(view.getFloat32(offset, true));
				offset += 4;
			}
			row.push(cell);
		}
		prediction.push(row);
	}
	return { round, seed, W, H, prediction };
}

function readGridBin(path: string, wantRound: number, wantSeed: number) {
	const data = Deno.readFileSync(path);
	const view = new DataView(data.buffer);
	let offset = 6;
	const count = view.getUint32(offset, true); offset += 4;

	for (let i = 0; i < count; i++) {
		const round = view.getInt32(offset, true); offset += 4;
		const seed = view.getInt32(offset, true); offset += 4;
		const W = view.getInt32(offset, true); offset += 4;
		const H = view.getInt32(offset, true); offset += 4;

		if (round === wantRound && seed === wantSeed) {
			const grid: number[][] = [];
			for (let y = 0; y < H; y++) {
				const row: number[] = [];
				for (let x = 0; x < W; x++) {
					row.push(view.getInt32(offset, true));
					offset += 4;
				}
				grid.push(row);
			}
			return { W, H, grid };
		}
		offset += W * H * 4;
	}
	return null;
}

function readGroundTruthBin(path: string, wantRound: number, wantSeed: number) {
	try {
		const data = Deno.readFileSync(path);
		const view = new DataView(data.buffer);
		let offset = 6;
		const count = view.getUint32(offset, true); offset += 4;

		for (let i = 0; i < count; i++) {
			const round = view.getInt32(offset, true); offset += 4;
			const seed = view.getInt32(offset, true); offset += 4;
			const W = view.getInt32(offset, true); offset += 4;
			const H = view.getInt32(offset, true); offset += 4;

			if (round === wantRound && seed === wantSeed) {
				const gt: number[][][] = [];
				for (let y = 0; y < H; y++) {
					const row: number[][] = [];
					for (let x = 0; x < W; x++) {
						const cell: number[] = [];
						for (let c = 0; c < NUM_CLASSES; c++) {
							cell.push(view.getFloat32(offset, true));
							offset += 4;
						}
						row.push(cell);
					}
					gt.push(row);
				}
				return gt;
			}
			offset += W * H * NUM_CLASSES * 4;
		}
	} catch { /* */ }
	return null;
}

const CLASSES = [
	{ name: "Empty/Ocean/Plains", color: [200, 184, 138] },
	{ name: "Settlement", color: [212, 118, 10] },
	{ name: "Port", color: [14, 116, 144] },
	{ name: "Ruin", color: [127, 29, 29] },
	{ name: "Forest", color: [45, 90, 39] },
	{ name: "Mountain", color: [107, 114, 128] },
];

interface SeedData {
	round: number;
	seed: number;
	W: number;
	H: number;
	prediction: number[][][];
	ground_truth: number[][][] | null;
	initial_grid: number[][] | null;
}

function generateHTML(seeds: SeedData[]): string {
	const seedsJson = seeds.map(s => ({
		round: s.round,
		seed: s.seed,
		width: s.W,
		height: s.H,
		prediction: s.prediction,
		ground_truth: s.ground_truth,
		initial_grid: s.initial_grid,
	}));

	return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Astar Island — Prediction Preview</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'SF Mono', 'Fira Code', monospace; background: #0a0a0a; color: #e0e0e0; padding: 20px; }
h1 { font-size: 18px; margin-bottom: 12px; color: #fff; }
.controls { display: flex; gap: 12px; align-items: center; margin-bottom: 16px; flex-wrap: wrap; }
.controls label { font-size: 12px; color: #888; }
.controls select, .controls input { background: #1a1a1a; color: #e0e0e0; border: 1px solid #333; padding: 4px 8px; font-family: inherit; font-size: 12px; border-radius: 3px; }
.grid-row { display: flex; gap: 20px; margin-bottom: 16px; flex-wrap: wrap; }
.grid-box { display: flex; flex-direction: column; align-items: center; }
.grid-box .label { font-size: 11px; color: #666; margin-bottom: 4px; }
canvas { image-rendering: pixelated; border: 1px solid #333; cursor: crosshair; }
.info-panel { background: #111; border: 1px solid #333; padding: 12px; font-size: 12px; margin-top: 8px; border-radius: 4px; min-height: 80px; }
.bar-row { display: flex; align-items: center; gap: 6px; margin: 2px 0; }
.bar-label { width: 120px; font-size: 11px; }
.bar-bg { width: 200px; height: 12px; background: #222; border-radius: 2px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 2px; }
.bar-val { font-size: 11px; width: 50px; text-align: right; }
.legend { display: flex; gap: 12px; margin: 8px 0; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 4px; font-size: 11px; }
.legend-swatch { width: 12px; height: 12px; border-radius: 2px; border: 1px solid #555; }
.stats { display: grid; grid-template-columns: repeat(auto-fill, 140px); gap: 8px; margin: 8px 0; }
.stat-card { background: #151515; border: 1px solid #282828; padding: 8px; border-radius: 4px; text-align: center; }
.stat-card .val { font-size: 18px; font-weight: bold; color: #fff; }
.stat-card .lbl { font-size: 10px; color: #666; margin-top: 2px; }
</style>
</head>
<body>
<h1>Prediction Preview</h1>
<div class="controls">
  <label>Round/Seed: <select id="seedSelect"></select></label>
  <label>View: <select id="viewMode">
    <option value="argmax">Argmax</option>
    <option value="entropy">Entropy</option>
    <option value="diff">Diff (if ground truth)</option>
    <option value="class">Single class</option>
  </select></label>
  <label id="classWrap" style="display:none">Class: <select id="classSelect">
    <option value="0">Empty/Plains</option><option value="1">Settlement</option>
    <option value="2">Port</option><option value="3">Ruin</option>
    <option value="4">Forest</option><option value="5">Mountain</option>
  </select></label>
  <label>Zoom: <input id="zoom" type="range" min="8" max="24" value="14"></label>
</div>
<div class="legend" id="legend"></div>
<div class="stats" id="stats"></div>
<div class="grid-row">
  <div class="grid-box"><div class="label">Initial State</div><canvas id="cvInit"></canvas></div>
  <div class="grid-box"><div class="label">Prediction</div><canvas id="cvPred"></canvas></div>
  <div class="grid-box" id="gtBox" style="display:none"><div class="label">Ground Truth</div><canvas id="cvGt"></canvas></div>
</div>
<div class="info-panel" id="info">Hover over a cell</div>
<script>
const SEEDS = ${JSON.stringify(seedsJson)};
const CN = ${JSON.stringify(CLASSES.map(c => c.name))};
const CC = ${JSON.stringify(CLASSES.map(c => c.color))};
const sel = document.getElementById('seedSelect');
const vm = document.getElementById('viewMode');
const cs = document.getElementById('classSelect');
const cw = document.getElementById('classWrap');
const zm = document.getElementById('zoom');
const info = document.getElementById('info');

SEEDS.forEach((s,i) => { const o=document.createElement('option'); o.value=i; o.textContent='r'+s.round+'.s'+s.seed; sel.appendChild(o); });

function tc(code) { return code===1?1:code===2?2:code===3?3:code===4?4:code===5?5:0; }
function ent(p) { let h=0; for(const v of p) if(v>0) h-=v*Math.log2(v); return h; }
function klDiv(p,q) { let k=0; for(let i=0;i<p.length;i++) if(p[i]>0) k+=p[i]*Math.log(p[i]/Math.max(q[i],1e-10)); return k; }
function argmax(p) { let b=0; for(let i=1;i<p.length;i++) if(p[i]>p[b]) b=i; return b; }
function colC(c,a) { return CC[c].map(v=>v*a+10*(1-a)); }

function draw(cv,w,h,sz,fn) {
  cv.width=w*sz; cv.height=h*sz;
  const ctx=cv.getContext('2d'), img=ctx.createImageData(w*sz,h*sz);
  for(let y=0;y<h;y++) for(let x=0;x<w;x++) {
    const [r,g,b]=fn(x,y);
    for(let dy=0;dy<sz;dy++) for(let dx=0;dx<sz;dx++) {
      const i=((y*sz+dy)*w*sz+x*sz+dx)*4;
      img.data[i]=r; img.data[i+1]=g; img.data[i+2]=b; img.data[i+3]=255;
    }
  }
  ctx.putImageData(img,0,0);
}

function render() {
  const s=SEEDS[sel.value]; if(!s) return;
  const {width:w,height:h,prediction:pr,ground_truth:gt,initial_grid:ig}=s;
  const sz=parseInt(zm.value), mode=vm.value, ci=parseInt(cs.value);
  cw.style.display=mode==='class'?'':'none';
  document.getElementById('gtBox').style.display=gt?'':'none';

  draw(document.getElementById('cvInit'),w,h,sz,(x,y)=> ig?CC[tc(ig[y][x])]:[30,30,30]);

  let maxE=0;
  if(gt) for(let y=0;y<h;y++) for(let x=0;x<w;x++) maxE=Math.max(maxE,ent(gt[y][x]));
  const maxEp=Math.max(maxE, ...Array.from({length:h},(_,y)=>Math.max(...Array.from({length:w},(_,x)=>ent(pr[y][x])))));

  draw(document.getElementById('cvPred'),w,h,sz,(x,y)=>{
    const p=pr[y][x];
    if(mode==='entropy'){const e=ent(p)/Math.max(maxEp,0.01);return[e*248+10,(1-e)*100+10,30];}
    if(mode==='class'){const v=p[ci],c=CC[ci];return c.map(cv=>cv*v+10*(1-v));}
    if(mode==='diff'&&gt){const k=klDiv(gt[y][x],p),i=Math.min(k/2,1);return[255*i,80*(1-i),30*(1-i)];}
    const c=argmax(p),cf=p[c]; return colC(c,0.3+0.7*cf);
  });

  if(gt) draw(document.getElementById('cvGt'),w,h,sz,(x,y)=>{
    const p=gt[y][x];
    if(mode==='entropy'){const e=ent(p)/Math.max(maxE,0.01);return[e*248+10,(1-e)*100+10,30];}
    if(mode==='class'){const v=p[ci],c=CC[ci];return c.map(cv=>cv*v+10*(1-v));}
    const c=argmax(p),cf=p[c]; return colC(c,0.3+0.7*cf);
  });

  const lg=document.getElementById('legend');
  if(mode==='entropy') lg.innerHTML='<span style="font-size:11px;color:#888">Red=uncertain Green=confident</span>';
  else if(mode==='diff') lg.innerHTML='<span style="font-size:11px;color:#888">Red=high KL divergence</span>';
  else lg.innerHTML=CN.map((n,i)=>'<div class="legend-item"><div class="legend-swatch" style="background:rgb('+CC[i]+')"></div>'+n+'</div>').join('');

  const st=document.getElementById('stats');
  let dynP=0; for(let y=0;y<h;y++) for(let x=0;x<w;x++) if(ent(pr[y][x])>0.1) dynP++;
  const cards=[{val:dynP,lbl:'Uncertain cells'}];
  if(gt){
    let tE=0,tWK=0,dc=0;
    for(let y=0;y<h;y++) for(let x=0;x<w;x++){
      const e=ent(gt[y][x]); if(e<1e-6) continue; dc++; tE+=e;
      tWK+=e*klDiv(gt[y][x],pr[y][x]);
    }
    const wk=tE>0?tWK/tE:0, sc=Math.max(0,Math.min(100,100*Math.exp(-3*wk)));
    cards.push({val:sc.toFixed(1),lbl:'Score'},{val:wk.toFixed(4),lbl:'Weighted KL'},{val:dc,lbl:'Dynamic cells'});
  }
  st.innerHTML=cards.map(c=>'<div class="stat-card"><div class="val">'+c.val+'</div><div class="lbl">'+c.lbl+'</div></div>').join('');
}

function hover(e,cv) {
  const s=SEEDS[sel.value]; if(!s) return;
  const sz=parseInt(zm.value), rect=cv.getBoundingClientRect();
  const x=Math.floor((e.clientX-rect.left)/sz), y=Math.floor((e.clientY-rect.top)/sz);
  if(x<0||x>=s.width||y<0||y>=s.height) return;
  const pr=s.prediction[y][x], gt=s.ground_truth?s.ground_truth[y][x]:null;
  const ig=s.initial_grid?tc(s.initial_grid[y][x]):-1;
  let html='<span style="color:#888">Cell ('+x+','+y+')';
  if(ig>=0) html+=' initial: '+CN[ig];
  html+=' entropy: '+ent(pr).toFixed(3);
  if(gt) html+=' KL: '+klDiv(gt,pr).toFixed(4);
  html+='</span><br>';
  for(let i=0;i<6;i++){
    const c=CC[i], pv=(pr[i]*100).toFixed(1);
    html+='<div class="bar-row"><div class="bar-label" style="color:rgb('+c+')">'+CN[i]+'</div>';
    html+='<div class="bar-bg"><div class="bar-fill" style="width:'+pv+'%;background:rgb('+c+')"></div></div>';
    html+='<div class="bar-val">'+pv+'%</div>';
    if(gt){const gv=(gt[i]*100).toFixed(1);html+='<div class="bar-bg"><div class="bar-fill" style="width:'+gv+'%;background:rgb('+c+');opacity:0.4"></div></div><div class="bar-val" style="opacity:0.5">'+gv+'%</div>';}
    html+='</div>';
  }
  if(gt) html+='<div style="font-size:10px;color:#555;margin-top:2px">Left=prediction Right=ground truth</div>';
  info.innerHTML=html;
}

[document.getElementById('cvInit'),document.getElementById('cvPred'),document.getElementById('cvGt')].forEach(cv=>cv.addEventListener('mousemove',e=>hover(e,cv)));
sel.addEventListener('change',render); vm.addEventListener('change',render);
cs.addEventListener('change',render); zm.addEventListener('input',render);
render();
</script>
</body></html>`;
}

const args = Deno.args;
if (args.length === 0) {
	console.log("Usage: preview-prediction.ts <round> [seeds=0,1,2,3,4]");
	console.log("  Reads prediction bins from simulation/data/pred_r{round}_s{seed}.bin");
	Deno.exit(1);
}

const roundNum = parseInt(args[0]);
const seedList = args[1] ? args[1].split(",").map(Number) : [0, 1, 2, 3, 4];

const seeds: SeedData[] = [];
for (const seed of seedList) {
	const predPath = `${BIN_DIR}/pred_r${roundNum}_s${seed}.bin`;
	try {
		const pred = readPredictionBin(predPath);
		const gridData = readGridBin(`${BIN_DIR}/grids.bin`, roundNum, seed);
		const gt = readGroundTruthBin(`${BIN_DIR}/ground_truth.bin`, roundNum, seed);

		seeds.push({
			round: roundNum,
			seed,
			W: pred.W,
			H: pred.H,
			prediction: pred.prediction,
			ground_truth: gt,
			initial_grid: gridData?.grid ?? null,
		});
		console.log(`  [r${roundNum}.s${seed}] loaded` + (gt ? " (with ground truth)" : ""));
	} catch {
		console.log(`  [r${roundNum}.s${seed}] not found, skipping`);
	}
}

if (seeds.length === 0) { console.error("No predictions found"); Deno.exit(1); }

const html = generateHTML(seeds);
const outPath = `${DATA_DIR}/preview.html`;
await Deno.writeTextFile(outPath, html);
console.log(`\nOpen: file://${Deno.cwd()}/${outPath}`);
