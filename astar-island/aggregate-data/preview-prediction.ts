import "@std/dotenv/load";
import { NUM_CLASSES, terrainToClass, readPrediction as readPredictionBin, readGridBin, readGroundTruthBin, entropy } from "./bin-io.ts";

const BIN_DIR = "../simulation/data";
const DATA_DIR = "data";

const CLASSES = [
	{ name: "Empty/Ocean/Plains", short: "Empty", color: [200, 184, 138] },
	{ name: "Settlement", short: "Settle", color: [212, 118, 10] },
	{ name: "Port", short: "Port", color: [14, 116, 144] },
	{ name: "Ruin", short: "Ruin", color: [127, 29, 29] },
	{ name: "Forest", short: "Forest", color: [45, 90, 39] },
	{ name: "Mountain", short: "Mtn", color: [107, 114, 128] },
];

interface SeedData {
	round: number;
	seed: number;
	W: number;
	H: number;
	bucket: number[][][] | null;
	mlp: number[][][] | null;
	sim: number[][][] | null;
	cnn: number[][][] | null;
	blend: number[][][] | null;
	ground_truth: number[][][] | null;
	initial_grid: number[][] | null;
}

interface RoundData {
	round: number;
	seeds: SeedData[];
}

interface ScoreRow {
	round: number;
	seed: number;
	bucket: number | null;
	mlp: number | null;
	sim: number | null;
	cnn: number | null;
	blend: number | null;
}

function klDivergence(p: number[], q: number[]): number {
	let k = 0;
	for (let i = 0; i < p.length; i++) if (p[i] > 0) k += p[i] * Math.log(p[i] / Math.max(q[i], 1e-10));
	return k;
}

function computeScore(gt: number[][][], pred: number[][][], W: number, H: number): number {
	let tE = 0, tWK = 0;
	for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
		const e = entropy(gt[y][x]);
		if (e < 1e-6) continue;
		tE += e;
		tWK += e * klDivergence(gt[y][x], pred[y][x]);
	}
	const wk = tE > 0 ? tWK / tE : 0;
	return Math.max(0, Math.min(100, 100 * Math.exp(-3 * wk)));
}

function computeAllScores(rounds: RoundData[]): ScoreRow[] {
	const rows: ScoreRow[] = [];
	for (const r of rounds) {
		for (const s of r.seeds) {
			const gt = s.ground_truth;
			rows.push({
				round: r.round,
				seed: s.seed,
				bucket: (gt && s.bucket) ? computeScore(gt, s.bucket, s.W, s.H) : null,
				mlp: (gt && s.mlp) ? computeScore(gt, s.mlp, s.W, s.H) : null,
				sim: (gt && s.sim) ? computeScore(gt, s.sim, s.W, s.H) : null,
				cnn: (gt && s.cnn) ? computeScore(gt, s.cnn, s.W, s.H) : null,
				blend: (gt && s.blend) ? computeScore(gt, s.blend, s.W, s.H) : null,
			});
		}
	}
	return rows;
}

function scoreColor(s: number | null): string {
	if (s === null) return '#555';
	if (s >= 90) return '#38bdf8';
	if (s >= 80) return '#4ade80';
	if (s >= 65) return '#a3e635';
	if (s >= 50) return '#facc15';
	return '#f87171';
}

function generateScoreTableHTML(scores: ScoreRow[]): string {
	const roundNums = [...new Set(scores.map(s => s.round))].sort((a, b) => a - b);
	const seedNums = [...new Set(scores.map(s => s.seed))].sort((a, b) => a - b);
	const models = ['bucket', 'mlp', 'sim', 'cnn', 'blend'] as const;
	const modelLabels: Record<string, string> = { bucket: 'Bkt', mlp: 'MLP', sim: 'Sim', cnn: 'CNN', blend: 'Blend' };

	// Only show models that have at least one score
	const activeModels = models.filter(m => scores.some(s => s[m] !== null));
	const cols = activeModels.length;

	let html = `<div class="score-table-wrap"><table class="score-table">`;
	html += `<thead><tr><th></th>`;
	for (const r of roundNums) html += `<th colspan="${cols}">R${r}</th>`;
	html += `<th colspan="${cols}">AVG</th></tr>`;
	html += `<tr><th>Seed</th>`;
	const subHeaders = activeModels.map(m => `<th>${modelLabels[m]}</th>`).join('');
	for (const _ of roundNums) html += subHeaders;
	html += subHeaders + `</tr></thead><tbody>`;

	for (const seed of seedNums) {
		html += `<tr><td class="seed-cell">S${seed}</td>`;
		const seedSums: Record<string, number> = {};
		const seedCounts: Record<string, number> = {};
		for (const m of activeModels) { seedSums[m] = 0; seedCounts[m] = 0; }

		for (const r of roundNums) {
			const row = scores.find(s => s.round === r && s.seed === seed);
			for (const m of activeModels) {
				const v = row ? row[m] : null;
				if (v !== null) {
					// Find best score for this round+seed to highlight
					const allVals = activeModels.map(mm => row ? row[mm] : null).filter(x => x !== null) as number[];
					const best = Math.max(...allVals);
					const isBest = Math.abs(v - best) < 0.01;
					const color = scoreColor(v);
					html += `<td style="color:${color}">${isBest ? '<b>' : ''}${v.toFixed(1)}${isBest ? '</b>' : ''}</td>`;
					seedSums[m] += v; seedCounts[m]++;
				} else {
					html += `<td style="color:#333">-</td>`;
				}
			}
		}
		// Seed averages
		for (const m of activeModels) {
			if (seedCounts[m] > 0) {
				const avg = seedSums[m] / seedCounts[m];
				html += `<td style="color:${scoreColor(avg)}"><b>${avg.toFixed(1)}</b></td>`;
			} else {
				html += `<td>-</td>`;
			}
		}
		html += `</tr>`;
	}

	// Average row
	html += `<tr class="avg-row"><td class="seed-cell"><b>AVG</b></td>`;
	const totalSums: Record<string, number> = {};
	const totalCounts: Record<string, number> = {};
	for (const m of activeModels) { totalSums[m] = 0; totalCounts[m] = 0; }

	for (const r of roundNums) {
		for (const m of activeModels) {
			const roundScores = scores.filter(s => s.round === r && s[m] !== null);
			if (roundScores.length > 0) {
				const avg = roundScores.reduce((a, s) => a + s[m]!, 0) / roundScores.length;
				const color = scoreColor(avg);
				html += `<td style="color:${color}"><b>${avg.toFixed(1)}</b></td>`;
				totalSums[m] += avg; totalCounts[m]++;
			} else {
				html += `<td>-</td>`;
			}
		}
	}
	// Grand averages
	for (const m of activeModels) {
		if (totalCounts[m] > 0) {
			const avg = totalSums[m] / totalCounts[m];
			html += `<td style="color:${scoreColor(avg)}"><b>${avg.toFixed(1)}</b></td>`;
		} else {
			html += `<td>-</td>`;
		}
	}
	html += `</tr></tbody></table></div>`;
	return html;
}

function generateCSV(scores: ScoreRow[]): string {
	const lines = ['round,seed,bucket,mlp,sim,cnn,blend'];
	for (const s of scores) {
		lines.push([
			s.round, s.seed,
			s.bucket !== null ? s.bucket.toFixed(2) : '',
			s.mlp !== null ? s.mlp.toFixed(2) : '',
			s.sim !== null ? s.sim.toFixed(2) : '',
			s.cnn !== null ? s.cnn.toFixed(2) : '',
			s.blend !== null ? s.blend.toFixed(2) : '',
		].join(','));
	}
	return lines.join('\n') + '\n';
}

function generateHTML(rounds: RoundData[], scoreTableHTML: string): string {
	return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>A* Island - Dashboard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'SF Mono','Fira Code',monospace;background:#0a0a0a;color:#e0e0e0;padding:20px}
h1{font-size:18px;margin-bottom:4px;color:#fff}
h1 span{font-size:12px;color:#555;font-weight:normal}
.controls{display:flex;gap:12px;align-items:center;margin-bottom:16px;flex-wrap:wrap}
.controls label{font-size:12px;color:#888}
.controls select,.controls input{background:#1a1a1a;color:#e0e0e0;border:1px solid #333;padding:4px 8px;font-family:inherit;font-size:12px;border-radius:3px}
.seed-row{display:flex;gap:8px;align-items:flex-start;margin-bottom:12px;padding:8px;background:#0d0d0d;border:1px solid #1a1a1a;border-radius:4px}
.seed-label{writing-mode:vertical-rl;text-orientation:mixed;font-size:13px;font-weight:bold;color:#555;padding:4px;display:flex;align-items:center;justify-content:center;min-width:24px}
.grid-col{display:flex;flex-direction:column;align-items:center}
.grid-col .col-label{font-size:10px;color:#555;margin-bottom:2px;text-transform:uppercase;letter-spacing:0.5px}
.grid-col .score-val{font-size:10px;margin-top:2px;font-weight:bold}
.models-2x2{display:grid;grid-template-columns:1fr 1fr;gap:4px;padding:2px;border:1px solid #222;border-radius:4px;background:#080808}
.models-2x2 .grid-col .col-label{font-size:9px}
.models-2x2 .grid-col .score-val{font-size:9px}
.na{color:#333;font-size:11px;display:flex;align-items:center;justify-content:center;border:1px solid #1a1a1a;background:#0d0d0d;border-radius:2px}
canvas{image-rendering:pixelated;border:1px solid #222;cursor:crosshair}
.legend{display:flex;gap:12px;margin:8px 0;flex-wrap:wrap}
.legend-item{display:flex;align-items:center;gap:4px;font-size:11px}
.legend-swatch{width:12px;height:12px;border-radius:2px;border:1px solid #555}
.separator{width:2px;background:#222;min-height:50px;margin:0 4px;border-radius:1px}
.info-panel{background:#111;border:1px solid #333;padding:12px;font-size:12px;border-radius:4px;min-height:60px;position:sticky;bottom:0}
.bar-row{display:flex;align-items:center;gap:6px;margin:2px 0}
.bar-label{width:50px;font-size:10px}
.bar-bg{width:100px;height:10px;background:#222;border-radius:2px;overflow:hidden}
.bar-fill{height:100%;border-radius:2px}
.bar-val{font-size:10px;width:40px;text-align:right}
.info-cols{display:flex;gap:16px;flex-wrap:wrap}
.info-col{flex:1;min-width:180px}
.info-col-title{font-size:10px;color:#555;margin-bottom:4px;text-transform:uppercase;font-weight:bold}
.score-table-wrap{margin-bottom:20px;overflow-x:auto}
.score-table{border-collapse:collapse;font-size:11px;width:auto}
.score-table th,.score-table td{padding:3px 8px;border:1px solid #222;text-align:right;white-space:nowrap}
.score-table th{background:#151515;color:#888;font-weight:600;position:sticky;top:0}
.score-table td{background:#0d0d0d}
.score-table .seed-cell{text-align:left;color:#888;font-weight:600}
.score-table .avg-row td{border-top:2px solid #444}
</style>
</head>
<body>
<h1>A* Island - Dashboard <span>NM i AI 2026</span></h1>
${scoreTableHTML}
<div class="controls">
  <label>Round: <select id="roundSelect"></select></label>
  <label>View: <select id="viewMode">
    <option value="argmax">Argmax</option>
    <option value="entropy">Entropy</option>
    <option value="diff">Diff vs GT</option>
    <option value="class">Single class</option>
  </select></label>
  <label id="classWrap" style="display:none">Class: <select id="classSelect">
    <option value="0">Empty/Plains</option><option value="1">Settlement</option>
    <option value="2">Port</option><option value="3">Ruin</option>
    <option value="4">Forest</option><option value="5">Mountain</option>
  </select></label>
  <label>Zoom: <input id="zoom" type="range" min="3" max="18" value="8"></label>
  <label>Model zoom: <input id="modelZoom" type="range" min="2" max="12" value="5"></label>
</div>
<div class="legend" id="legend"></div>
<div id="seedRows"></div>
<div class="info-panel" id="info">Hover over any cell</div>
<script>
const ROUNDS=${JSON.stringify(rounds.map(r=>({round:r.round,seeds:r.seeds.map(s=>({seed:s.seed,W:s.W,H:s.H,bucket:s.bucket,mlp:s.mlp,sim:s.sim,cnn:s.cnn,blend:s.blend,ground_truth:s.ground_truth,initial_grid:s.initial_grid}))})))};
const CN=${JSON.stringify(CLASSES.map(c=>c.name))};
const CS=${JSON.stringify(CLASSES.map(c=>c.short))};
const CC=${JSON.stringify(CLASSES.map(c=>c.color))};

const roundSel=document.getElementById('roundSelect');
const vm=document.getElementById('viewMode');
const cs=document.getElementById('classSelect');
const cw=document.getElementById('classWrap');
const zm=document.getElementById('zoom');
const mzm=document.getElementById('modelZoom');
const info=document.getElementById('info');
const seedRows=document.getElementById('seedRows');

ROUNDS.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent='Round '+r.round;roundSel.appendChild(o);});

function tc(code){return code===1?1:code===2?2:code===3?3:code===4?4:code===5?5:0}
function ent(p){let h=0;for(const v of p)if(v>0)h-=v*Math.log2(v);return h}
function klDiv(p,q){let k=0;for(let i=0;i<p.length;i++)if(p[i]>0)k+=p[i]*Math.log(p[i]/Math.max(q[i],1e-10));return k}
function argmax(p){let b=0;for(let i=1;i<p.length;i++)if(p[i]>p[b])b=i;return b}
function colC(c,a){return CC[c].map(v=>Math.floor(v*a+10*(1-a)))}

function score(gt,pred,W,H){
  let tE=0,tWK=0;
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){
    const e=ent(gt[y][x]);if(e<1e-6)continue;tE+=e;
    tWK+=e*klDiv(gt[y][x],pred[y][x]);
  }
  const wk=tE>0?tWK/tE:0;
  return Math.max(0,Math.min(100,100*Math.exp(-3*wk)));
}

function scColor(s){
  if(s>=90)return'#38bdf8';if(s>=80)return'#4ade80';if(s>=65)return'#a3e635';if(s>=50)return'#facc15';return'#f87171';
}

function draw(cv,w,h,sz,fn){
  cv.width=w*sz;cv.height=h*sz;
  const ctx=cv.getContext('2d'),img=ctx.createImageData(w*sz,h*sz);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    const[r,g,b]=fn(x,y);
    for(let dy=0;dy<sz;dy++)for(let dx=0;dx<sz;dx++){
      const i=((y*sz+dy)*w*sz+x*sz+dx)*4;
      img.data[i]=r;img.data[i+1]=g;img.data[i+2]=b;img.data[i+3]=255;
    }
  }
  ctx.putImageData(img,0,0);
}

function drawProbs(cv,probs,w,h,sz,mode,ci,gt,maxE){
  draw(cv,w,h,sz,(x,y)=>{
    const p=probs[y][x];
    if(mode==='entropy'){const e=ent(p)/Math.max(maxE,0.01);return[Math.floor(e*248+10),Math.floor((1-e)*100+10),30]}
    if(mode==='class'){const v=p[ci],c=CC[ci];return c.map(cv=>Math.floor(cv*v+10*(1-v)))}
    if(mode==='diff'&&gt){const k=klDiv(gt[y][x],p),i=Math.min(k/2,1);return[Math.floor(255*i),Math.floor(80*(1-i)),Math.floor(30*(1-i))]}
    const c=argmax(p),cf=p[c];return colC(c,0.3+0.7*cf);
  });
}

let canvasMap=new Map();
let locked=false;
let lockedX=-1,lockedY=-1,lockedSeed=null;

function render(){
  const rd=ROUNDS[roundSel.value];if(!rd)return;
  locked=false;lockedSeed=null;info.style.borderColor='#333';
  const sz=parseInt(zm.value),msz=parseInt(mzm.value),mode=vm.value,ci=parseInt(cs.value);
  cw.style.display=mode==='class'?'':'none';

  const lg=document.getElementById('legend');
  if(mode==='entropy')lg.innerHTML='<span style="font-size:11px;color:#888">Red=uncertain Green=confident</span>';
  else if(mode==='diff')lg.innerHTML='<span style="font-size:11px;color:#888">Red=high KL divergence</span>';
  else lg.innerHTML=CN.map((n,i)=>'<div class="legend-item"><div class="legend-swatch" style="background:rgb('+CC[i]+')"></div>'+n+'</div>').join('');

  seedRows.innerHTML='';
  canvasMap=new Map();

  for(const s of rd.seeds){
    const row=document.createElement('div');row.className='seed-row';

    const lbl=document.createElement('div');lbl.className='seed-label';
    lbl.textContent='S'+s.seed;
    row.appendChild(lbl);

    const {W:w,H:h}=s;
    let maxE=0;
    const allPreds=[s.bucket,s.mlp,s.sim,s.cnn,s.blend,s.ground_truth];
    for(const pred of allPreds){
      if(pred)for(let y=0;y<h;y++)for(let x=0;x<w;x++)maxE=Math.max(maxE,ent(pred[y][x]));
    }

    // Column 1: Initial grid (full size)
    const initDiv=document.createElement('div');initDiv.className='grid-col';
    const initLbl=document.createElement('div');initLbl.className='col-label';initLbl.textContent='Initial';
    initDiv.appendChild(initLbl);
    if(s.initial_grid){
      const cv=document.createElement('canvas');
      draw(cv,w,h,sz,(x,y)=>CC[tc(s.initial_grid[y][x])]);
      initDiv.appendChild(cv);
      canvasMap.set(cv,{seed:s,source:'initial'});
    } else {
      const na=document.createElement('div');na.className='na';na.style.width=w*sz+'px';na.style.height=h*sz+'px';na.textContent='N/A';
      initDiv.appendChild(na);
    }
    row.appendChild(initDiv);

    // Separator
    const sep1=document.createElement('div');sep1.className='separator';row.appendChild(sep1);

    // Column 2: 2x2 model grid (smaller zoom)
    const models2x2=document.createElement('div');models2x2.className='models-2x2';
    const modelDefs=[
      {label:'Bucket',data:s.bucket,key:'bucket'},
      {label:'MLP',data:s.mlp,key:'mlp'},
      {label:'Sim',data:s.sim,key:'sim'},
      {label:'CNN',data:s.cnn,key:'cnn'},
    ];
    for(const md of modelDefs){
      const col=document.createElement('div');col.className='grid-col';
      const clbl=document.createElement('div');clbl.className='col-label';clbl.textContent=md.label;
      col.appendChild(clbl);
      if(md.data){
        const cv=document.createElement('canvas');
        drawProbs(cv,md.data,w,h,msz,mode,ci,s.ground_truth,maxE);
        col.appendChild(cv);
        canvasMap.set(cv,{seed:s,source:md.key});
        if(s.ground_truth){
          const sc=score(s.ground_truth,md.data,w,h);
          const sv=document.createElement('div');sv.className='score-val';
          sv.style.color=scColor(sc);
          sv.textContent=sc.toFixed(1);
          col.appendChild(sv);
        }
      } else {
        const na=document.createElement('div');na.className='na';na.style.width=w*msz+'px';na.style.height=h*msz+'px';na.textContent='N/A';
        col.appendChild(na);
      }
      models2x2.appendChild(col);
    }
    row.appendChild(models2x2);

    // Separator
    const sep2=document.createElement('div');sep2.className='separator';row.appendChild(sep2);

    // Column 3: Blend / Submission (full size)
    const blendDiv=document.createElement('div');blendDiv.className='grid-col';
    const blendLbl=document.createElement('div');blendLbl.className='col-label';blendLbl.textContent='Submission';
    blendDiv.appendChild(blendLbl);
    if(s.blend){
      const cv=document.createElement('canvas');
      drawProbs(cv,s.blend,w,h,sz,mode,ci,s.ground_truth,maxE);
      blendDiv.appendChild(cv);
      canvasMap.set(cv,{seed:s,source:'blend'});
      if(s.ground_truth){
        const sc=score(s.ground_truth,s.blend,w,h);
        const sv=document.createElement('div');sv.className='score-val';
        sv.style.color=scColor(sc);
        sv.textContent=sc.toFixed(1);
        blendDiv.appendChild(sv);
      }
    } else {
      const na=document.createElement('div');na.className='na';na.style.width=w*sz+'px';na.style.height=h*sz+'px';na.textContent='N/A';
      blendDiv.appendChild(na);
    }
    row.appendChild(blendDiv);

    // Separator
    const sep3=document.createElement('div');sep3.className='separator';row.appendChild(sep3);

    // Column 4: Ground Truth (full size)
    const gtDiv=document.createElement('div');gtDiv.className='grid-col';
    const gtLbl=document.createElement('div');gtLbl.className='col-label';gtLbl.textContent='Ground Truth';
    gtDiv.appendChild(gtLbl);
    if(s.ground_truth){
      const cv=document.createElement('canvas');
      drawProbs(cv,s.ground_truth,w,h,sz,mode,ci,s.ground_truth,maxE);
      gtDiv.appendChild(cv);
      canvasMap.set(cv,{seed:s,source:'ground_truth'});
    } else {
      const na=document.createElement('div');na.className='na';na.style.width=w*sz+'px';na.style.height=h*sz+'px';na.textContent='No GT';
      gtDiv.appendChild(na);
    }
    row.appendChild(gtDiv);

    // Dynamic cell count
    if(s.ground_truth){
      let dynCells=0;
      for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(ent(s.ground_truth[y][x])>1e-6)dynCells++;
      const statDiv=document.createElement('div');
      statDiv.style.cssText='font-size:10px;color:#555;writing-mode:vertical-rl;padding:4px;display:flex;align-items:center';
      statDiv.textContent=dynCells+' dyn';
      row.appendChild(statDiv);
    }

    seedRows.appendChild(row);
  }

  document.querySelectorAll('canvas').forEach(cv=>{
    cv.addEventListener('mousemove',e=>hover(e,cv));
    cv.addEventListener('mouseleave',()=>{if(!locked)info.innerHTML='Hover over any cell';});
    cv.addEventListener('click',e=>{
      const meta=canvasMap.get(cv);if(!meta)return;
      const s=meta.seed;
      const sz=cv.width/s.W,rect=cv.getBoundingClientRect();
      const x=Math.floor((e.clientX-rect.left)/sz),y=Math.floor((e.clientY-rect.top)/sz);
      if(x<0||x>=s.W||y<0||y>=s.H)return;
      if(locked&&lockedX===x&&lockedY===y&&lockedSeed===s){
        locked=false;info.style.borderColor='#333';
      } else {
        locked=true;lockedX=x;lockedY=y;lockedSeed=s;
        info.style.borderColor='#38bdf8';
        showCellInfo(s,x,y);
      }
    });
  });
}

function showCellInfo(s,x,y){
  const ig=s.initial_grid?tc(s.initial_grid[y][x]):-1;
  let html='<span style="color:#888">Cell ('+x+','+y+') R'+s.round+'.S'+s.seed;
  if(ig>=0)html+=' | initial: <b>'+CS[ig]+'</b>';
  if(locked)html+=' | <span style="color:#38bdf8">LOCKED</span> <span style="color:#555">(click to unlock, Esc to clear)</span>';
  html+='</span><br><div class="info-cols">';

  const sources=[
    {name:'Bucket',data:s.bucket},
    {name:'MLP',data:s.mlp},
    {name:'Sim',data:s.sim},
    {name:'CNN',data:s.cnn},
    {name:'Submission',data:s.blend},
    {name:'Ground Truth',data:s.ground_truth},
  ];

  for(const src of sources){
    if(!src.data)continue;
    const p=src.data[y][x];
    const h=ent(p);
    let col='<div class="info-col"><div class="info-col-title">'+src.name+' <span style="color:#666">(H:'+h.toFixed(2)+')</span></div>';
    for(let i=0;i<6;i++){
      const c=CC[i],pv=(p[i]*100).toFixed(1);
      if(p[i]<0.005)continue; // skip near-zero
      col+='<div class="bar-row"><div class="bar-label" style="color:rgb('+c+')">'+CS[i]+'</div>';
      col+='<div class="bar-bg"><div class="bar-fill" style="width:'+pv+'%;background:rgb('+c+')"></div></div>';
      col+='<div class="bar-val">'+pv+'%</div></div>';
    }
    if(s.ground_truth&&src.name!=='Ground Truth'){
      const gt_p=s.ground_truth[y][x];
      const cellEntropy=ent(gt_p);
      const kl=klDiv(gt_p,p);
      col+='<div style="font-size:9px;color:#666;margin-top:2px">KL: '+kl.toFixed(3)+' | wKL: '+(cellEntropy*kl).toFixed(3)+'</div>';
    }
    col+='</div>';
    html+=col;
  }
  html+='</div>';
  info.innerHTML=html;
}

function hover(e,cv){
  if(locked)return;
  const meta=canvasMap.get(cv);if(!meta)return;
  const s=meta.seed;
  const sz=cv.width/s.W,rect=cv.getBoundingClientRect();
  const x=Math.floor((e.clientX-rect.left)/sz),y=Math.floor((e.clientY-rect.top)/sz);
  if(x<0||x>=s.W||y<0||y>=s.H)return;
  showCellInfo(s,x,y);
}

document.addEventListener('keydown',e=>{
  if(e.key==='Escape'&&locked){locked=false;info.style.borderColor='#333';info.innerHTML='Hover over any cell';}
  if(e.key==='ArrowLeft'){const v=parseInt(roundSel.value);if(v>0){roundSel.value=v-1;render();}}
  if(e.key==='ArrowRight'){const v=parseInt(roundSel.value);if(v<ROUNDS.length-1){roundSel.value=v+1;render();}}
});

roundSel.addEventListener('change',render);
vm.addEventListener('change',render);
cs.addEventListener('change',render);
zm.addEventListener('input',render);
mzm.addEventListener('input',render);
render();
</script>
</body></html>`;
}

// Load all rounds
const gridsPath = `${BIN_DIR}/grids.bin`;
const gtPath = `${BIN_DIR}/ground_truth.bin`;

const allRounds: RoundData[] = [];

// Discover which rounds have grids
const gridsData = Deno.readFileSync(gridsPath);
const gridsView = new DataView(gridsData.buffer);
let offset = 6;
const gridCount = gridsView.getUint32(offset, true); offset += 4;
const roundSet = new Set<number>();
for (let i = 0; i < gridCount; i++) {
	const round = gridsView.getInt32(offset, true); offset += 4;
	offset += 4; // seed
	const W = gridsView.getInt32(offset, true); offset += 4;
	const H = gridsView.getInt32(offset, true); offset += 4;
	roundSet.add(round);
	offset += W * H * 4;
}

const roundNums = [...roundSet].sort((a, b) => a - b);
console.log(`Found rounds: ${roundNums.join(", ")}`);

for (const roundNum of roundNums) {
	const seeds: SeedData[] = [];
	for (let seed = 0; seed < 5; seed++) {
		const gridData = readGridBin(gridsPath, roundNum, seed);
		if (!gridData) continue;

		let bucket: number[][][] | null = null;
		try { bucket = readPredictionBin(`${BIN_DIR}/pred_bucket_r${roundNum}_s${seed}.bin`).prediction; } catch { /* */ }

		let mlp: number[][][] | null = null;
		try { mlp = readPredictionBin(`${BIN_DIR}/pred_mlp_r${roundNum}_s${seed}.bin`).prediction; } catch { /* */ }

		let sim: number[][][] | null = null;
		try { sim = readPredictionBin(`${BIN_DIR}/pred_sim_r${roundNum}_s${seed}.bin`).prediction; } catch { /* */ }

		let cnn: number[][][] | null = null;
		try { cnn = readPredictionBin(`${BIN_DIR}/pred_cnn_r${roundNum}_s${seed}.bin`).prediction; } catch { /* */ }

		// Blend = submission file (pred_r{N}_s{S}.bin)
		let blend: number[][][] | null = null;
		try { blend = readPredictionBin(`${BIN_DIR}/pred_r${roundNum}_s${seed}.bin`).prediction; } catch {
			try { blend = readPredictionBin(`${BIN_DIR}/pred_blend_r${roundNum}_s${seed}.bin`).prediction; } catch { /* */ }
		}

		const gt = readGroundTruthBin(gtPath, roundNum, seed);

		seeds.push({
			round: roundNum, seed,
			W: gridData.W, H: gridData.H,
			bucket, mlp, sim, cnn, blend,
			ground_truth: gt,
			initial_grid: gridData.grid,
		});
	}

	if (seeds.length > 0) {
		allRounds.push({ round: roundNum, seeds });
		const has = (k: keyof SeedData) => seeds.some(s => s[k]);
		console.log(`  Round ${roundNum}: ${seeds.length} seeds [bkt:${has('bucket')} mlp:${has('mlp')} sim:${has('sim')} cnn:${has('cnn')} blend:${has('blend')} gt:${has('ground_truth')}]`);
	}
}

const scores = computeAllScores(allRounds);
const scoreTableHTML = generateScoreTableHTML(scores);
const html = generateHTML(allRounds, scoreTableHTML);
const outPath = `${DATA_DIR}/preview.html`;
await Deno.writeTextFile(outPath, html);

const csvPath = `${DATA_DIR}/scores.csv`;
await Deno.writeTextFile(csvPath, generateCSV(scores));

// Print summary table to console
console.log('');
const models = ['bucket', 'mlp', 'sim', 'cnn', 'blend'] as const;
const activeModels = models.filter(m => scores.some(s => s[m] !== null));
const roundNumsForTable = [...new Set(scores.map(s => s.round))].sort((a, b) => a - b);
const header = `Seed  ${roundNumsForTable.map(r => `R${r}`).join('     ')}    AVG`;
console.log(header);
console.log('-'.repeat(header.length + 20));

const seedNums = [...new Set(scores.map(s => s.seed))].sort((a, b) => a - b);
for (const seed of seedNums) {
	let line = `S${seed}    `;
	for (const r of roundNumsForTable) {
		const row = scores.find(s => s.round === r && s.seed === seed);
		const vals = activeModels.map(m => row && row[m] !== null ? row[m]!.toFixed(0) : '-');
		line += vals.join('/') + '  ';
	}
	// Seed average
	const avgVals = activeModels.map(m => {
		const ss = scores.filter(s => s.seed === seed && s[m] !== null);
		return ss.length > 0 ? (ss.reduce((a, s) => a + s[m]!, 0) / ss.length).toFixed(0) : '-';
	});
	line += avgVals.join('/');
	console.log(line);
}

// Grand average
let avgLine = 'AVG   ';
for (const r of roundNumsForTable) {
	const vals = activeModels.map(m => {
		const rs = scores.filter(s => s.round === r && s[m] !== null);
		return rs.length > 0 ? (rs.reduce((a, s) => a + s[m]!, 0) / rs.length).toFixed(0) : '-';
	});
	avgLine += vals.join('/') + '  ';
}
const grandVals = activeModels.map(m => {
	const all = scores.filter(s => s[m] !== null);
	return all.length > 0 ? (all.reduce((a, s) => a + s[m]!, 0) / all.length).toFixed(0) : '-';
});
avgLine += grandVals.join('/');
console.log(avgLine);
console.log(`\nModels: ${activeModels.join('/')}`);
console.log(`\nOpen: file://${Deno.cwd()}/${outPath}`);
