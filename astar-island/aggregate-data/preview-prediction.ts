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
			for (let c = 0; c < NUM_CLASSES; c++) { cell.push(view.getFloat32(offset, true)); offset += 4; }
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
			for (let y = 0; y < H; y++) { const row: number[] = []; for (let x = 0; x < W; x++) { row.push(view.getInt32(offset, true)); offset += 4; } grid.push(row); }
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
				for (let y = 0; y < H; y++) { const row: number[][] = []; for (let x = 0; x < W; x++) { const cell: number[] = []; for (let c = 0; c < NUM_CLASSES; c++) { cell.push(view.getFloat32(offset, true)); offset += 4; } row.push(cell); } gt.push(row); }
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
	bucket: number[][][] | null;
	mlp: number[][][] | null;
	sim: number[][][] | null;
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
	diff: number | null;
}

function entropy(p: number[]): number {
	let h = 0;
	for (const v of p) if (v > 0) h -= v * Math.log2(v);
	return h;
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
			const bucketScore = (s.ground_truth && s.bucket) ? computeScore(s.ground_truth, s.bucket, s.W, s.H) : null;
			const mlpScore = (s.ground_truth && s.mlp) ? computeScore(s.ground_truth, s.mlp, s.W, s.H) : null;
			const simScore = (s.ground_truth && s.sim) ? computeScore(s.ground_truth, s.sim, s.W, s.H) : null;
			const diff = (bucketScore !== null && mlpScore !== null) ? mlpScore - bucketScore : null;
			rows.push({ round: r.round, seed: s.seed, bucket: bucketScore, mlp: mlpScore, sim: simScore, diff });
		}
	}
	return rows;
}

function generateScoreTableHTML(scores: ScoreRow[], rounds: RoundData[]): string {
	const roundNums = [...new Set(scores.map(s => s.round))].sort((a, b) => a - b);
	const seedNums = [...new Set(scores.map(s => s.seed))].sort((a, b) => a - b);
	const showSim = scores.some(s => s.sim !== null);
	const cols = showSim ? 4 : 3;

	let html = `<div class="score-table-wrap"><table class="score-table">`;
	html += `<thead><tr><th></th>`;
	for (const r of roundNums) html += `<th colspan="${cols}">R${r}</th>`;
	html += `<th colspan="${cols}">AVG</th></tr>`;
	html += `<tr><th>Seed</th>`;
	const subHeaders = showSim ? '<th>Bucket</th><th>MLP</th><th>Sim</th><th>Diff</th>' : '<th>Bucket</th><th>MLP</th><th>Diff</th>';
	for (const _ of roundNums) html += subHeaders;
	html += subHeaders + `</tr></thead><tbody>`;

	for (const seed of seedNums) {
		html += `<tr><td class="seed-cell">S${seed}</td>`;
		let seedBucketSum = 0, seedMlpSum = 0, seedSimSum = 0, seedCount = 0, seedSimCount = 0;
		for (const r of roundNums) {
			const row = scores.find(s => s.round === r && s.seed === seed);
			if (row) {
				const bStr = row.bucket !== null ? row.bucket.toFixed(1) : '—';
				const mStr = row.mlp !== null ? row.mlp.toFixed(1) : '—';
				const dStr = row.diff !== null ? (row.diff >= 0 ? '+' : '') + row.diff.toFixed(1) : '—';
				const dClass = row.diff !== null ? (row.diff > 0 ? 'pos' : row.diff < 0 ? 'neg' : '') : '';
				html += `<td>${bStr}</td><td>${mStr}</td>`;
				if (showSim) {
					const sStr = row.sim !== null ? row.sim.toFixed(1) : '—';
					html += `<td>${sStr}</td>`;
					if (row.sim !== null) { seedSimSum += row.sim; seedSimCount++; }
				}
				html += `<td class="${dClass}">${dStr}</td>`;
				if (row.bucket !== null && row.mlp !== null) {
					seedBucketSum += row.bucket; seedMlpSum += row.mlp; seedCount++;
				}
			} else {
				html += `<td>—</td><td>—</td>`;
				if (showSim) html += `<td>—</td>`;
				html += `<td>—</td>`;
			}
		}
		if (seedCount > 0) {
			const ab = seedBucketSum / seedCount, am = seedMlpSum / seedCount, ad = am - ab;
			const dClass = ad > 0 ? 'pos' : ad < 0 ? 'neg' : '';
			html += `<td><b>${ab.toFixed(1)}</b></td><td><b>${am.toFixed(1)}</b></td>`;
			if (showSim) html += `<td><b>${seedSimCount > 0 ? (seedSimSum / seedSimCount).toFixed(1) : '—'}</b></td>`;
			html += `<td class="${dClass}"><b>${(ad >= 0 ? '+' : '') + ad.toFixed(1)}</b></td>`;
		} else {
			html += `<td>—</td><td>—</td>`;
			if (showSim) html += `<td>—</td>`;
			html += `<td>—</td>`;
		}
		html += `</tr>`;
	}

	html += `<tr class="avg-row"><td class="seed-cell"><b>AVG</b></td>`;
	let totalBucket = 0, totalMlp = 0, totalSim = 0, totalCount = 0, totalSimCount = 0;
	for (const r of roundNums) {
		const roundScores = scores.filter(s => s.round === r && s.bucket !== null && s.mlp !== null);
		if (roundScores.length > 0) {
			const ab = roundScores.reduce((a, s) => a + s.bucket!, 0) / roundScores.length;
			const am = roundScores.reduce((a, s) => a + s.mlp!, 0) / roundScores.length;
			const ad = am - ab;
			const dClass = ad > 0 ? 'pos' : ad < 0 ? 'neg' : '';
			html += `<td><b>${ab.toFixed(1)}</b></td><td><b>${am.toFixed(1)}</b></td>`;
			if (showSim) {
				const simScores = roundScores.filter(s => s.sim !== null);
				if (simScores.length > 0) {
					const as2 = simScores.reduce((a, s) => a + s.sim!, 0) / simScores.length;
					html += `<td><b>${as2.toFixed(1)}</b></td>`;
					totalSim += as2; totalSimCount++;
				} else { html += `<td>—</td>`; }
			}
			html += `<td class="${dClass}"><b>${(ad >= 0 ? '+' : '') + ad.toFixed(1)}</b></td>`;
			totalBucket += ab; totalMlp += am; totalCount++;
		} else {
			html += `<td>—</td><td>—</td>`;
			if (showSim) html += `<td>—</td>`;
			html += `<td>—</td>`;
		}
	}
	if (totalCount > 0) {
		const ab = totalBucket / totalCount, am = totalMlp / totalCount, ad = am - ab;
		const dClass = ad > 0 ? 'pos' : ad < 0 ? 'neg' : '';
		html += `<td><b>${ab.toFixed(1)}</b></td><td><b>${am.toFixed(1)}</b></td>`;
		if (showSim) html += `<td><b>${totalSimCount > 0 ? (totalSim / totalSimCount).toFixed(1) : '—'}</b></td>`;
		html += `<td class="${dClass}"><b>${(ad >= 0 ? '+' : '') + ad.toFixed(1)}</b></td>`;
	} else {
		html += `<td>—</td><td>—</td>`;
		if (showSim) html += `<td>—</td>`;
		html += `<td>—</td>`;
	}
	html += `</tr></tbody></table></div>`;
	return html;
}

function generateCSV(scores: ScoreRow[]): string {
	const lines = ['round,seed,bucket,mlp,sim,diff'];
	for (const s of scores) {
		lines.push([
			s.round,
			s.seed,
			s.bucket !== null ? s.bucket.toFixed(2) : '',
			s.mlp !== null ? s.mlp.toFixed(2) : '',
			s.sim !== null ? s.sim.toFixed(2) : '',
			s.diff !== null ? s.diff.toFixed(2) : '',
		].join(','));
	}
	// Add round averages
	const roundNums = [...new Set(scores.map(s => s.round))].sort((a, b) => a - b);
	for (const r of roundNums) {
		const rs = scores.filter(s => s.round === r);
		const bScores = rs.filter(s => s.bucket !== null).map(s => s.bucket!);
		const mScores = rs.filter(s => s.mlp !== null).map(s => s.mlp!);
		const bAvg = bScores.length > 0 ? (bScores.reduce((a, b) => a + b, 0) / bScores.length) : null;
		const mAvg = mScores.length > 0 ? (mScores.reduce((a, b) => a + b, 0) / mScores.length) : null;
		const diff = (bAvg !== null && mAvg !== null) ? mAvg - bAvg : null;
		lines.push([
			r,
			'avg',
			bAvg !== null ? bAvg.toFixed(2) : '',
			mAvg !== null ? mAvg.toFixed(2) : '',
			diff !== null ? diff.toFixed(2) : '',
		].join(','));
	}
	// Grand average
	const allB = scores.filter(s => s.bucket !== null).map(s => s.bucket!);
	const allM = scores.filter(s => s.mlp !== null).map(s => s.mlp!);
	const gB = allB.length > 0 ? allB.reduce((a, b) => a + b, 0) / allB.length : null;
	const gM = allM.length > 0 ? allM.reduce((a, b) => a + b, 0) / allM.length : null;
	const gD = (gB !== null && gM !== null) ? gM - gB : null;
	lines.push([
		'all',
		'avg',
		gB !== null ? gB.toFixed(2) : '',
		gM !== null ? gM.toFixed(2) : '',
		gD !== null ? gD.toFixed(2) : '',
	].join(','));
	return lines.join('\n') + '\n';
}

function generateHTML(rounds: RoundData[], scoreTableHTML: string): string {
	return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Astar Island — Dashboard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'SF Mono','Fira Code',monospace;background:#0a0a0a;color:#e0e0e0;padding:20px}
h1{font-size:18px;margin-bottom:12px;color:#fff}
.controls{display:flex;gap:12px;align-items:center;margin-bottom:16px;flex-wrap:wrap}
.controls label{font-size:12px;color:#888}
.controls select,.controls input{background:#1a1a1a;color:#e0e0e0;border:1px solid #333;padding:4px 8px;font-family:inherit;font-size:12px;border-radius:3px}
.seed-row{display:flex;gap:8px;align-items:flex-start;margin-bottom:12px;padding:8px;background:#0d0d0d;border:1px solid #1a1a1a;border-radius:4px}
.seed-label{writing-mode:vertical-rl;text-orientation:mixed;font-size:13px;font-weight:bold;color:#555;padding:4px;display:flex;align-items:center;justify-content:center;min-width:24px}
.seed-score{font-size:11px;color:#888;margin-top:4px}
.grid-col{display:flex;flex-direction:column;align-items:center}
.grid-col .col-label{font-size:10px;color:#555;margin-bottom:2px;text-transform:uppercase;letter-spacing:0.5px}
.grid-col .score-val{font-size:10px;margin-top:2px}
.na{color:#333;font-size:11px;display:flex;align-items:center;justify-content:center;border:1px solid #1a1a1a;background:#0d0d0d;border-radius:2px}
canvas{image-rendering:pixelated;border:1px solid #222;cursor:crosshair}
.legend{display:flex;gap:12px;margin:8px 0;flex-wrap:wrap}
.legend-item{display:flex;align-items:center;gap:4px;font-size:11px}
.legend-swatch{width:12px;height:12px;border-radius:2px;border:1px solid #555}
.info-panel{background:#111;border:1px solid #333;padding:12px;font-size:12px;border-radius:4px;min-height:60px;position:sticky;bottom:0}
.bar-row{display:flex;align-items:center;gap:6px;margin:2px 0}
.bar-label{width:110px;font-size:11px}
.bar-bg{width:120px;height:10px;background:#222;border-radius:2px;overflow:hidden}
.bar-fill{height:100%;border-radius:2px}
.bar-val{font-size:10px;width:40px;text-align:right}
.info-cols{display:flex;gap:24px}
.info-col{flex:1}
.info-col-title{font-size:10px;color:#555;margin-bottom:4px;text-transform:uppercase}
.score-table-wrap{margin-bottom:20px;overflow-x:auto}
.score-table{border-collapse:collapse;font-size:11px;width:auto}
.score-table th,.score-table td{padding:3px 8px;border:1px solid #222;text-align:right;white-space:nowrap}
.score-table th{background:#151515;color:#888;font-weight:600;position:sticky;top:0}
.score-table td{background:#0d0d0d}
.score-table .seed-cell{text-align:left;color:#888;font-weight:600}
.score-table .pos{color:#4ade80}
.score-table .neg{color:#f87171}
.score-table .avg-row td{border-top:2px solid #444}
</style>
</head>
<body>
<h1>Astar Island — Dashboard</h1>
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
  <label>Zoom: <input id="zoom" type="range" min="4" max="18" value="10"></label>
</div>
<div class="legend" id="legend"></div>
<div id="seedRows"></div>
<div class="info-panel" id="info">Hover over any cell</div>
<script>
const ROUNDS=${JSON.stringify(rounds.map(r=>({round:r.round,seeds:r.seeds.map(s=>({seed:s.seed,W:s.W,H:s.H,bucket:s.bucket,mlp:s.mlp,ground_truth:s.ground_truth,initial_grid:s.initial_grid}))})))};
const CN=${JSON.stringify(CLASSES.map(c=>c.name))};
const CC=${JSON.stringify(CLASSES.map(c=>c.color))};

const roundSel=document.getElementById('roundSelect');
const vm=document.getElementById('viewMode');
const cs=document.getElementById('classSelect');
const cw=document.getElementById('classWrap');
const zm=document.getElementById('zoom');
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

function render(){
  const rd=ROUNDS[roundSel.value];if(!rd)return;
  const sz=parseInt(zm.value),mode=vm.value,ci=parseInt(cs.value);
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
    if(s.ground_truth)for(let y=0;y<h;y++)for(let x=0;x<w;x++)maxE=Math.max(maxE,ent(s.ground_truth[y][x]));
    if(s.bucket)for(let y=0;y<h;y++)for(let x=0;x<w;x++)maxE=Math.max(maxE,ent(s.bucket[y][x]));
    if(s.mlp)for(let y=0;y<h;y++)for(let x=0;x<w;x++)maxE=Math.max(maxE,ent(s.mlp[y][x]));

    const cols=[
      {label:'Initial',type:'grid'},
      {label:'Bucket',type:'probs',data:s.bucket},
      {label:'Neural Net',type:'probs',data:s.mlp},
      {label:'Ground Truth',type:'probs',data:s.ground_truth},
    ];

    for(const col of cols){
      const div=document.createElement('div');div.className='grid-col';
      const clbl=document.createElement('div');clbl.className='col-label';clbl.textContent=col.label;
      div.appendChild(clbl);

      if(col.type==='grid'){
        if(s.initial_grid){
          const cv=document.createElement('canvas');
          draw(cv,w,h,sz,(x,y)=>CC[tc(s.initial_grid[y][x])]);
          div.appendChild(cv);
          canvasMap.set(cv,{seed:s,source:'initial'});
        } else {
          const na=document.createElement('div');na.className='na';na.style.width=w*sz+'px';na.style.height=h*sz+'px';na.textContent='N/A';
          div.appendChild(na);
        }
      } else {
        if(col.data){
          const cv=document.createElement('canvas');
          drawProbs(cv,col.data,w,h,sz,mode,ci,s.ground_truth,maxE);
          div.appendChild(cv);
          canvasMap.set(cv,{seed:s,source:col.label.toLowerCase().replace(' ','_')});
          if(s.ground_truth&&col.data!==s.ground_truth){
            const sc=score(s.ground_truth,col.data,w,h);
            const sv=document.createElement('div');sv.className='score-val';
            sv.style.color=sc>75?'#4ade80':sc>50?'#facc15':'#f87171';
            sv.textContent=sc.toFixed(1);
            div.appendChild(sv);
          }
        } else {
          const na=document.createElement('div');na.className='na';na.style.width=w*sz+'px';na.style.height=h*sz+'px';na.textContent='N/A';
          div.appendChild(na);
        }
      }
      row.appendChild(div);
    }
    seedRows.appendChild(row);
  }

  document.querySelectorAll('canvas').forEach(cv=>{
    cv.addEventListener('mousemove',e=>hover(e,cv));
  });
}

function hover(e,cv){
  const meta=canvasMap.get(cv);if(!meta)return;
  const s=meta.seed;
  const sz=parseInt(zm.value),rect=cv.getBoundingClientRect();
  const x=Math.floor((e.clientX-rect.left)/sz),y=Math.floor((e.clientY-rect.top)/sz);
  if(x<0||x>=s.W||y<0||y>=s.H)return;

  const ig=s.initial_grid?tc(s.initial_grid[y][x]):-1;
  let html='<span style="color:#888">Cell ('+x+','+y+') seed '+s.seed;
  if(ig>=0)html+=' — initial: '+CN[ig];
  html+='</span><br><div class="info-cols">';

  const sources=[
    {name:'Bucket',data:s.bucket},
    {name:'Neural Net',data:s.mlp},
    {name:'Ground Truth',data:s.ground_truth},
  ];

  for(const src of sources){
    if(!src.data)continue;
    const p=src.data[y][x];
    let col='<div class="info-col"><div class="info-col-title">'+src.name+' (H:'+ent(p).toFixed(2)+')</div>';
    for(let i=0;i<6;i++){
      const c=CC[i],pv=(p[i]*100).toFixed(1);
      col+='<div class="bar-row"><div class="bar-label" style="color:rgb('+c+')">'+CN[i]+'</div>';
      col+='<div class="bar-bg"><div class="bar-fill" style="width:'+pv+'%;background:rgb('+c+')"></div></div>';
      col+='<div class="bar-val">'+pv+'%</div></div>';
    }
    col+='</div>';
    html+=col;
  }
  html+='</div>';
  info.innerHTML=html;
}

roundSel.addEventListener('change',render);
vm.addEventListener('change',render);
cs.addEventListener('change',render);
zm.addEventListener('input',render);
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
		try { bucket = readPredictionBin(`${BIN_DIR}/pred_bucket_r${roundNum}_s${seed}.bin`).prediction; } catch {
			try { bucket = readPredictionBin(`${BIN_DIR}/pred_r${roundNum}_s${seed}.bin`).prediction; } catch { /* */ }
		}

		let mlp: number[][][] | null = null;
		try { mlp = readPredictionBin(`${BIN_DIR}/pred_mlp_r${roundNum}_s${seed}.bin`).prediction; } catch { /* */ }

		let sim: number[][][] | null = null;
		try { sim = readPredictionBin(`${BIN_DIR}/pred_sim_r${roundNum}_s${seed}.bin`).prediction; } catch { /* */ }

		const gt = readGroundTruthBin(gtPath, roundNum, seed);

		seeds.push({
			round: roundNum,
			seed,
			W: gridData.W,
			H: gridData.H,
			bucket,
			mlp,
			sim,
			ground_truth: gt,
			initial_grid: gridData.grid,
		});
	}

	if (seeds.length > 0) {
		allRounds.push({ round: roundNum, seeds });
		const hasBucket = seeds.some(s => s.bucket);
		const hasMlp = seeds.some(s => s.mlp);
		const hasSim = seeds.some(s => s.sim);
		const hasGt = seeds.some(s => s.ground_truth);
		console.log(`  Round ${roundNum}: ${seeds.length} seeds [bucket:${hasBucket} mlp:${hasMlp} sim:${hasSim} gt:${hasGt}]`);
	}
}

const scores = computeAllScores(allRounds);
const scoreTableHTML = generateScoreTableHTML(scores, allRounds);
const html = generateHTML(allRounds, scoreTableHTML);
const outPath = `${DATA_DIR}/preview.html`;
await Deno.writeTextFile(outPath, html);

const csvPath = `${DATA_DIR}/scores.csv`;
await Deno.writeTextFile(csvPath, generateCSV(scores));
console.log(`\nScores CSV: ${csvPath}`);

// Print summary table to console
console.log('');
const roundNumsForTable = [...new Set(scores.map(s => s.round))].sort((a, b) => a - b);
const hasSim = scores.some(s => s.sim !== null);
const colLabel = hasSim ? 'B/M/S' : 'B/M/D';
const header = ['Seed', ...roundNumsForTable.map(r => `R${r} ${colLabel}`), `AVG ${colLabel}`];
console.log(header.join('  '));
console.log('-'.repeat(header.join('  ').length));
const seedNums = [...new Set(scores.map(s => s.seed))].sort((a, b) => a - b);
for (const seed of seedNums) {
	const parts = [`S${seed}  `];
	for (const r of roundNumsForTable) {
		const row = scores.find(s => s.round === r && s.seed === seed);
		if (row && row.bucket !== null && row.mlp !== null) {
			if (hasSim) {
				parts.push(`${row.bucket!.toFixed(0)}/${row.mlp!.toFixed(0)}/${row.sim !== null ? row.sim!.toFixed(0) : '-'}`);
			} else {
				parts.push(`${row.bucket!.toFixed(0)}/${row.mlp!.toFixed(0)}/${row.diff! >= 0 ? '+' : ''}${row.diff!.toFixed(0)}`);
			}
		} else {
			parts.push('—');
		}
	}
	const seedScores = scores.filter(s => s.seed === seed && s.bucket !== null && s.mlp !== null);
	if (seedScores.length > 0) {
		const ab = seedScores.reduce((a, s) => a + s.bucket!, 0) / seedScores.length;
		const am = seedScores.reduce((a, s) => a + s.mlp!, 0) / seedScores.length;
		if (hasSim) {
			const simScores = seedScores.filter(s => s.sim !== null);
			const as2 = simScores.length > 0 ? simScores.reduce((a, s) => a + s.sim!, 0) / simScores.length : 0;
			parts.push(`${ab.toFixed(0)}/${am.toFixed(0)}/${simScores.length > 0 ? as2.toFixed(0) : '-'}`);
		} else {
			const ad = am - ab;
			parts.push(`${ab.toFixed(0)}/${am.toFixed(0)}/${ad >= 0 ? '+' : ''}${ad.toFixed(0)}`);
		}
	}
	console.log(parts.join('  '));
}

console.log(`\nOpen: file://${Deno.cwd()}/${outPath}`);
